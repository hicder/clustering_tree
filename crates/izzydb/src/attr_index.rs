//! Cluster-aware inverted attribute index (`cf_attr_clusters`, `cf_attr_postings`, `cf_attr_zones`)
//! and two-phase filter planner/evaluator.

use std::collections::{HashMap, HashSet};

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, DocId, LocalId, NodeId};
use izzydb_core::error::Result;
use izzydb_core::filter::{Filter, FtsPredicateKind};
use izzydb_core::schema::{AttributeValue, Attributes, CollectionSchema, ScalarValue};
use izzydb_storage::codec::{
    LeafPostingView, decode_attribute_entries, encode_present_cluster_key,
    encode_scalar_cluster_key, encode_scalar_val_bytes, extract_index_keys,
    extract_index_keys_from_entries, make_posting_key, ordered_type_tag,
};
use izzydb_storage::node::LeafNodeView;
use izzydb_storage::{ByteRange, DBPinnableSlice, Storage, TxContext};

use crate::fts_index::FtsIndex;
use crate::tree::ClusterDelta;

/// Compact slot-indexed bitmask for a single leaf node.
/// Stores up to 256 slots inline without heap allocation.
#[derive(Debug, Clone, Default)]
pub struct LeafMask {
    inline: [u64; 4],
    overflow: Vec<u64>,
    len: usize,
}

impl LeafMask {
    #[inline]
    pub fn empty(word_count: usize) -> Self {
        if word_count <= 4 {
            Self {
                inline: [0u64; 4],
                overflow: Vec::new(),
                len: word_count,
            }
        } else {
            Self {
                inline: [0u64; 4],
                overflow: vec![0u64; word_count],
                len: word_count,
            }
        }
    }

    #[inline]
    pub fn from_live_mask_bytes(bytes: &[u8], slot_cap: usize) -> Self {
        let word_count = slot_cap.div_ceil(64);
        let mut mask = Self::empty(word_count);
        let dst = mask.as_mut_slice();
        for (i, chunk) in bytes.chunks_exact(8).take(word_count).enumerate() {
            dst[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        let rem = slot_cap % 64;
        if rem != 0 && word_count > 0 {
            dst[word_count - 1] &= (1u64 << rem) - 1;
        }
        mask
    }

    #[inline]
    pub fn from_words_padded(src: &[u64], word_count: usize) -> Self {
        let mut mask = Self::empty(word_count);
        let copy_len = src.len().min(word_count);
        mask.as_mut_slice()[..copy_len].copy_from_slice(&src[..copy_len]);
        mask
    }

    #[inline]
    pub fn as_slice(&self) -> &[u64] {
        if self.len <= 4 {
            &self.inline[..self.len]
        } else {
            &self.overflow
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        if self.len <= 4 {
            &mut self.inline[..self.len]
        } else {
            &mut self.overflow
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.as_slice().iter().all(|&w| w == 0)
    }

    #[inline]
    pub fn and_assign(&mut self, other: &Self) {
        let a = self.as_mut_slice();
        let b = other.as_slice();
        let n = a.len().min(b.len());
        for i in 0..n {
            a[i] &= b[i];
        }
        for v in &mut a[n..] {
            *v = 0;
        }
    }

    #[inline]
    pub fn or_assign(&mut self, other: &Self) {
        let a = self.as_mut_slice();
        let b = other.as_slice();
        let n = a.len().min(b.len());
        for i in 0..n {
            a[i] |= b[i];
        }
    }

    #[inline]
    pub fn and_not_assign(&mut self, other: &Self) {
        let a = self.as_mut_slice();
        let b = other.as_slice();
        let n = a.len().min(b.len());
        for i in 0..n {
            a[i] &= !b[i];
        }
    }

    #[inline]
    pub fn and_live_mask_bytes(&mut self, bytes: &[u8]) {
        let a = self.as_mut_slice();
        let avail = bytes.len() / 8;
        let n = a.len().min(avail);
        for i in 0..n {
            let w = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
            a[i] &= w;
        }
        for v in &mut a[n..] {
            *v = 0;
        }
    }

    #[inline]
    pub fn for_each_set_bit(&self, slot_cap: usize, mut f: impl FnMut(usize)) {
        for (wi, &word) in self.as_slice().iter().enumerate() {
            let mut w = word;
            let base = wi * 64;
            while w != 0 {
                let tz = w.trailing_zeros() as usize;
                let slot = base + tz;
                if slot < slot_cap {
                    f(slot);
                }
                w &= w - 1;
            }
        }
    }
}

/// Conservative cluster-level pruning result from `cf_attr_clusters` and `cf_attr_zones`.
#[derive(Debug, Clone)]
pub enum ClusterCandidates {
    /// Filter cannot safely prune clusters (e.g., top-level `Not(...)`).
    All,
    /// Guaranteed superset of leaf `NodeId`s that may contain matching docs.
    Subset(RoaringBitmap),
}

impl ClusterCandidates {
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Subset(bm) if bm.is_empty())
    }
}

/// Prepared filter plan containing cluster-level pruning candidates and resolved index keys
/// so per-leaf evaluation can batch all `cf_attr_postings` reads in one `multi_get_cf`.
#[derive(Debug, Clone)]
pub struct PreparedFilter {
    pub cluster_candidates: ClusterCandidates,
    root: PreparedNode,
}

#[derive(Debug, Clone)]
enum PreparedNode {
    True,
    False,
    And(Vec<PreparedNode>),
    Or(Vec<PreparedNode>),
    Not(Box<PreparedNode>),
    /// Union of `(cluster_key, cluster_bitmap)` entries.
    Keys(Vec<(Vec<u8>, RoaringBitmap)>),
    /// Range predicate over cluster-first `cf_attr_postings`, pruned via `cf_attr_zones`.
    Range {
        attr_id: AttrId,
        type_tag: u8,
        lo: Option<(Vec<u8>, bool)>,
        hi: Option<(Vec<u8>, bool)>,
        cands: RoaringBitmap,
    },
    /// Pre-resolved per-leaf slot word bitmasks (used for FTS predicates).
    LeafSlotMasks(HashMap<NodeId, Vec<u64>>),
}

/// Cluster-aware inverted attribute index backed by `cf_attr_clusters` and `cf_attr_postings`.
#[derive(Debug, Default, Clone, Copy)]
pub struct AttributeIndex;

impl AttributeIndex {
    /// Index a newly inserted document's attributes at `(leaf_id, local_id)` inside `tx`.
    pub fn insert_doc(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        attrs: &Attributes,
        leaf_id: NodeId,
        local_id: LocalId,
    ) -> Result<()> {
        let keys = extract_index_keys(schema, attrs)?;
        for cluster_key in keys {
            tx.mutate_attr_posting(&cluster_key, leaf_id, |bm| {
                bm.insert(local_id);
            })?;
        }
        Ok(())
    }

    /// Index pre-decoded `(AttrId, AttributeValue)` entries at `(leaf_id, local_id)` inside `tx`.
    pub fn insert_doc_entries(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        entries: &[(AttrId, AttributeValue)],
        leaf_id: NodeId,
        local_id: LocalId,
    ) -> Result<()> {
        let keys = extract_index_keys_from_entries(schema, entries)?;
        for cluster_key in keys {
            tx.mutate_attr_posting(&cluster_key, leaf_id, |bm| {
                bm.insert(local_id);
            })?;
        }
        Ok(())
    }

    /// Remove a deleted/replaced document's attributes at `(leaf_id, local_id)` inside `tx`.
    pub fn remove_doc_entries(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        entries: &[(AttrId, AttributeValue)],
        leaf_id: NodeId,
        local_id: LocalId,
    ) -> Result<()> {
        let keys = extract_index_keys_from_entries(schema, entries)?;
        for cluster_key in keys {
            tx.mutate_attr_posting(&cluster_key, leaf_id, |bm| {
                bm.remove(local_id);
            })?;
        }
        Ok(())
    }

    /// Update a document's attribute postings in place at `(leaf_id, local_id)` by computing
    /// the symmetric difference between `old_entries` and `new_attrs`.
    pub fn diff_doc_in_place(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        old_entries: &[(AttrId, AttributeValue)],
        new_attrs: &Attributes,
        leaf_id: NodeId,
        local_id: LocalId,
    ) -> Result<()> {
        let old_keys = extract_index_keys_from_entries(schema, old_entries)?;
        let new_keys = extract_index_keys(schema, new_attrs)?;
        diff_sorted_keys(tx, &old_keys, &new_keys, leaf_id, local_id)
    }

    /// Update only the patched subset of attributes in place at `(leaf_id, local_id)` so
    /// untouched filterable attributes incur zero index work.
    pub fn diff_patched_in_place(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        old_entries: &[(AttrId, AttributeValue)],
        patch_attrs: &Attributes,
        leaf_id: NodeId,
        local_id: LocalId,
    ) -> Result<()> {
        if patch_attrs.is_empty() {
            return Ok(());
        }
        let mut patched_ids: HashSet<AttrId> = HashSet::with_capacity(patch_attrs.len());
        for name in patch_attrs.keys() {
            if let Some(entry) = schema.get_by_name(name)
                && entry.filterable
            {
                patched_ids.insert(entry.id);
            }
        }
        if patched_ids.is_empty() {
            return Ok(());
        }
        let filtered_old: Vec<(AttrId, AttributeValue)> = old_entries
            .iter()
            .filter(|(id, _)| patched_ids.contains(id))
            .cloned()
            .collect();
        let old_keys = extract_index_keys_from_entries(schema, &filtered_old)?;
        let new_keys = extract_index_keys(schema, patch_attrs)?;
        diff_sorted_keys(tx, &old_keys, &new_keys, leaf_id, local_id)
    }

    /// Apply cluster moves / leaf renumberings (`ClusterDelta`) to `AttributeIndex` inside `tx`.
    ///
    /// Coalesces multi-step renumberings and applies all net removals before net additions so
    /// renumbered `local_id` collisions in the same leaf never cancel out newly added entries.
    pub fn apply_cluster_delta(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        delta: &ClusterDelta,
    ) -> Result<()> {
        if delta.is_empty() {
            return Ok(());
        }

        let (net_removed, net_added) = delta.net_changes();
        if net_removed.is_empty() && net_added.is_empty() {
            return Ok(());
        }

        let mut doc_ids: Vec<DocId> = Vec::with_capacity(net_removed.len() + net_added.len());
        for &(doc_id, _, _) in &net_removed {
            doc_ids.push(doc_id);
        }
        for &(doc_id, _, _) in &net_added {
            doc_ids.push(doc_id);
        }
        doc_ids.sort_unstable();
        doc_ids.dedup();

        let raw_docs = tx.multi_get_doc_bytes(&doc_ids)?;
        let mut keys_by_doc: HashMap<DocId, Vec<Vec<u8>>> = HashMap::with_capacity(doc_ids.len());
        for (&doc_id, opt_bytes) in doc_ids.iter().zip(raw_docs.iter()) {
            let Some(bytes) = opt_bytes else {
                // Standalone ClusteringTree mode (without Collection docs) has no cf_docs entries.
                continue;
            };
            let entries = decode_attribute_entries(bytes)?;
            let keys = extract_index_keys_from_entries(schema, &entries)?;
            keys_by_doc.insert(doc_id, keys);
        }

        for &(doc_id, leaf_id, local_id) in &net_removed {
            if let Some(keys) = keys_by_doc.get(&doc_id) {
                for cluster_key in keys {
                    tx.mutate_attr_posting(cluster_key, leaf_id, |bm| {
                        bm.remove(local_id);
                    })?;
                }
            }
        }

        for &(doc_id, leaf_id, local_id) in &net_added {
            if let Some(keys) = keys_by_doc.get(&doc_id) {
                for cluster_key in keys {
                    tx.mutate_attr_posting(cluster_key, leaf_id, |bm| {
                        bm.insert(local_id);
                    })?;
                }
            }
        }

        Ok(())
    }

    /// Evaluate cluster-level candidates from `cf_attr_clusters` (and `FtsIndex` if needed)
    /// and build a [`PreparedFilter`] for batched exact per-leaf evaluation.
    pub fn prepare_filter(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        filter: &Filter,
    ) -> Result<PreparedFilter> {
        self.prepare_filter_with_fts(storage, schema, filter, None)
    }

    /// Evaluate cluster-level candidates with optional [`FtsIndex`] support for FTS predicates.
    pub fn prepare_filter_with_fts(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        filter: &Filter,
        fts_index: Option<&FtsIndex>,
    ) -> Result<PreparedFilter> {
        let (cluster_candidates, root) = eval_cluster_node(storage, schema, filter, fts_index)?;
        Ok(PreparedFilter {
            cluster_candidates,
            root,
        })
    }

    /// Evaluate exact matching slot bitmasks for a batch of `(leaf_id, Option<&LeafNodeView>)`
    /// using a single `batched_multi_get_cf` on `cf_attr_postings` plus prefix-bounded range scans.
    pub fn eval_leaves_masks<'a>(
        &self,
        storage: &Storage,
        prepared: &PreparedFilter,
        leaves: &[(NodeId, Option<&LeafNodeView<'a>>)],
    ) -> Result<Vec<LeafMask>> {
        if leaves.is_empty() {
            return Ok(Vec::new());
        }

        let mut leaf_ids: Vec<NodeId> = Vec::with_capacity(leaves.len());
        for &(leaf_id, ref view_opt) in leaves {
            if let Some(view) = view_opt
                && view.live_count > 0
            {
                leaf_ids.push(leaf_id);
            }
        }
        if leaf_ids.is_empty() {
            return Ok(vec![LeafMask::default(); leaves.len()]);
        }

        let mut needed_keys: Vec<Vec<u8>> = Vec::new();
        collect_posting_keys(&prepared.root, &leaf_ids, &mut needed_keys);
        needed_keys.sort_unstable();
        needed_keys.dedup();

        let fetched = storage.multi_get_attr_posting_slices(&needed_keys)?;

        let mut out = Vec::with_capacity(leaves.len());
        for &(leaf_id, ref view_opt) in leaves {
            let Some(view) = view_opt else {
                out.push(LeafMask::default());
                continue;
            };
            if view.live_count == 0 {
                out.push(LeafMask::default());
                continue;
            }
            out.push(eval_prepared_leaf(
                storage,
                &prepared.root,
                leaf_id,
                view,
                &needed_keys,
                &fetched,
            )?);
        }
        Ok(out)
    }
}

fn diff_sorted_keys(
    tx: &mut TxContext<'_>,
    old_keys: &[Vec<u8>],
    new_keys: &[Vec<u8>],
    leaf_id: NodeId,
    local_id: LocalId,
) -> Result<()> {
    let (mut i, mut j) = (0usize, 0usize);
    while i < old_keys.len() && j < new_keys.len() {
        match old_keys[i].cmp(&new_keys[j]) {
            std::cmp::Ordering::Less => {
                tx.mutate_attr_posting(&old_keys[i], leaf_id, |bm| {
                    bm.remove(local_id);
                })?;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                tx.mutate_attr_posting(&new_keys[j], leaf_id, |bm| {
                    bm.insert(local_id);
                })?;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    while i < old_keys.len() {
        tx.mutate_attr_posting(&old_keys[i], leaf_id, |bm| {
            bm.remove(local_id);
        })?;
        i += 1;
    }
    while j < new_keys.len() {
        tx.mutate_attr_posting(&new_keys[j], leaf_id, |bm| {
            bm.insert(local_id);
        })?;
        j += 1;
    }
    Ok(())
}

fn eval_cluster_node(
    storage: &Storage,
    schema: &CollectionSchema,
    filter: &Filter,
    fts_index: Option<&FtsIndex>,
) -> Result<(ClusterCandidates, PreparedNode)> {
    match filter {
        Filter::And(children) => eval_and_cluster_node(storage, schema, children, fts_index),
        Filter::Or(children) => eval_or_cluster_node(storage, schema, children, fts_index),
        Filter::Not(child) => {
            let (_cands, pnode) = eval_cluster_node(storage, schema, child, fts_index)?;
            Ok((ClusterCandidates::All, PreparedNode::Not(Box::new(pnode))))
        }
        Filter::Eq(attr, val) => {
            let attr_id = schema.get_by_name(attr).unwrap().id;
            if matches!(val, ScalarValue::Null) {
                let pnode = eval_is_null_node(storage, attr_id)?;
                return Ok((ClusterCandidates::All, pnode));
            }
            let key = encode_scalar_cluster_key(attr_id, val)?;
            eval_single_key_cluster_node(storage, key)
        }
        Filter::NotEq(attr, val) => {
            let eq = Filter::Eq(attr.clone(), val.clone());
            let (_cands, pnode) = eval_cluster_node(storage, schema, &eq, fts_index)?;
            if matches!(val, ScalarValue::Null) {
                // NotEq(attr, Null) == IsNotNull(attr), which CAN prune clusters.
                let attr_id = schema.get_by_name(attr).unwrap().id;
                return eval_single_key_cluster_node(storage, encode_present_cluster_key(attr_id));
            }
            Ok((ClusterCandidates::All, PreparedNode::Not(Box::new(pnode))))
        }
        Filter::In(attr, vals) | Filter::ContainsAny(attr, vals) => {
            let attr_id = schema.get_by_name(attr).unwrap().id;
            eval_in_cluster_node(storage, attr_id, vals)
        }
        Filter::NotIn(attr, vals) => {
            let inner = Filter::In(attr.clone(), vals.clone());
            let (_cands, pnode) = eval_cluster_node(storage, schema, &inner, fts_index)?;
            Ok((ClusterCandidates::All, PreparedNode::Not(Box::new(pnode))))
        }
        Filter::Contains(attr, val) => {
            let attr_id = schema.get_by_name(attr).unwrap().id;
            eval_single_key_cluster_node(storage, encode_scalar_cluster_key(attr_id, val)?)
        }
        Filter::Lt(attr, val)
        | Filter::Lte(attr, val)
        | Filter::Gt(attr, val)
        | Filter::Gte(attr, val) => eval_range_cluster_node(storage, schema, filter, attr, val),
        Filter::IsNull(attr) => {
            let attr_id = schema.get_by_name(attr).unwrap().id;
            let pnode = eval_is_null_node(storage, attr_id)?;
            Ok((ClusterCandidates::All, pnode))
        }
        Filter::IsNotNull(attr) => {
            let attr_id = schema.get_by_name(attr).unwrap().id;
            eval_single_key_cluster_node(storage, encode_present_cluster_key(attr_id))
        }
        other => {
            let (kind, attr, query, last_as_prefix) = other
                .as_fts_predicate()
                .expect("all non-FTS Filter variants handled above");
            eval_fts_cluster_node(storage, schema, fts_index, kind, attr, query, last_as_prefix)
        }
    }
}

fn empty_cluster_result() -> (ClusterCandidates, PreparedNode) {
    (
        ClusterCandidates::Subset(RoaringBitmap::new()),
        PreparedNode::False,
    )
}

fn eval_single_key_cluster_node(
    storage: &Storage,
    key: Vec<u8>,
) -> Result<(ClusterCandidates, PreparedNode)> {
    match storage.get_attr_cluster_bitmap(&key)? {
        Some(bm) if !bm.is_empty() => Ok((
            ClusterCandidates::Subset(bm.clone()),
            PreparedNode::Keys(vec![(key, bm)]),
        )),
        _ => Ok(empty_cluster_result()),
    }
}

fn eval_and_cluster_node(
    storage: &Storage,
    schema: &CollectionSchema,
    children: &[Filter],
    fts_index: Option<&FtsIndex>,
) -> Result<(ClusterCandidates, PreparedNode)> {
    if children.is_empty() {
        return Ok((ClusterCandidates::All, PreparedNode::True));
    }
    let mut combined: Option<RoaringBitmap> = None;
    let mut non_range = Vec::with_capacity(children.len());
    let mut ranges: Vec<PreparedNode> = Vec::new();

    for child in children {
        let (cands, pnode) = eval_cluster_node(storage, schema, child, fts_index)?;
        if let ClusterCandidates::Subset(bm) = cands {
            if bm.is_empty() {
                return Ok(empty_cluster_result());
            }
            combined = Some(match combined {
                Some(mut cur) => {
                    cur &= &bm;
                    if cur.is_empty() {
                        return Ok(empty_cluster_result());
                    }
                    cur
                }
                None => bm,
            });
        }
        match pnode {
            PreparedNode::True => {}
            PreparedNode::False => return Ok(empty_cluster_result()),
            PreparedNode::And(sub) => {
                for item in sub {
                    push_and_child(item, &mut non_range, &mut ranges);
                }
            }
            other => push_and_child(other, &mut non_range, &mut ranges),
        }
    }

    // Validate coalesced ranges (e.g., Gte(200) && Lte(400) or contradictory bounds).
    for r in &ranges {
        if let PreparedNode::Range {
            lo: Some((lo_b, lo_inc)),
            hi: Some((hi_b, hi_inc)),
            cands,
            ..
        } = r
            && (cands.is_empty() || lo_b > hi_b || (lo_b == hi_b && (!*lo_inc || !*hi_inc)))
        {
            return Ok(empty_cluster_result());
        }
    }
    // Evaluate non-range predicates before range scans so cheap equality/FTS masks short-circuit ranges.
    non_range.extend(ranges);
    let cands = match combined {
        Some(bm) => ClusterCandidates::Subset(bm),
        None => ClusterCandidates::All,
    };
    let node = if non_range.is_empty() {
        PreparedNode::True
    } else if non_range.len() == 1 {
        non_range.pop().unwrap()
    } else {
        PreparedNode::And(non_range)
    };
    Ok((cands, node))
}

fn eval_or_cluster_node(
    storage: &Storage,
    schema: &CollectionSchema,
    children: &[Filter],
    fts_index: Option<&FtsIndex>,
) -> Result<(ClusterCandidates, PreparedNode)> {
    if children.is_empty() {
        return Ok(empty_cluster_result());
    }
    let mut is_all = false;
    let mut union_bm = RoaringBitmap::new();
    let mut prep_children = Vec::with_capacity(children.len());
    for child in children {
        let (cands, pnode) = eval_cluster_node(storage, schema, child, fts_index)?;
        match cands {
            ClusterCandidates::All => is_all = true,
            ClusterCandidates::Subset(bm) if !is_all => union_bm |= &bm,
            ClusterCandidates::Subset(_) => {}
        }
        prep_children.push(pnode);
    }
    let cands = if is_all {
        ClusterCandidates::All
    } else {
        ClusterCandidates::Subset(union_bm)
    };
    Ok((cands, PreparedNode::Or(prep_children)))
}

fn eval_in_cluster_node(
    storage: &Storage,
    attr_id: AttrId,
    vals: &[ScalarValue],
) -> Result<(ClusterCandidates, PreparedNode)> {
    let mut has_null = false;
    let mut keys: Vec<Vec<u8>> = Vec::with_capacity(vals.len());
    for val in vals {
        if matches!(val, ScalarValue::Null) {
            has_null = true;
        } else {
            keys.push(encode_scalar_cluster_key(attr_id, val)?);
        }
    }
    keys.sort_unstable();
    keys.dedup();

    let bitmaps = storage.multi_get_attr_cluster_bitmaps(&keys)?;
    let mut union_bm = RoaringBitmap::new();
    let mut items = Vec::new();
    for (key, opt_bm) in keys.into_iter().zip(bitmaps) {
        if let Some(bm) = opt_bm
            && !bm.is_empty()
        {
            union_bm |= &bm;
            items.push((key, bm));
        }
    }
    let non_null_node = if items.is_empty() {
        PreparedNode::False
    } else {
        PreparedNode::Keys(items)
    };

    if has_null {
        let null_node = eval_is_null_node(storage, attr_id)?;
        Ok((
            ClusterCandidates::All,
            PreparedNode::Or(vec![non_null_node, null_node]),
        ))
    } else {
        Ok((ClusterCandidates::Subset(union_bm), non_null_node))
    }
}

fn eval_range_cluster_node(
    storage: &Storage,
    schema: &CollectionSchema,
    filter: &Filter,
    attr: &str,
    val: &ScalarValue,
) -> Result<(ClusterCandidates, PreparedNode)> {
    let entry = schema.get_by_name(attr).unwrap();
    let type_tag = ordered_type_tag(entry.attr_type)?;
    let mut bound_bytes = Vec::with_capacity(16);
    let val_tag = encode_scalar_val_bytes(val, &mut bound_bytes)?;
    if val_tag != type_tag {
        return Ok(empty_cluster_result());
    }

    let zones = storage.get_attr_zones_cached()?;
    let mut bm = RoaringBitmap::new();
    if let Some(zmap) = zones.get(&(entry.id, type_tag)) {
        for (&leaf_id, (min_b, max_b)) in zmap {
            let overlaps = match filter {
                Filter::Lt(_, _) => min_b.as_slice() < bound_bytes.as_slice(),
                Filter::Lte(_, _) => min_b.as_slice() <= bound_bytes.as_slice(),
                Filter::Gt(_, _) => max_b.as_slice() > bound_bytes.as_slice(),
                Filter::Gte(_, _) => max_b.as_slice() >= bound_bytes.as_slice(),
                _ => unreachable!(),
            };
            if overlaps {
                bm.insert(leaf_id);
            }
        }
    }
    if bm.is_empty() {
        return Ok((ClusterCandidates::Subset(bm), PreparedNode::False));
    }
    let (lo, hi) = match filter {
        Filter::Lt(_, _) => (None, Some((bound_bytes, false))),
        Filter::Lte(_, _) => (None, Some((bound_bytes, true))),
        Filter::Gt(_, _) => (Some((bound_bytes, false)), None),
        Filter::Gte(_, _) => (Some((bound_bytes, true)), None),
        _ => unreachable!(),
    };
    Ok((
        ClusterCandidates::Subset(bm.clone()),
        PreparedNode::Range {
            attr_id: entry.id,
            type_tag,
            lo,
            hi,
            cands: bm,
        },
    ))
}

fn eval_fts_cluster_node(
    storage: &Storage,
    schema: &CollectionSchema,
    fts_index: Option<&FtsIndex>,
    kind: FtsPredicateKind,
    attr: &str,
    query: &izzydb_core::filter::TokenQuery,
    last_as_prefix: bool,
) -> Result<(ClusterCandidates, PreparedNode)> {
    let default_fts;
    let fts = match fts_index {
        Some(idx) => idx,
        None => {
            default_fts = FtsIndex::new();
            &default_fts
        }
    };
    let fts_bm = fts.eval_fts_predicate(storage, schema, kind, attr, query, last_as_prefix)?;
    if fts_bm.is_empty() {
        return Ok((
            ClusterCandidates::Subset(RoaringBitmap::new()),
            PreparedNode::False,
        ));
    }
    let (leaf_bm, per_leaf) = storage.resolve_fts_bitmap_to_leaf_masks(&fts_bm)?;
    if leaf_bm.is_empty() {
        Ok((
            ClusterCandidates::Subset(RoaringBitmap::new()),
            PreparedNode::False,
        ))
    } else {
        Ok((
            ClusterCandidates::Subset(leaf_bm),
            PreparedNode::LeafSlotMasks(per_leaf),
        ))
    }
}

fn push_and_child(
    node: PreparedNode,
    non_range: &mut Vec<PreparedNode>,
    ranges: &mut Vec<PreparedNode>,
) {
    if let PreparedNode::Range {
        attr_id,
        type_tag,
        lo,
        hi,
        cands,
    } = node
    {
        for existing in ranges.iter_mut() {
            if let PreparedNode::Range {
                attr_id: ea,
                type_tag: et,
                lo: elo,
                hi: ehi,
                cands: ecands,
            } = existing
                && *ea == attr_id
                && *et == type_tag
            {
                *elo = match (elo.take(), lo) {
                    (Some((a, ai)), Some((b, bi))) => match a.cmp(&b) {
                        std::cmp::Ordering::Greater => Some((a, ai)),
                        std::cmp::Ordering::Less => Some((b, bi)),
                        std::cmp::Ordering::Equal => Some((a, ai && bi)),
                    },
                    (Some(x), None) | (None, Some(x)) => Some(x),
                    (None, None) => None,
                };
                *ehi = match (ehi.take(), hi) {
                    (Some((a, ai)), Some((b, bi))) => match a.cmp(&b) {
                        std::cmp::Ordering::Less => Some((a, ai)),
                        std::cmp::Ordering::Greater => Some((b, bi)),
                        std::cmp::Ordering::Equal => Some((a, ai && bi)),
                    },
                    (Some(x), None) | (None, Some(x)) => Some(x),
                    (None, None) => None,
                };
                *ecands &= &cands;
                return;
            }
        }
        ranges.push(PreparedNode::Range {
            attr_id,
            type_tag,
            lo,
            hi,
            cands,
        });
    } else {
        non_range.push(node);
    }
}

fn eval_is_null_node(storage: &Storage, attr_id: AttrId) -> Result<PreparedNode> {
    let key = encode_present_cluster_key(attr_id);
    match storage.get_attr_cluster_bitmap(&key)? {
        Some(bm) if !bm.is_empty() => Ok(PreparedNode::Not(Box::new(PreparedNode::Keys(vec![(
            key, bm,
        )])))),
        _ => Ok(PreparedNode::True),
    }
}

fn collect_posting_keys(node: &PreparedNode, leaf_ids: &[NodeId], out: &mut Vec<Vec<u8>>) {
    match node {
        PreparedNode::True
        | PreparedNode::False
        | PreparedNode::Range { .. }
        | PreparedNode::LeafSlotMasks(_) => {}
        PreparedNode::And(children) | PreparedNode::Or(children) => {
            for c in children {
                collect_posting_keys(c, leaf_ids, out);
            }
        }
        PreparedNode::Not(child) => collect_posting_keys(child, leaf_ids, out),
        PreparedNode::Keys(items) => {
            for (cluster_key, cluster_bm) in items {
                for &leaf_id in leaf_ids {
                    if cluster_bm.contains(leaf_id) {
                        out.push(make_posting_key(cluster_key, leaf_id));
                    }
                }
            }
        }
    }
}

fn eval_prepared_leaf(
    storage: &Storage,
    node: &PreparedNode,
    leaf_id: NodeId,
    view: &LeafNodeView<'_>,
    needed_keys: &[Vec<u8>],
    fetched: &[Option<DBPinnableSlice<'_>>],
) -> Result<LeafMask> {
    let word_count = view.slot_cap.div_ceil(64);
    match node {
        PreparedNode::True => Ok(LeafMask::from_live_mask_bytes(
            view.live_mask_bytes,
            view.slot_cap,
        )),
        PreparedNode::False => Ok(LeafMask::empty(word_count)),
        PreparedNode::LeafSlotMasks(map) => match map.get(&leaf_id) {
            Some(words) => Ok(LeafMask::from_words_padded(words, word_count)),
            None => Ok(LeafMask::empty(word_count)),
        },
        PreparedNode::Range {
            attr_id,
            type_tag,
            lo,
            hi,
            cands,
        } => {
            let mut acc = LeafMask::empty(word_count);
            if cands.contains(leaf_id) {
                storage.scan_attr_posting_range_into_words(
                    *attr_id,
                    *type_tag,
                    leaf_id,
                    ByteRange {
                        lo: lo.as_ref().map(|(b, inc)| (b.as_slice(), *inc)),
                        hi: hi.as_ref().map(|(b, inc)| (b.as_slice(), *inc)),
                    },
                    acc.as_mut_slice(),
                )?;
            }
            Ok(acc)
        }
        PreparedNode::And(children) => {
            let mut acc: Option<LeafMask> = None;
            for child in children {
                let mask = eval_prepared_leaf(storage, child, leaf_id, view, needed_keys, fetched)?;
                if mask.is_empty() {
                    return Ok(LeafMask::empty(word_count));
                }
                acc = Some(match acc {
                    Some(mut cur) => {
                        cur.and_assign(&mask);
                        if cur.is_empty() {
                            return Ok(LeafMask::empty(word_count));
                        }
                        cur
                    }
                    None => mask,
                });
            }
            Ok(acc.unwrap_or_else(|| {
                LeafMask::from_live_mask_bytes(view.live_mask_bytes, view.slot_cap)
            }))
        }
        PreparedNode::Or(children) => {
            let mut acc = LeafMask::empty(word_count);
            for child in children {
                let mask = eval_prepared_leaf(storage, child, leaf_id, view, needed_keys, fetched)?;
                acc.or_assign(&mask);
            }
            Ok(acc)
        }
        PreparedNode::Not(child) => {
            let sub = eval_prepared_leaf(storage, child, leaf_id, view, needed_keys, fetched)?;
            let mut out = LeafMask::from_live_mask_bytes(view.live_mask_bytes, view.slot_cap);
            out.and_not_assign(&sub);
            Ok(out)
        }
        PreparedNode::Keys(items) => {
            let mut acc = LeafMask::empty(word_count);
            for (cluster_key, cluster_bm) in items {
                if cluster_bm.contains(leaf_id) {
                    let pkey = make_posting_key(cluster_key, leaf_id);
                    if let Ok(idx) = needed_keys.binary_search(&pkey)
                        && let Some(slice) = &fetched[idx]
                    {
                        LeafPostingView::parse(slice)?.or_into_words(acc.as_mut_slice())?;
                    }
                }
            }
            Ok(acc)
        }
    }
}
