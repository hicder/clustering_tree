use std::collections::HashMap;

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, DocId, LocalId, NodeId};
use izzydb_core::error::Result;
use izzydb_core::filter::Filter;
use izzydb_core::schema::{AttributeValue, Attributes, CollectionSchema, ScalarValue};
use izzydb_storage::codec::{
    decode_attribute_entries, encode_attr_prefix, encode_present_cluster_key,
    encode_scalar_cluster_key, extract_index_keys, extract_index_keys_from_entries,
    make_posting_key, ordered_type_tag,
};
use izzydb_storage::node::LeafNodeView;
use izzydb_storage::{Storage, TxContext};

use crate::tree::ClusterDelta;

/// Conservative cluster-level pruning result from `cf_attr_clusters`.
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

    /// Apply cluster moves / leaf renumberings (`ClusterDelta`) to `AttributeIndex` inside `tx`.
    ///
    /// Applies all removals before additions so renumbered `local_id` collisions in the same leaf
    /// never cancel out newly added entries.
    pub fn apply_cluster_delta(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        delta: &ClusterDelta,
    ) -> Result<()> {
        if delta.is_empty() {
            return Ok(());
        }

        let mut doc_ids: Vec<DocId> = Vec::with_capacity(delta.removed.len() + delta.added.len());
        for &(doc_id, _, _) in &delta.removed {
            doc_ids.push(doc_id);
        }
        for &(doc_id, _, _) in &delta.added {
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

        for &(doc_id, leaf_id, local_id) in &delta.removed {
            if let Some(keys) = keys_by_doc.get(&doc_id) {
                for cluster_key in keys {
                    tx.mutate_attr_posting(cluster_key, leaf_id, |bm| {
                        bm.remove(local_id);
                    })?;
                }
            }
        }

        for &(doc_id, leaf_id, local_id) in &delta.added {
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

    /// Evaluate cluster-level candidates from `cf_attr_clusters` and build a [`PreparedFilter`]
    /// for batched exact per-leaf evaluation.
    pub fn prepare_filter(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        filter: &Filter,
    ) -> Result<PreparedFilter> {
        let (cluster_candidates, root) = eval_cluster_node(storage, schema, filter)?;
        Ok(PreparedFilter {
            cluster_candidates,
            root,
        })
    }

    /// Evaluate exact matching `local_id` bitmaps for a batch of `(leaf_id, Option<&LeafNodeView>)`
    /// using a single `batched_multi_get_cf` on `cf_attr_postings`.
    pub fn eval_leaves_masks<'a>(
        &self,
        storage: &Storage,
        prepared: &PreparedFilter,
        leaves: &[(NodeId, Option<&LeafNodeView<'a>>)],
    ) -> Result<Vec<RoaringBitmap>> {
        if leaves.is_empty() {
            return Ok(Vec::new());
        }

        let mut leaf_ids: Vec<NodeId> = Vec::with_capacity(leaves.len());
        for (idx, &(leaf_id, ref view_opt)) in leaves.iter().enumerate() {
            if let Some(view) = view_opt
                && view.entry_count > 0
            {
                leaf_ids.push(leaf_id);
            } else {
                let _ = idx;
            }
        }
        if leaf_ids.is_empty() {
            return Ok(vec![RoaringBitmap::new(); leaves.len()]);
        }

        let mut needed_keys: Vec<Vec<u8>> = Vec::new();
        collect_posting_keys(&prepared.root, &leaf_ids, &mut needed_keys);
        needed_keys.sort_unstable();
        needed_keys.dedup();

        let fetched = storage.multi_get_attr_posting_bitmaps(&needed_keys)?;
        let mut postings: HashMap<Vec<u8>, RoaringBitmap> = HashMap::with_capacity(needed_keys.len());
        for (key, opt_bm) in needed_keys.into_iter().zip(fetched) {
            if let Some(bm) = opt_bm {
                postings.insert(key, bm);
            }
        }

        let needs_all_mask = node_needs_complement(&prepared.root);
        let mut out = Vec::with_capacity(leaves.len());
        for &(leaf_id, ref view_opt) in leaves {
            let Some(view) = view_opt else {
                out.push(RoaringBitmap::new());
                continue;
            };
            if view.entry_count == 0 {
                out.push(RoaringBitmap::new());
                continue;
            }
            let all_local = if needs_all_mask {
                let mut bm = RoaringBitmap::new();
                for slot in 0..view.entry_count {
                    bm.insert(view.local_id(slot));
                }
                Some(bm)
            } else {
                None
            };
            out.push(eval_prepared_leaf(
                &prepared.root,
                leaf_id,
                &postings,
                all_local.as_ref(),
            ));
        }
        Ok(out)
    }
}

fn eval_cluster_node(
    storage: &Storage,
    schema: &CollectionSchema,
    filter: &Filter,
) -> Result<(ClusterCandidates, PreparedNode)> {
    match filter {
        Filter::And(children) => {
            if children.is_empty() {
                return Ok((ClusterCandidates::All, PreparedNode::True));
            }
            let mut combined: Option<RoaringBitmap> = None;
            let mut prep_children = Vec::with_capacity(children.len());
            for child in children {
                let (cands, pnode) = eval_cluster_node(storage, schema, child)?;
                match cands {
                    ClusterCandidates::All => {}
                    ClusterCandidates::Subset(bm) => {
                        if bm.is_empty() {
                            return Ok((
                                ClusterCandidates::Subset(RoaringBitmap::new()),
                                PreparedNode::False,
                            ));
                        }
                        combined = Some(match combined {
                            Some(mut cur) => {
                                cur &= &bm;
                                cur
                            }
                            None => bm,
                        });
                        if combined.as_ref().is_some_and(RoaringBitmap::is_empty) {
                            return Ok((
                                ClusterCandidates::Subset(RoaringBitmap::new()),
                                PreparedNode::False,
                            ));
                        }
                    }
                }
                prep_children.push(pnode);
            }
            let cands = match combined {
                Some(bm) => ClusterCandidates::Subset(bm),
                None => ClusterCandidates::All,
            };
            Ok((cands, PreparedNode::And(prep_children)))
        }
        Filter::Or(children) => {
            if children.is_empty() {
                return Ok((
                    ClusterCandidates::Subset(RoaringBitmap::new()),
                    PreparedNode::False,
                ));
            }
            let mut is_all = false;
            let mut union_bm = RoaringBitmap::new();
            let mut prep_children = Vec::with_capacity(children.len());
            for child in children {
                let (cands, pnode) = eval_cluster_node(storage, schema, child)?;
                match cands {
                    ClusterCandidates::All => is_all = true,
                    ClusterCandidates::Subset(bm) => {
                        if !is_all {
                            union_bm |= &bm;
                        }
                    }
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
        Filter::Not(child) => {
            let (_cands, pnode) = eval_cluster_node(storage, schema, child)?;
            Ok((ClusterCandidates::All, PreparedNode::Not(Box::new(pnode))))
        }
        Filter::Eq(attr, val) => {
            let attr_id = schema.get_by_name(attr).unwrap().id;
            if matches!(val, ScalarValue::Null) {
                let pnode = eval_is_null_node(storage, attr_id)?;
                return Ok((ClusterCandidates::All, pnode));
            }
            let key = encode_scalar_cluster_key(attr_id, val)?;
            match storage.get_attr_cluster_bitmap(&key)? {
                Some(bm) if !bm.is_empty() => Ok((
                    ClusterCandidates::Subset(bm.clone()),
                    PreparedNode::Keys(vec![(key, bm)]),
                )),
                _ => Ok((
                    ClusterCandidates::Subset(RoaringBitmap::new()),
                    PreparedNode::False,
                )),
            }
        }
        Filter::NotEq(attr, val) => {
            let eq = Filter::Eq(attr.clone(), val.clone());
            let (_cands, pnode) = eval_cluster_node(storage, schema, &eq)?;
            let cands = if matches!(val, ScalarValue::Null) {
                // NotEq(attr, Null) == IsNotNull(attr), which CAN prune clusters!
                let attr_id = schema.get_by_name(attr).unwrap().id;
                let key = encode_present_cluster_key(attr_id);
                match storage.get_attr_cluster_bitmap(&key)? {
                    Some(bm) if !bm.is_empty() => {
                        return Ok((
                            ClusterCandidates::Subset(bm.clone()),
                            PreparedNode::Keys(vec![(key, bm)]),
                        ));
                    }
                    _ => ClusterCandidates::Subset(RoaringBitmap::new()),
                }
            } else {
                ClusterCandidates::All
            };
            Ok((cands, PreparedNode::Not(Box::new(pnode))))
        }
        Filter::In(attr, vals) | Filter::ContainsAny(attr, vals) => {
            let attr_id = schema.get_by_name(attr).unwrap().id;
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
        Filter::NotIn(attr, vals) => {
            let inner = Filter::In(attr.clone(), vals.clone());
            let (_cands, pnode) = eval_cluster_node(storage, schema, &inner)?;
            Ok((ClusterCandidates::All, PreparedNode::Not(Box::new(pnode))))
        }
        Filter::Contains(attr, val) => {
            let attr_id = schema.get_by_name(attr).unwrap().id;
            let key = encode_scalar_cluster_key(attr_id, val)?;
            match storage.get_attr_cluster_bitmap(&key)? {
                Some(bm) if !bm.is_empty() => Ok((
                    ClusterCandidates::Subset(bm.clone()),
                    PreparedNode::Keys(vec![(key, bm)]),
                )),
                _ => Ok((
                    ClusterCandidates::Subset(RoaringBitmap::new()),
                    PreparedNode::False,
                )),
            }
        }
        Filter::Lt(attr, val)
        | Filter::Lte(attr, val)
        | Filter::Gt(attr, val)
        | Filter::Gte(attr, val) => {
            let entry = schema.get_by_name(attr).unwrap();
            let type_tag = ordered_type_tag(entry.attr_type)?;
            let prefix = encode_attr_prefix(entry.id, type_tag);
            let bound = encode_scalar_cluster_key(entry.id, val)?;

            let entries = match filter {
                Filter::Lt(_, _) => {
                    storage.scan_attr_cluster_range(prefix, &prefix, true, Some(&bound), false)?
                }
                Filter::Lte(_, _) => {
                    storage.scan_attr_cluster_range(prefix, &prefix, true, Some(&bound), true)?
                }
                Filter::Gt(_, _) => {
                    storage.scan_attr_cluster_range(prefix, &bound, false, None, false)?
                }
                Filter::Gte(_, _) => {
                    storage.scan_attr_cluster_range(prefix, &bound, true, None, false)?
                }
                _ => unreachable!(),
            };

            let mut union_bm = RoaringBitmap::new();
            let mut items = Vec::with_capacity(entries.len());
            for (k, bm) in entries {
                if !bm.is_empty() {
                    union_bm |= &bm;
                    items.push((k, bm));
                }
            }
            let node = if items.is_empty() {
                PreparedNode::False
            } else {
                PreparedNode::Keys(items)
            };
            Ok((ClusterCandidates::Subset(union_bm), node))
        }
        Filter::IsNull(attr) => {
            let attr_id = schema.get_by_name(attr).unwrap().id;
            let pnode = eval_is_null_node(storage, attr_id)?;
            Ok((ClusterCandidates::All, pnode))
        }
        Filter::IsNotNull(attr) => {
            let attr_id = schema.get_by_name(attr).unwrap().id;
            let key = encode_present_cluster_key(attr_id);
            match storage.get_attr_cluster_bitmap(&key)? {
                Some(bm) if !bm.is_empty() => Ok((
                    ClusterCandidates::Subset(bm.clone()),
                    PreparedNode::Keys(vec![(key, bm)]),
                )),
                _ => Ok((
                    ClusterCandidates::Subset(RoaringBitmap::new()),
                    PreparedNode::False,
                )),
            }
        }
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
        PreparedNode::True | PreparedNode::False => {}
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

fn node_needs_complement(node: &PreparedNode) -> bool {
    match node {
        PreparedNode::True | PreparedNode::Not(_) => true,
        PreparedNode::False | PreparedNode::Keys(_) => false,
        PreparedNode::And(children) | PreparedNode::Or(children) => {
            children.iter().any(node_needs_complement)
        }
    }
}

fn eval_prepared_leaf(
    node: &PreparedNode,
    leaf_id: NodeId,
    postings: &HashMap<Vec<u8>, RoaringBitmap>,
    all_local: Option<&RoaringBitmap>,
) -> RoaringBitmap {
    match node {
        PreparedNode::True => all_local.cloned().unwrap_or_default(),
        PreparedNode::False => RoaringBitmap::new(),
        PreparedNode::And(children) => {
            let mut acc: Option<RoaringBitmap> = None;
            for child in children {
                let bm = eval_prepared_leaf(child, leaf_id, postings, all_local);
                if bm.is_empty() {
                    return RoaringBitmap::new();
                }
                acc = Some(match acc {
                    Some(mut cur) => {
                        cur &= &bm;
                        if cur.is_empty() {
                            return RoaringBitmap::new();
                        }
                        cur
                    }
                    None => bm,
                });
            }
            acc.unwrap_or_else(|| all_local.cloned().unwrap_or_default())
        }
        PreparedNode::Or(children) => {
            let mut acc = RoaringBitmap::new();
            for child in children {
                acc |= &eval_prepared_leaf(child, leaf_id, postings, all_local);
            }
            acc
        }
        PreparedNode::Not(child) => {
            let sub = eval_prepared_leaf(child, leaf_id, postings, all_local);
            let mut out = all_local.cloned().unwrap_or_default();
            out -= &sub;
            out
        }
        PreparedNode::Keys(items) => {
            let mut acc = RoaringBitmap::new();
            for (cluster_key, cluster_bm) in items {
                if cluster_bm.contains(leaf_id) {
                    let pkey = make_posting_key(cluster_key, leaf_id);
                    if let Some(bm) = postings.get(&pkey) {
                        acc |= bm;
                    }
                }
            }
            acc
        }
    }
}
