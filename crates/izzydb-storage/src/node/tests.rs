use izzydb_core::config::{DistanceMetric, RowId};

use super::*;

#[test]
fn test_packed_leaf_node_insert_remove_roundtrip() {
    let dim = 5;
    let padded_dim = 64;
    let words = 1;
    let mut leaf = PackedLeafNode::empty(Some(9), dim, padded_dim);

    let r30 = RowId::new(1, 30);
    let r10 = RowId::new(0, 10);
    let r20 = RowId::new(2, 20);
    let r40 = RowId::new(3, 40);

    assert_eq!(leaf.upsert_entry(r30, None, 1.5, 2.0, &[0xAA], 4, words), 0);
    assert_eq!(leaf.upsert_entry(r10, None, 0.5, 1.0, &[0x11], 2, words), 1);
    assert_eq!(leaf.upsert_entry(r20, None, 2.5, 3.0, &[0x55], 4, words), 2);
    assert_eq!(leaf.slot_cap, 3);
    assert_eq!(leaf.live_count, 3);

    let bytes = leaf.encode(dim, padded_dim, words);
    let decoded = PackedLeafNode::decode(&bytes, dim, padded_dim, words).unwrap();
    assert_eq!(leaf, decoded);

    // Removing slot 1 (r10) keeps slot 0 (r30) and slot 2 (r20) untouched!
    assert_eq!(leaf.remove_entry(r10, Some(1), words), Some(1));
    assert_eq!(leaf.live_count, 2);
    assert_eq!(leaf.slot_cap, 3);
    assert!(!leaf.is_live(1));
    assert!(leaf.is_live(0));
    assert!(leaf.is_live(2));
    assert_eq!(leaf.row_id(0), r30);
    assert_eq!(leaf.row_id(2), r20);

    // Next insert reuses free slot 1!
    assert_eq!(leaf.upsert_entry(r40, None, 3.5, 4.0, &[0xFF], 8, words), 1);
    assert_eq!(leaf.live_count, 3);
    assert_eq!(leaf.slot_cap, 3);
    assert_eq!(leaf.row_id(1), r40);

    // Multi-word test
    let words2 = 3;
    let mut leaf2 = PackedLeafNode::empty(None, dim, 192);
    let s2 = leaf2.upsert_entry(RowId::new(0, 2), None, 1.0, 1.0, &[1, 2, 3], 6, words2);
    let s4 = leaf2.upsert_entry(RowId::new(0, 4), None, 2.0, 2.0, &[4, 5, 6], 6, words2);
    let s1 = leaf2.upsert_entry(RowId::new(0, 1), None, 3.0, 3.0, &[7, 8, 9], 7, words2);
    let s3 = leaf2.upsert_entry(RowId::new(0, 3), None, 4.0, 4.0, &[10, 11, 12], 7, words2);
    assert_eq!((s2, s4, s1, s3), (0, 1, 2, 3));
    assert_eq!(leaf2.entry_code_word(2, 1, words2), 8);
    assert_eq!(leaf2.entry_code_word(3, 2, words2), 12);
    let bytes2 = leaf2.encode(dim, 192, words2);
    assert_eq!(
        PackedLeafNode::decode(&bytes2, dim, 192, words2).unwrap(),
        leaf2
    );

    // AnnMeta roundtrip
    let meta = AnnMeta {
        root_id: 1,
        root_level: 2,
        next_node_id: 99,
        dim: 128,
        metric: DistanceMetric::Cosine,
        num_vectors: 1000,
        rabitq_seed: 42,
        rabitq_bits: 1,
        stale_rows: 17,
        vacuum_epoch: 3,
        vacuum_cursor: 55,
    };
    assert_eq!(AnnMeta::decode(&meta.encode()).unwrap(), meta);
}
