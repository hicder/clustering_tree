use super::*;
use izzydb_core::config::{DistanceMetric, TreeConfig};
use izzydb_core::rowset::Liveness;
use izzydb_core::schema::CollectionConfig;
use tempfile::TempDir;

#[test]
fn test_tx_vectors_id_map_and_segment_snapshot_isolation() {
    let dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::new(TreeConfig::new(4, DistanceMetric::L2Squared));
    let (storage, manifest, ann) = Storage::open(dir.path(), &cfg).unwrap();
    let mut cache = HashMap::new();
    let mut leaf_parent = HashMap::new();

    let r0 = RowId::new(0, 0);
    let r1 = RowId::new(0, 1);
    let r2 = RowId::new(0, 2);

    let (manifest, ann) = {
        let mut tx = TxContext::new(&storage, &mut cache, &mut leaf_parent, manifest, ann);
        let mut seg0 = SegmentMeta::new_open(0);
        seg0.row_count = 2;
        tx.put_seg_meta(0, seg0);
        tx.manifest_mut().open_segment = Some(0);
        tx.manifest_mut().next_segment_id = 1;
        tx.manifest_mut().live_rows = 2;
        tx.manifest_mut().total_rows = 2;

        tx.put_id_map(100, r0);
        tx.put_id_map(200, r1);
        tx.set_row_doc_id(r0, 100).unwrap();
        tx.set_row_doc_id(r1, 200).unwrap();
        tx.put_vector(r0, vec![1.0, 1.1, 1.2, 1.3]);
        tx.put_vector(r1, vec![2.0, 2.1, 2.2, 2.3]);
        tx.commit().unwrap()
    };

    let snap1 = storage.segment_snapshot();
    assert!(snap1.is_live(r0));
    assert!(snap1.is_live(r1));
    assert_eq!(storage.resolve_doc_ids(&[r0, r1]).unwrap(), vec![Some(100), Some(200)]);

    let mut tx = TxContext::new(&storage, &mut cache, &mut leaf_parent, manifest, ann);
    tx.tombstone_row(r0).unwrap();
    let mut seg0 = tx.get_seg_meta(0).unwrap();
    seg0.row_count = 3;
    tx.put_seg_meta(0, seg0);
    tx.put_id_map(100, r2);
    tx.set_row_doc_id(r2, 100).unwrap();
    tx.put_vector(r2, vec![9.0, 9.1, 9.2, 9.3]);

    let flat = tx.get_vectors_flat(&[r1, r2]).unwrap();
    assert_eq!(flat, vec![2.0, 2.1, 2.2, 2.3, 9.0, 9.1, 9.2, 9.3]);

    // Snap1 still sees r0 live before commit, and even after commit!
    tx.commit().unwrap();
    assert!(snap1.is_live(r0));

    let snap2 = storage.segment_snapshot();
    assert!(!snap2.is_live(r0));
    assert!(snap2.is_live(r1));
    assert!(snap2.is_live(r2));
    assert_eq!(storage.get_id_map(100).unwrap(), Some(r2));
}
