use super::*;
use crate::codec::row_key;
use izzydb_core::schema::CollectionSchema;
use izzydb_core::config::{DistanceMetric, TreeConfig};
use rocksdb::SstFileWriter;
use tempfile::TempDir;

#[test]
fn test_sst_ingest_and_bulk_import_marker() {
    let dir = TempDir::new().unwrap();
    let sst_dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::new(TreeConfig::new(4, DistanceMetric::L2Squared));

    let storage = Storage::open_for_bulk_ingest(dir.path(), &cfg).unwrap();
    storage.put_bulk_import_marker(42).unwrap();
    assert!(storage.has_bulk_import_marker().unwrap());

    let sst_path = sst_dir.path().join("rows.sst");
    {
        let opts = storage.cf_options(CF_ROWS).unwrap();
        let mut writer = SstFileWriter::create(&opts);
        writer.open(&sst_path).unwrap();
        writer.put(row_key(RowId::new(0, 10)), b"row-10").unwrap();
        writer.put(row_key(RowId::new(0, 20)), b"row-20").unwrap();
        writer.finish().unwrap();
    }
    storage
        .ingest_cf(CF_ROWS, std::slice::from_ref(&sst_path))
        .unwrap();
    drop(storage);

    assert!(matches!(
        Storage::open(dir.path(), &cfg),
        Err(TreeError::CorruptedData(msg)) if msg.contains("incomplete bulk import")
    ));

    {
        let block_cache = default_sst_block_cache();
        let raw = Storage::open_raw(dir.path(), 4, &block_cache, &cfg).unwrap();
        raw.clear_bulk_import_marker().unwrap();
    }
    let (reopened, _manifest, _ann) = Storage::open(dir.path(), &cfg).unwrap();
    assert_eq!(
        reopened.get_row_bytes(RowId::new(0, 10)).unwrap().as_deref(),
        Some(b"row-10".as_slice())
    );
    assert_eq!(
        reopened.get_row_bytes(RowId::new(0, 20)).unwrap().as_deref(),
        Some(b"row-20".as_slice())
    );
}

#[test]
fn test_vectorless_collection_open_and_mismatch() {
    let dir = TempDir::new().unwrap();
    let cfg_no_vec = CollectionConfig::without_vectors().with_seal_rows(128);
    let (storage, manifest, ann) = Storage::open(dir.path(), &cfg_no_vec).unwrap();
    assert!(ann.is_none());
    assert_eq!(manifest.seal_rows, 128);
    storage
        .put_schema(&CollectionSchema::from_config(&cfg_no_vec).unwrap())
        .unwrap();
    drop(storage);

    // Reopening vectorless DB with vectors must fail with ConfigMismatch
    let cfg_vec = CollectionConfig::new(TreeConfig::new(4, DistanceMetric::L2Squared));
    assert!(matches!(
        Storage::open(dir.path(), &cfg_vec),
        Err(TreeError::ConfigMismatch { .. })
    ));
}
