# Bulk Import & Append Architecture (`izzydb-bulk`)

This document describes the offline Parquet bulk importer, **populated-target appender (`bulk_append`)**, and detached column ingestion CLI in [`crates/izzydb-bulk`](../../crates/izzydb-bulk).

Implementation sources:
- Orchestrator (`bulk_import`, `build_ssts`, `ingest_ssts`, `bulk_append`): [`crates/izzydb-bulk/src/lib.rs`](../../crates/izzydb-bulk/src/lib.rs)
- Parallel Parquet scanner & detached scanner: [`crates/izzydb-bulk/src/common/parquet_reader/mod.rs`](../../crates/izzydb-bulk/src/common/parquet_reader/mod.rs)
- Segment SST emitters: [`crates/izzydb-bulk/src/pipeline/mod.rs`](../../crates/izzydb-bulk/src/pipeline/mod.rs)
- ANN tree builder over `RowId`s: [`crates/izzydb-bulk/src/tree_builder/mod.rs`](../../crates/izzydb-bulk/src/tree_builder/mod.rs)
- SST bundle manifest v2 (`SST_BUNDLE_FORMAT_VERSION = 2`): [`crates/izzydb-bulk/src/common/sst.rs`](../../crates/izzydb-bulk/src/common/sst.rs)
- CLI (`izzydb-import import | append | columns`): [`crates/izzydb-bulk/src/bin/izzydb-import.rs`](../../crates/izzydb-bulk/src/bin/izzydb-import.rs)

---

## 1. Empty Import (`bulk_import`, `build_ssts`, `ingest_ssts`)

Because v3 rows live in independent sealed segments and secondary indexes use `RowId = (seg, local)` coordinates, `bulk_import` partitions imported rows directly into sealed segments sized by `SegmentConfig.seal_rows` (default `65,536`):

1. **Phase 0 (Prepare & Validate):** Validates `CollectionConfig` (supporting both vectored and **vectorless** collections via `CollectionConfig::without_vectors()`), `BulkImportConfig`, target emptiness, and `work_dir` filesystem device (`move_files = true` hard-links when on the same filesystem).
2. **Phase 1 (Parallel Parquet Scan & Stage):** Projects `id_column`, optional `vector_column`, inline attributes, and any detached attributes via `rayon`:
   - Writes `vectors.bin` (if `has_vectors`), `doc_ids_by_row.bin`, `docs_shard_*.bin` + `docs.idx` (v3 `RowView`), and staged detached attribute blocks when present.
   - Deduplicates `(doc_id, row)` via external LZ4-framed sorting under `DuplicatePolicy::{Error, LastWins}` and assigns sequential `(seg, local)` `RowId`s across new sealed segments.
3. **Phase 2 (Optional ANN Build):** When `has_vectors` is true, trains the clustering skeleton via `balanced_two_means`, routes all vectors in parallel, assigns BFS `NodeId`s, and records `RowId`s (`VecEntry { row_id, vec_idx }`) directly into `PackedLeafNode` v3 leaves (`cf_leaf_entries`) and internal nodes (`default[0x02]`).
4. **Phase 3 (Emit Segment SSTs):** Streams sorted SSTs directly in segment/column-family order:
   - `cf_rows`, `cf_row_doc_ids`, `cf_vectors`, `cf_columns` (sealed inline columnar blocks built directly), `cf_id_map`, `cf_attr_postings` (`SegPosting`, value-major), `cf_detached` / `cf_detached_postings`, FTS CFs (`cf_fts_terms` v4, `cf_fts_skips`, `cf_fts_postings`, `cf_fts_positions`), and `default` (`\x01schema` v4, `\x01manifest`, `\x01ann_meta`, `\x01fts_meta` v4, `0x02`, `0x05`, `0x06`, `0x07`, `0x11` sealed `SegmentMeta`, `0x12` zones, `0x13` `DetachedMeta`).
5. **Phase 4 (Crash-Safe Ingest):** Brackets multi-CF `ingest_external_file_cf_opts` with `default[b"\x01bulk_import_v3"]` (`sync = true`), ingesting all data CFs before `default`, then clearing the marker.

---

## 2. Populated-Target Bulk Append (`bulk_append`)

Segmented v3 storage enables **`bulk_append(target, files, config, progress)`** onto an **already-populated** collection without rewriting existing segments:

1. **Pre-flight & Open:** Opens the existing `Collection` (verifying no incomplete `\x01bulk_import_v3` marker and matching schema).
2. **Scan & Assign New Segments:** Scans the Parquet input and assigns fresh sealed segment IDs strictly starting at `Manifest.next_segment_id`, writing new non-overlapping segments via `emit_segment_range_ssts` (note: `cf_attr_postings` global SST keys are disjoint or merged cleanly by `seg`, while existing `0x06` / `0x07` counts and FTS terms with colliding tokens are reconciled transactionally).
3. **Overlap / Replacement Handling:** For any incoming `DocId` already present in `cf_id_map`, marks the previous `old_row` tombstoned (`Collection::with_bulk_tx` / `delete_row_for_replace`) in one atomic transaction (`docs_replaced`).
4. **Global ANN Strategy:**
   - **Incremental insertion** (when appended rows are a fraction of existing collection size): inserts new vectors into the existing `AnnIndex` (`rebuilt_ann_tree = false`).
   - **Threshold tree rebuild** (`rebuilt_ann_tree = true`): when appended vectors exceed existing live vectors (or tree was empty), trains/rebuilds the global ANN tree across all live vectors atomically via `replace_ann_tree`.
5. **Recovery & Safety:** `bulk_append_with_hook` verifies clean recovery before commit. If a crash occurs after `\x01bulk_import_v3` ingestion starts, `Storage::open` rejects opening with `CorruptedData`.

---

## 3. CLI Subcommands (`izzydb-import`)

`izzydb-import` provides three subcommands:
- `izzydb-import import`: populate a new/empty collection (vectored with `--dim`, or vectorless when `--dim` is omitted).
- `izzydb-import append`: bulk-append Parquet files as new sealed segments into an existing collection.
- `izzydb-import columns`: scan `--columns col1 col2 ...` from Parquet and ingest into existing documents via `Collection::detached_ingest` + `publish_detached` (`--mode replace|merge`).
