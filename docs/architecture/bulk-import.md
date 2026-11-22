# Bulk Import Architecture (`izzydb-bulk`)

This document describes the offline Parquet bulk importer ([`crates/izzydb-bulk`](../../crates/izzydb-bulk)) and its external SST ingestion protocol into `izzydb`.

## 1. Overview & Empty-Target Scope

Calling `Collection::upsert` row-by-row runs a full RocksDB `WriteBatch` transaction, clustering-tree traversal, RaBitQ re-encoding, and read-modify-write attribute/FTS aggregation per document. `izzydb-bulk` replaces that with an offline multi-phase pipeline that builds sorted, non-overlapping RocksDB SST files via `SstFileWriter` and links them straight into the bottommost level via `DB::ingest_external_file_cf_opts`.

Because internal identifiers (`NodeId`, `LocalId`, `FtsDocId`, `TermId`) and aggregate payloads (`TermRecord` stats, `cf_attr_clusters` bitmaps, `default[0x06]` counts, `AttrZoneMap`, `FtsMeta`) are dense or read-modify-write, **v1 requires an empty target** (either a non-existent path or an existing collection with `num_vectors == 0` and no attribute/FTS state). Populated targets fail immediately with `TreeError::InvalidConfig`.

## 2. Multi-Phase Offline Pipeline

1. **Phase 0 (Prepare & Validation):**
   - Validates `CollectionConfig`, `BulkImportConfig` (`fill_factor`, `target_sst_bytes`, column names), and builds `CollectionSchema` and `RabitqQuantizer`.
   - Rejects non-empty targets before any write.
   - Checks `st_dev` between `work_dir` and `target` so `ingest_ssts` uses `move_files = true` (hard-linking) on the same filesystem and falls back to copying across filesystems.
2. **Phase 1 (Parallel Parquet Scan & Stage):**
   - Projects only `id_column`, `vector_column`, and mapped schema attribute columns via `ProjectionMask::roots`.
   - Decodes Arrow batches across `(file_idx, row_group_idx)` tasks in parallel (`rayon`), validates vectors/attributes with `prepare_vector` and `CollectionSchema::validate_attributes`, and stages:
     - `vectors.bin` (fixed-stride `[f32; dim]` LE, mmapped later),
     - `doc_ids_by_row.bin` (`[u64; R]` LE),
     - `docs_shard_*.bin` + `docs.idx` (v2 encoded forward attributes),
     - `(doc_id, row)` pairs into an LZ4-framed `ExternalSorter`.
   - Merges sorted `(doc_id, row)` pairs to enforce `DuplicatePolicy::Error` or `DuplicatePolicy::LastWins` (keeping the highest `(file_idx, row_in_file)`), emitting `rows_by_doc_id.bin`.
3. **Phase 2 (Train Tree Skeleton):**
   - Draws a uniform sample of up to `min(train_sample, N)` live vectors (capped to respect `memory_budget_bytes`).
   - Recursively bisects the sample using `izzydb_core::kmeans::balanced_two_means` until projected cluster sizes fall to `<= fill_factor * max_leaf_size`, then groups leaf centroids bottom-up into a uniform-depth internal hierarchy with fanouts in `[min_internal_fanout, max_internal_fanout]` (root in `[2, max_internal_fanout]`).
4. **Phase 3 (Assign & Repair):**
   - Routes every live vector through the skeleton in parallel using `izzydb::search::route_to_leaf_with` (`insert_beam_width`).
   - Repairs oversize leaves (`> max_leaf_size`) via `balanced_two_means`, merges/rebalances undersize leaves (`< min_leaf_size`), repairs internal node fanouts bottom-up, and recomputes exact leaf and internal centroids.
   - Assigns BFS `NodeId`s (`root_id = 1`, internal nodes contiguous, leaves contiguous), dense slots `0..n`, and leaf-major `fts_doc_id`s (`0..N` when FTS is enabled), persisting `placement.bin`.
5. **Phase 4 (Emit Pass A — Leaf-Major):**
   - Parallel over disjoint contiguous leaf-ID ranges:
     - Writes `cf_leaf_entries` SSTs directly in ascending `leaf_id` order using `PackedLeafNode::from_entries`.
     - Extracts filterable attribute index keys (`cf_attr_postings` sorter, `cf_attr_clusters` + `default[0x06]` counts sorter, and per-leaf `AttrZoneMap` bounds).
     - Runs `FtsAnalyzer::analyze_attribute_value` to record per-attribute document lengths (`fts_dls`) and emit `(attr_id, token, fts_doc_id, tf, positions)` to the FTS sorter.
6. **Phase 5 (Emit Pass B — `DocId` Order):**
   - Parallel over disjoint `doc_id` ranges from `rows_by_doc_id.bin`, writing `cf_docs`, `cf_vectors`, and `cf_doc_loc` SSTs directly in ascending `doc_id` order.
7. **Phase 6 (Finish Sorted CFs & `default` CF):**
   - Streams sorted `cf_attr_postings` and `cf_attr_clusters` (`RoaringBitmap::from_sorted_iter`, spilling summed global counts for `default[0x06]`).
   - Streams sorted FTS postings in `(attr_id, token, fts_doc_id)` order, assigning monotonic `TermId`s, building v3 `PackedPostingBlock`s, `PackedPositionBlock`s, `SkipBlock` chunks, and inline `TermRecord`s (`doc_freq <= 8` on 1 block).
   - Writes `default` CF in strict key-tag order: `\x01attr_counts_v1`, `\x01fts_meta`, `\x01schema`, `\x01tree_meta`, `0x02` internal nodes, `0x03` attribute zones, `0x04` FTS reverse blocks, `0x05` FTS doc lengths, and `0x06` global attribute counts.

## 3. SST Options Parity & Crash-Safe Marker Protocol

Every `SstSink` uses `izzydb_storage::cf_options_for` so prefix extractors, Bloom filters, and `DBCompressionType::None` settings on zero-copy CFs match live `Storage::open` column families exactly.

Because `rocksdb` 0.23 exposes per-CF `ingest_external_file_cf_opts` rather than atomic multi-CF `IngestExternalFiles`, `ingest_ssts` brackets ingestion with a WAL-synced marker in `default`:

1. `Storage::open_for_bulk_ingest(target, &tree_config)` verifies the target has no pending marker and is empty.
2. `storage.put_bulk_import_marker(build_id)` writes `default[b"\x01bulk_import_v1"]` with `sync = true` (and deletes any empty initial root leaf `1`).
3. All data CFs (`cf_vectors`, `cf_docs`, `cf_doc_loc`, `cf_leaf_entries`, `cf_attr_clusters`, `cf_attr_postings`, `cf_fts_positions`, `cf_fts_postings`, `cf_fts_skips`, `cf_fts_terms`) are ingested first.
4. `default` (containing `tree_meta`, `schema`, `fts_meta`, internal nodes, zones, rev blocks, doc lengths, and counts) is ingested last (`allow_global_seqno = true` handles overlap with the marker range).
5. `storage.clear_bulk_import_marker()` deletes `b"\x01bulk_import_v1"` with `sync = true`.

If a crash occurs between steps 2 and 5, `Storage::open` and `Collection::open` detect `b"\x01bulk_import_v1"` and fail with `TreeError::CorruptedData("incomplete bulk import; delete the database directory and re-run")`.

## 4. Split Build & Ingest (`build_ssts` / `ingest_ssts`)

`build_ssts` writes an `SstBundle` directory with `manifest.json` (format version, `CollectionConfig` hash, `dim`, `metric`, encoded `CollectionSchema`, `TreeMeta`, and per-CF SST paths). `ingest_ssts` validates the manifest against the target configuration before ingesting.
