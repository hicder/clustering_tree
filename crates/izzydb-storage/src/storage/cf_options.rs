//! Column family option builders.

use std::sync::Arc;

use rocksdb::{
    BlockBasedIndexType, BlockBasedOptions, Cache, DBCompressionType, DEFAULT_COLUMN_FAMILY_NAME,
    DataBlockIndexType, Options, SliceTransform,
};

use izzydb_core::error::{Result, TreeError};

use super::compaction_filter::*;
use super::*;

/// Default size of the shared RocksDB block cache (128 MiB).
///
/// The cache is shared by every column family; partitioned index and filter blocks keep the
/// per-CF metadata footprint small enough to stay resident.
pub const DEFAULT_BLOCK_CACHE_BYTES: usize = 128 * 1024 * 1024;

/// Data block size for large point-lookup column families (16 KiB).
const LARGE_CF_BLOCK_SIZE: usize = 16 * 1024;

/// Data block size for the remaining column families (RocksDB default).
const DEFAULT_CF_BLOCK_SIZE: usize = 4 * 1024;

/// Create the default shared block cache used by [`Storage::open`](crate::Storage::open).
pub fn default_sst_block_cache() -> Cache {
    sst_block_cache(DEFAULT_BLOCK_CACHE_BYTES)
}

/// Create a shared block cache with an explicit byte capacity.
pub fn sst_block_cache(capacity: usize) -> Cache {
    Cache::new_lru_cache(capacity)
}

pub fn cf_options_for(cf_name: &str, block_cache: &Cache) -> Result<Options> {
    cf_options_with_filters(cf_name, block_cache, None)
}

/// Per-column-family RocksDB table tuning.
///
/// Controls Bloom filters, index partitioning, data block size, and compression. Search-path
/// column families whose keys are always known to exist skip Bloom filters entirely, while
/// large point-lookup families use two-level partitioned indexes so metadata reloads fetch a
/// single 4 KiB partition instead of a multi-megabyte index/filter blob.
///
/// ```text
/// // cf_vectors: no Bloom, two-level index, 16 KiB blocks, uncompressed
/// CfTuning { bloom: false, two_level_index: true, block_size: 16384, .. }
/// ```
#[derive(Debug, Clone, Copy)]
pub(super) struct CfTuning {
    /// Enable a 10-bit whole-key Bloom filter.
    pub bloom: bool,
    /// Use a two-level partitioned index instead of a monolithic binary-search index.
    pub two_level_index: bool,
    /// Partition Bloom filters so individual 4 KiB partitions can be reloaded.
    pub partition_filters: bool,
    /// Data block size in bytes.
    pub block_size: usize,
    /// Disable compression (store uncompressed) for the column family.
    pub uncompressed: bool,
    /// Append a compact hash table to data blocks for faster point lookups.
    pub data_block_hash: bool,
}

impl CfTuning {
    const fn new() -> Self {
        Self {
            bloom: true,
            two_level_index: false,
            partition_filters: false,
            block_size: DEFAULT_CF_BLOCK_SIZE,
            uncompressed: false,
            data_block_hash: false,
        }
    }
}

/// Return the table tuning for `cf_name`.
///
/// Search hot paths read `cf_vectors`, `cf_leaf_entries`, `cf_rows`, and `cf_row_doc_ids` by
/// `RowId`/`NodeId` that are known to exist, so Bloom filters there only cost cache capacity
/// and hash cycles. Attributes, id maps, and FTS terms keep their Bloom filters because
/// negative lookups are common on those paths.
pub(super) fn cf_tuning(cf_name: &str) -> CfTuning {
    let mut tuning = CfTuning::new();
    match cf_name {
        CF_VECTORS => {
            tuning.bloom = false;
            tuning.two_level_index = true;
            tuning.block_size = LARGE_CF_BLOCK_SIZE;
            tuning.uncompressed = true;
            tuning.data_block_hash = true;
        }
        CF_LEAF_ENTRIES => {
            tuning.bloom = false;
            tuning.two_level_index = true;
            tuning.block_size = LARGE_CF_BLOCK_SIZE;
            tuning.uncompressed = true;
        }
        CF_ROW_DOC_IDS => {
            tuning.bloom = false;
            tuning.two_level_index = true;
            tuning.uncompressed = true;
        }
        CF_ROWS => {
            tuning.bloom = false;
            tuning.two_level_index = true;
        }
        CF_ID_MAP => {
            tuning.two_level_index = true;
            tuning.partition_filters = true;
        }
        CF_COLUMNS => {
            tuning.two_level_index = true;
            tuning.partition_filters = true;
            tuning.uncompressed = true;
            tuning.data_block_hash = true;
        }
        CF_DETACHED => {
            tuning.two_level_index = true;
            tuning.partition_filters = true;
            tuning.uncompressed = true;
        }
        _ => {}
    }
    tuning
}

pub(super) fn cf_options_with_filters(
    cf_name: &str,
    block_cache: &Cache,
    filters: Option<FilterHandles<'_>>,
) -> Result<Options> {
    let mut opts = match cf_name {
        DEFAULT_COLUMN_FAMILY_NAME
        | CF_ID_MAP
        | CF_ROWS
        | CF_COLUMNS
        | CF_DETACHED
        | CF_ROW_DOC_IDS
        | CF_VECTORS
        | CF_LEAF_ENTRIES => point_lookup_cf_opts(block_cache, cf_tuning(cf_name)),
        CF_ATTR_POSTINGS => prefix_cf_opts(block_cache, ATTR_POSTING_PREFIX_LEN, true),
        CF_DETACHED_POSTINGS => prefix_cf_opts(block_cache, DETACHED_POSTING_PREFIX_LEN, true),
        CF_FTS_TERMS => prefix_cf_opts(block_cache, 2, false),
        CF_FTS_SKIPS | CF_FTS_POSTINGS => prefix_cf_opts_no_filter(block_cache, 4, true),
        CF_FTS_POSITIONS => prefix_cf_opts(block_cache, 4, false),
        other => {
            return Err(TreeError::InvalidConfig(format!(
                "unknown column family '{other}'"
            )));
        }
    };

    if let Some((dropped, retired_detached)) = filters {
        let kind = match cf_name {
            DEFAULT_COLUMN_FAMILY_NAME => Some(OrphanFilterKind::DefaultCf),
            CF_ROWS | CF_ROW_DOC_IDS | CF_VECTORS | CF_COLUMNS => {
                Some(OrphanFilterKind::LeadingSeg)
            }
            CF_ATTR_POSTINGS => Some(OrphanFilterKind::AttrPostings),
            CF_DETACHED => Some(OrphanFilterKind::DetachedBlocks),
            CF_DETACHED_POSTINGS => Some(OrphanFilterKind::DetachedPostings),
            CF_FTS_SKIPS | CF_FTS_POSTINGS | CF_FTS_POSITIONS => Some(OrphanFilterKind::FtsTermSeg),
            _ => None,
        };
        if let Some(kind) = kind {
            opts.set_compaction_filter_factory(OrphanCompactionFilterFactory {
                kind,
                dropped: Arc::clone(dropped),
                retired_detached: Arc::clone(retired_detached),
            });
        }
    }

    Ok(opts)
}

fn apply_table_tuning(bbo: &mut BlockBasedOptions, tuning: CfTuning) {
    bbo.set_block_size(tuning.block_size);
    if tuning.bloom {
        bbo.set_bloom_filter(10.0, false);
    }
    if tuning.two_level_index {
        bbo.set_index_type(BlockBasedIndexType::TwoLevelIndexSearch);
        bbo.set_pin_top_level_index_and_filter(true);
        bbo.set_partition_filters(tuning.partition_filters);
    }
    if tuning.data_block_hash {
        bbo.set_data_block_index_type(DataBlockIndexType::BinaryAndHash);
    }
    bbo.set_cache_index_and_filter_blocks(true);
    bbo.set_pin_l0_filter_and_index_blocks_in_cache(true);
}

fn point_lookup_cf_opts(block_cache: &Cache, tuning: CfTuning) -> Options {
    let mut opts = Options::default();
    opts.set_memtable_whole_key_filtering(tuning.bloom);
    if tuning.uncompressed {
        opts.set_compression_type(DBCompressionType::None);
    }

    let mut bbo = BlockBasedOptions::default();
    bbo.set_block_cache(block_cache);
    apply_table_tuning(&mut bbo, tuning);
    opts.set_block_based_table_factory(&bbo);
    opts
}

fn prefix_cf_opts(block_cache: &Cache, prefix_len: usize, disable_compression: bool) -> Options {
    let mut opts = Options::default();
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(prefix_len));
    opts.set_memtable_whole_key_filtering(true);
    if disable_compression {
        opts.set_compression_type(DBCompressionType::None);
    }

    let mut bbo = BlockBasedOptions::default();
    bbo.set_block_cache(block_cache);
    bbo.set_bloom_filter(10.0, false);
    bbo.set_whole_key_filtering(true);
    bbo.set_cache_index_and_filter_blocks(true);
    bbo.set_pin_l0_filter_and_index_blocks_in_cache(true);
    opts.set_block_based_table_factory(&bbo);
    opts
}

fn prefix_cf_opts_no_filter(
    block_cache: &Cache,
    prefix_len: usize,
    disable_compression: bool,
) -> Options {
    let mut opts = Options::default();
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(prefix_len));
    if disable_compression {
        opts.set_compression_type(DBCompressionType::None);
    }

    let mut bbo = BlockBasedOptions::default();
    bbo.set_block_cache(block_cache);
    bbo.set_cache_index_and_filter_blocks(true);
    bbo.set_pin_l0_filter_and_index_blocks_in_cache(true);
    opts.set_block_based_table_factory(&bbo);
    opts
}
