//! Column family option builders.


use std::sync::Arc;

use rocksdb::{
    BlockBasedOptions, Cache, DBCompressionType, DEFAULT_COLUMN_FAMILY_NAME, Options, SliceTransform,
};

use izzydb_core::error::{Result, TreeError};



use super::*;
use super::compaction_filter::*;

pub fn default_sst_block_cache() -> Cache {
    Cache::new_lru_cache(BLOCK_CACHE_BYTES)
}

pub fn cf_options_for(cf_name: &str, block_cache: &Cache) -> Result<Options> {
    cf_options_with_filters(cf_name, block_cache, None)
}


pub(super) fn cf_options_with_filters(
    cf_name: &str,
    block_cache: &Cache,
    filters: Option<FilterHandles<'_>>,
) -> Result<Options> {
    let mut opts = match cf_name {
        DEFAULT_COLUMN_FAMILY_NAME | CF_ID_MAP | CF_ROWS | CF_COLUMNS | CF_DETACHED => {
            point_lookup_cf_opts(block_cache, false)
        }
        CF_ROW_DOC_IDS | CF_VECTORS | CF_LEAF_ENTRIES => {
            point_lookup_cf_opts(block_cache, true)
        }
        CF_ATTR_POSTINGS => prefix_cf_opts(block_cache, ATTR_POSTING_PREFIX_LEN, true),
        CF_DETACHED_POSTINGS => {
            prefix_cf_opts(block_cache, DETACHED_POSTING_PREFIX_LEN, true)
        }
        CF_FTS_TERMS => prefix_cf_opts(block_cache, 2, false),
        CF_FTS_SKIPS | CF_FTS_POSTINGS => prefix_cf_opts(block_cache, 4, true),
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
            CF_ROWS | CF_ROW_DOC_IDS | CF_VECTORS | CF_COLUMNS => Some(OrphanFilterKind::LeadingSeg),
            CF_ATTR_POSTINGS => Some(OrphanFilterKind::AttrPostings),
            CF_DETACHED => Some(OrphanFilterKind::DetachedBlocks),
            CF_DETACHED_POSTINGS => Some(OrphanFilterKind::DetachedPostings),
            CF_FTS_SKIPS | CF_FTS_POSTINGS | CF_FTS_POSITIONS => {
                Some(OrphanFilterKind::FtsTermSeg)
            }
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

fn point_lookup_cf_opts(block_cache: &Cache, disable_compression: bool) -> Options {
    let mut opts = Options::default();
    opts.set_memtable_whole_key_filtering(true);
    if disable_compression {
        opts.set_compression_type(DBCompressionType::None);
    }

    let mut bbo = BlockBasedOptions::default();
    bbo.set_block_cache(block_cache);
    bbo.set_bloom_filter(10.0, false);
    bbo.set_cache_index_and_filter_blocks(true);
    bbo.set_pin_l0_filter_and_index_blocks_in_cache(true);
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
