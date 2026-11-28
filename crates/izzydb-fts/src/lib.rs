//! Text analysis, 256-doc posting blocks, BM25 scoring, and SIMD Block-Max MAXSCORE for `izzydb`.

pub mod analyzer;
pub mod block;
pub mod bm25;
pub mod maxscore;
pub mod simd;

pub use alyze::analyze::ReusableBuffer;
pub use analyzer::{AnalyzedDoc, AnalyzedTerm, FtsAnalyzer, QueryToken, STRING_ARRAY_POSITION_GAP};
pub use block::{
    AttrFtsStats, DOC_LEN_BLOCK_HEADER_LEN, DocLenBlock, DocLenBlockView, EncodedPosting,
    FTS_CHUNK_HEADER_LEN, FtsChunk, FtsChunkLayout, FtsChunkView, FtsMeta, INLINE_TERM_MAX_DF,
    IntWidth, POSTING_BLOCK_HEADER_LEN, POSTING_DENSE_THRESHOLD, PackedPositionBlock,
    PackedPostingBlock, PositionBlockView, PostingBlockHeader, PostingBlockPayload,
    PostingBlockView, SKIP_BLOCK_HEADER_LEN, SkipBlock, SkipBlockLayout, SkipBlockView,
    TERM_RECORD_HEADER_LEN, TERM_STATS_LEN, TermRecord, TermStats, TfEncoding,
    dl_over_tf_lower_bound,
};
pub use bm25::{Bm25TermScorer, bm25_idf, bm25_qtf_norm};
pub use maxscore::{
    BlockMaskProvider, Bm25Result, DEFAULT_MAX_PREFIX_EXPANSIONS, FtsDocLenCursor,
    FtsPostingSource, FtsSkipCursor, RankModifier, execute_rank_by, execute_rank_by_with,
    execute_rank_by_with_shards,
};
pub use simd::SimdTier;
