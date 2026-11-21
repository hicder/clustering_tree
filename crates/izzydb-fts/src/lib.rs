//! Text analysis, 256-doc posting blocks, BM25 scoring, and Block-Max MAXSCORE for `izzydb`.

pub mod analyzer;
pub mod block;
pub mod bm25;
pub mod maxscore;

pub use alyze::analyze::ReusableBuffer;
pub use analyzer::{AnalyzedDoc, AnalyzedTerm, FtsAnalyzer, QueryToken, STRING_ARRAY_POSITION_GAP};
pub use block::{
    AttrFtsStats, FtsMeta, POSTING_BLOCK_HEADER_LEN, PackedPostingBlock, PackedPositionBlock,
    PositionBlockView, PostingBlockHeader, PostingBlockView, TERM_STATS_LEN, TermStats,
};
pub use bm25::{Bm25TermScorer, bm25_idf, bm25_qtf_norm};
pub use maxscore::{
    Bm25Result, DEFAULT_MAX_PREFIX_EXPANSIONS, FtsBlockCursor, FtsPostingSource, execute_rank_by,
};
