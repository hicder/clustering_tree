//! Packed FTS block formats.

mod doclen;
mod meta;
mod position;
mod posting;
mod skip;
#[cfg(test)]
mod tests;

pub use doclen::*;
pub use meta::*;
pub use position::*;
pub use posting::*;
pub use skip::*;


pub const POSTING_BLOCK_HEADER_LEN: usize = 12;
pub const DOC_LEN_BLOCK_HEADER_LEN: usize = 4;
pub const SKIP_BLOCK_HEADER_LEN: usize = 4;
pub const TERM_STATS_LEN: usize = 16;
pub const TERM_RECORD_HEADER_LEN: usize = 21;

/// Minimum posting count in a 256-doc block at which posting encoding switches from Sparse to Dense.
pub const POSTING_DENSE_THRESHOLD: usize = 96;

/// Maximum `doc_freq` (within a single block) for inlining a posting block directly inside `TermRecord`.
pub const INLINE_TERM_MAX_DF: usize = 8;

const POSTING_VERSION_V3: u16 = 0x0300;
const POSTING_VERSION_MASK: u16 = 0xFF00;
const FLAG_DENSE: u16 = 0x0001;
const WIDTH_SHIFT: u16 = 1;
const WIDTH_MASK: u16 = 0x0006;

const TERM_FLAG_INLINE: u8 = 0x01;
const POS_FLAG_U32_OFFSETS: u8 = 0x01;
