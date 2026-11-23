//! Phase 8: Detached columns (`cf_detached`, `cf_detached_postings`, `default[0x07/0x13/0x14]`),
//! atomic whole-column build/publish ingest, point overwrites, and CoW carry-over.

mod filter_eval;
mod ingest;
mod point_write;

pub(crate) use filter_eval::*;
pub use ingest::*;
pub(crate) use point_write::*;





const DEFAULT_SST_BLOCK_THRESHOLD: usize = 64;
