//! Vectorized Block-Max MAXSCORE BM25 query compiler and execution engine.

mod clause;
mod compile;
mod exec;
mod params;
mod stream;
#[cfg(test)]
mod tests;
mod traits;

use clause::*;
use compile::*;
pub use exec::*;
pub use params::*;
use stream::*;
pub use traits::*;





pub const DEFAULT_MAX_PREFIX_EXPANSIONS: usize = 128;
