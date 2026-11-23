//! 256-row columnar block codec (`cf_columns`, `cf_detached`).
//!
//! Wire layout (`storage-format.md` §4.4):
//! - Header (8 B): `[version u8 = 1][kind u8][width u8][flags u8][present_count u16 LE][reserved u16 = 0]`
//!   - `kind`: matches `AttributeType as u8` (`1=String, 2=Int, 3=Uint, 4=Float, 5=Bool, 6=StringArray, 7=IntArray, 8=UintArray`)
//!   - `flags & 0x01`: `ALL_PRESENT` (`present: [u64; 4]` LE bitmap omitted when all 256 slots present)
//!   - `flags & 0x02`: `FOR` (frame-of-reference base `u64`/`i64` LE preceding narrow deltas `width` in `0/1/2/4`)
//! - Present bitmap (32 B `[u64; 4]` LE, unless `ALL_PRESENT`)
//! - Payload per `kind` (slot-indexed over `0..256`).

mod builder;
mod view;
#[cfg(test)]
mod tests;

pub use builder::*;
pub use view::*;


const COLUMN_BLOCK_VERSION: u8 = 1;
const COL_HEADER_LEN: usize = 8;
const COL_FLAG_ALL_PRESENT: u8 = 0x01;
const COL_FLAG_FOR: u8 = 0x02;
