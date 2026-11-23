//! Packed row format v3 (`cf_rows`): writer, zero-copy view, raw scalar comparisons, decoding.

mod decode;
mod scalar;
mod view;
mod writer;

pub use decode::*;
pub use scalar::*;
pub use view::*;
use decode::decode_payload_value;
use scalar::{raw_cmp_scalar, raw_contains_scalar, raw_eq_scalar};
use writer::ByteReader;
pub use writer::encode_row;
