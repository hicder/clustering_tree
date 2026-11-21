//! 1-bit RaBitQ quantization, randomized FWHT rotation, and SIMD popcount leaf scoring for `izzydb`.

pub mod quantizer;
pub mod simd;

pub use quantizer::{
    EncodeScratch, LeafQueryState, QuantizedEntry, QuantizedQueryBits, QuantizedQueryMeta,
    QueryScratch, RabitqQuantizer, pack_query_bits_from_diff_bytes_into, pack_query_bits_into,
};
pub use simd::{LeafCandidate, RabitqLeafScoreParams, score_leaf_rabitq};
