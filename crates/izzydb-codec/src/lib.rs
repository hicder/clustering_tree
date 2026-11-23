//! Binary codecs: keyspace layout, order-preserving scalar encodings, posting/segment
//! metadata formats, and the packed row format.

mod consts;
mod doc_id;
mod keys;
mod ordered;
mod posting;
mod row;
mod segment;
mod terms;

#[cfg(test)]
mod tests;

pub use consts::*;
pub use doc_id::*;
pub use keys::*;
pub use ordered::*;
pub use posting::*;
pub use row::*;
pub use segment::*;
pub use terms::*;
