//! Generalized [`RankBy`] execution on v3 `RowId` / `RowSet` / [`FilterPlan`].
//!
//! | normalized shape                                    | strategy                                              |
//! |-----------------------------------------------------|-------------------------------------------------------|
//! | pure lexical                                        | MAXSCORE with `LivenessMaskProvider`                  |
//! | `L * m(row) + a(row)` (one lexical factor + mods)   | MAXSCORE on `L * UB(m)` + exact `RowView` rescoring   |
//! | no BM25 (`Filter` / `Attribute` / `Saturate` / ...) | index-order `stream_value_groups` if monotone, else `RowSet` scan |
//! | anything else (e.g. `BM25 * BM25`)                  | exact evaluation over the positive `RowSet` universe  |
//!
//! Every path returns the exact top-k under `score DESC, DocId ASC`; scores `<= 0` are excluded.

mod exec;
mod monotone;
mod node;
mod plan;
mod scan;
mod topk;

use node::*;
use plan::*;
use scan::*;
use topk::*;




const SCAN_BATCH: usize = 1024;

// ------------------------------------------------------------------------------------------
// Compiled per-row evaluation tree
// ------------------------------------------------------------------------------------------
