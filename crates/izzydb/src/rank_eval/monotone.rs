//! Index-order monotone rank queries.


use izzydb_core::error::Result;
use izzydb_core::filter::Filter;
use izzydb_core::rank::{DecayMode, OrderBySpec, RankBy};
use izzydb_core::schema::ScalarValue;
use izzydb_fts::Bm25Result;

use crate::collection::Collection;
use crate::filter_plan::{self, FilterPlan, FilterPlanTrace};
use crate::order_by::OrderPopulation;
use crate::segments::RowResolver;


// ------------------------------------------------------------------------------------------
// Compiled per-row evaluation tree
// ------------------------------------------------------------------------------------------

use super::*;

impl Collection {
    pub(super) fn monotone_rank_query(
        &self,
        norm: &RankBy,
        filter: Option<&Filter>,
        top_k: usize,
    ) -> Result<Option<Vec<Bm25Result>>> {
        let (weight, leaf) = match norm {
            RankBy::Product(c) if c.len() == 2 => match (&c[0], &c[1]) {
                (RankBy::Const(w), leaf) => (Some(*w), leaf),
                _ => return Ok(None),
            },
            leaf => (None, leaf),
        };
        let (attr, descending, missing) = match leaf {
            RankBy::Attribute { attr, missing } => (attr, true, *missing),
            RankBy::Saturate { attr, missing, .. } => (attr, true, *missing),
            RankBy::Decay {
                attr,
                mode,
                missing,
                ..
            } => match mode {
                DecayMode::Forward => (attr, false, *missing),
                DecayMode::Backward => (attr, true, *missing),
                DecayMode::Symmetric => return Ok(None),
            },
            _ => return Ok(None),
        };
        let Some(entry) = self.schema.get_by_name(attr) else {
            return Ok(None);
        };
        if missing != 0.0 || !entry.filterable || weight.is_some_and(|w| w <= 0.0) {
            return Ok(None);
        }

        let mut no_lex = Vec::new();
        let node = Node::compile(norm, &self.schema, &mut no_lex);
        let leaf_node = Node::compile(leaf, &self.schema, &mut no_lex);
        let pop = match filter {
            None => OrderPopulation::AllLive,
            Some(f) => {
                let mut trace = FilterPlanTrace::default();
                match filter_plan::plan_filter_with_trace(
                    &self.storage,
                    &self.schema,
                    f,
                    &self.fts_index,
                    &mut trace,
                )? {
                    FilterPlan::Empty => return Ok(Some(Vec::new())),
                    FilterPlan::All => OrderPopulation::AllLive,
                    FilterPlan::Materialized(rs) => {
                        if rs.is_empty() {
                            return Ok(Some(Vec::new()));
                        }
                        OrderPopulation::RowSet(rs)
                    }
                    FilterPlan::Lazy { residual, seed } => OrderPopulation::Lazy { seed, residual },
                }
            }
        };

        let spec = if descending {
            OrderBySpec::desc(attr.clone())
        } else {
            OrderBySpec::asc(attr.clone())
        };
        let score_of = |value: &ScalarValue| -> Option<f32> {
            let v = match value {
                ScalarValue::Int(v) => *v as f64,
                ScalarValue::Uint(v) => *v as f64,
                ScalarValue::Float(v) => *v,
                _ => return None,
            };
            Some(eval_value(&node, &leaf_node, v))
        };

        let mut collected: Vec<Bm25Result> = Vec::new();
        let mut boundary: Option<f32> = None;
        self.stream_value_groups(&spec, &pop, &mut |value, group_rows| {
            let Some(score) = score_of(&value) else {
                return Ok(true);
            };
            if score <= 0.0 || score.is_nan() {
                return Ok(false);
            }
            if let Some(b) = boundary
                && score < b
            {
                return Ok(false);
            }
            let doc_ids = RowResolver::resolve_all(&self.storage, &group_rows)?;
            collected.extend(doc_ids.into_iter().map(|id| Bm25Result { id, score }));
            if boundary.is_none() && collected.len() >= top_k {
                boundary = Some(score);
            }
            Ok(true)
        })?;
        sort_results(&mut collected);
        collected.truncate(top_k);
        Ok(Some(collected))
    }
}
