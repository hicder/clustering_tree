//! Top-k collection (`score DESC, DocId ASC`).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use izzydb_core::config::DocId;
use izzydb_fts::Bm25Result;



// ------------------------------------------------------------------------------------------
// Compiled per-row evaluation tree
// ------------------------------------------------------------------------------------------


#[derive(Debug, Clone, Copy)]
pub(super) struct Ranked {
    pub(super) score: f32,
    pub(super) id: DocId,
}

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    /// "Greater" means *worse* (lower score, then higher id) so the max-heap root is evicted first.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.id.cmp(&other.id))
    }
}

pub(super) struct TopK {
    pub(super) k: usize,
    pub(super) heap: BinaryHeap<Ranked>,
}

impl TopK {
    pub(super) fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.min(4096) + 1),
        }
    }

    pub(super) fn push(&mut self, id: DocId, score: f32) {
        let r = Ranked { score, id };
        if self.heap.len() < self.k {
            self.heap.push(r);
        } else if let Some(worst) = self.heap.peek()
            && r < *worst
        {
            self.heap.pop();
            self.heap.push(r);
        }
    }

    pub(super) fn into_sorted(self) -> Vec<Bm25Result> {
        let mut v = self.heap.into_vec();
        v.sort_unstable();
        v.into_iter()
            .map(|r| Bm25Result {
                id: r.id,
                score: r.score,
            })
            .collect()
    }
}

pub(super) fn sort_results(v: &mut [Bm25Result]) {
    v.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
}

// ------------------------------------------------------------------------------------------
// Planning
// ------------------------------------------------------------------------------------------
