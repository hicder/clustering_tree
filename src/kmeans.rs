use crate::config::DistanceMetric;
use crate::metric::{cluster_distance, compute_weighted_centroid};

/// Result of partitioning items into two balanced clusters.
pub struct SplitResult {
    pub part1: Vec<usize>,
    pub part2: Vec<usize>,
    pub centroid1: Vec<f32>,
    pub centroid2: Vec<f32>,
}

/// Deterministic, balanced 2-means clustering for both leaves and internal nodes.
///
/// Each item is `(vec, weight)` where `weight` is `1` for leaf vectors or `subtree_count`
/// for internal child entries. Guarantees both partitions have at least `min_split` items
/// (and at most `max_split` when `items.len() <= 2 * max_split`).
pub fn balanced_two_means(
    items: &[(&[f32], u64)],
    min_split: usize,
    max_split: usize,
    dim: usize,
    metric: DistanceMetric,
) -> SplitResult {
    let n = items.len();
    debug_assert!(n >= 2);

    let target_min = min_split.max(n.saturating_sub(max_split)).min(n / 2);

    // Compute overall centroid to initialize deterministic k-means++ seeds.
    let overall = compute_weighted_centroid(items, dim, metric);

    let mut idx1 = 0usize;
    let mut max_d1 = -1.0f32;
    for (i, &(v, _)) in items.iter().enumerate() {
        let d = cluster_distance(metric, v, &overall);
        if d > max_d1 {
            max_d1 = d;
            idx1 = i;
        }
    }

    let mut idx2 = 0usize;
    let mut max_d2 = -1.0f32;
    for (i, &(v, _)) in items.iter().enumerate() {
        if i == idx1 {
            continue;
        }
        let d = cluster_distance(metric, v, items[idx1].0);
        if d > max_d2 {
            max_d2 = d;
            idx2 = i;
        }
    }

    // Degenerate case: all vectors are identical (or numerically indistinguishable).
    if max_d2 <= 1e-12 {
        return even_split(items, dim, metric);
    }

    let mut c1 = items[idx1].0.to_vec();
    let mut c2 = items[idx2].0.to_vec();
    let mut assign = vec![0u8; n];

    const MAX_ITERS: usize = 15;
    for _ in 0..MAX_ITERS {
        let mut changed = false;
        let mut count0 = 0usize;
        let mut count1 = 0usize;

        for (i, &(v, _)) in items.iter().enumerate() {
            let d1 = cluster_distance(metric, v, &c1);
            let d2 = cluster_distance(metric, v, &c2);
            let next = if d1 <= d2 { 0 } else { 1 };
            if assign[i] != next {
                assign[i] = next;
                changed = true;
            }
            if next == 0 {
                count0 += 1;
            } else {
                count1 += 1;
            }
        }

        if count0 == 0 || count1 == 0 {
            return even_split(items, dim, metric);
        }

        let p1_items: Vec<(&[f32], u64)> = items
            .iter()
            .enumerate()
            .filter(|&(i, _)| assign[i] == 0)
            .map(|(_, &item)| item)
            .collect();
        let p2_items: Vec<(&[f32], u64)> = items
            .iter()
            .enumerate()
            .filter(|&(i, _)| assign[i] == 1)
            .map(|(_, &item)| item)
            .collect();

        c1 = compute_weighted_centroid(&p1_items, dim, metric);
        c2 = compute_weighted_centroid(&p2_items, dim, metric);

        if !changed {
            break;
        }
    }

    let mut part1 = Vec::new();
    let mut part2 = Vec::new();
    for (i, &a) in assign.iter().enumerate() {
        if a == 0 {
            part1.push(i);
        } else {
            part2.push(i);
        }
    }

    // Enforce minimum split balance by moving lowest-margin items from the larger partition.
    if part1.len() < target_min {
        rebalance(&mut part1, &mut part2, &c1, &c2, target_min, items, metric);
    } else if part2.len() < target_min {
        rebalance(&mut part2, &mut part1, &c2, &c1, target_min, items, metric);
    }

    let p1_items: Vec<(&[f32], u64)> = part1.iter().map(|&i| items[i]).collect();
    let p2_items: Vec<(&[f32], u64)> = part2.iter().map(|&i| items[i]).collect();
    let centroid1 = compute_weighted_centroid(&p1_items, dim, metric);
    let centroid2 = compute_weighted_centroid(&p2_items, dim, metric);

    SplitResult {
        part1,
        part2,
        centroid1,
        centroid2,
    }
}

fn even_split(items: &[(&[f32], u64)], dim: usize, metric: DistanceMetric) -> SplitResult {
    let mid = items.len() / 2;
    let part1: Vec<usize> = (0..mid).collect();
    let part2: Vec<usize> = (mid..items.len()).collect();
    let p1_items: Vec<(&[f32], u64)> = part1.iter().map(|&i| items[i]).collect();
    let p2_items: Vec<(&[f32], u64)> = part2.iter().map(|&i| items[i]).collect();
    let centroid1 = compute_weighted_centroid(&p1_items, dim, metric);
    let centroid2 = compute_weighted_centroid(&p2_items, dim, metric);
    SplitResult {
        part1,
        part2,
        centroid1,
        centroid2,
    }
}

fn rebalance(
    small: &mut Vec<usize>,
    large: &mut Vec<usize>,
    c_small: &[f32],
    c_large: &[f32],
    target_min: usize,
    items: &[(&[f32], u64)],
    metric: DistanceMetric,
) {
    let needed = target_min.saturating_sub(small.len());
    if needed == 0 {
        return;
    }

    let mut scored: Vec<(usize, f32)> = large
        .iter()
        .map(|&idx| {
            let v = items[idx].0;
            let margin = cluster_distance(metric, v, c_small) - cluster_distance(metric, v, c_large);
            (idx, margin)
        })
        .collect();

    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let move_count = needed.min(scored.len());
    let mut moved = Vec::with_capacity(move_count);
    for &(idx, _) in scored.iter().take(move_count) {
        small.push(idx);
        moved.push(idx);
    }
    moved.sort_unstable();
    large.retain(|idx| moved.binary_search(idx).is_err());
}
