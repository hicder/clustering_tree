use crate::config::DistanceMetric;
use crate::error::{Result, TreeError};

/// Validate vector length and ensure all components are finite.
pub fn validate_vector(v: &[f32], expected_dim: usize) -> Result<()> {
    if v.len() != expected_dim {
        return Err(TreeError::DimensionMismatch {
            expected: expected_dim,
            actual: v.len(),
        });
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(TreeError::NonFiniteVector);
    }
    Ok(())
}

/// L2-normalize `v` in place if its norm is positive.
pub fn normalize_in_place(v: &mut [f32]) {
    let norm_sq: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    if norm_sq > 0.0 {
        let inv = (1.0 / norm_sq.sqrt()) as f32;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Validate and optionally L2-normalize an input vector according to `metric`.
pub fn prepare_vector(v: &[f32], dim: usize, metric: DistanceMetric) -> Result<Vec<f32>> {
    validate_vector(v, dim)?;
    let mut out = v.to_vec();
    if metric == DistanceMetric::Cosine {
        normalize_in_place(&mut out);
    }
    Ok(out)
}

/// Squared Euclidean distance using a 4-way unrolled accumulator for auto-vectorization.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    let (chunks_a, rem_a) = a.as_chunks::<4>();
    let (chunks_b, rem_b) = b.as_chunks::<4>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b.iter()) {
        let d0 = ca[0] - cb[0];
        let d1 = ca[1] - cb[1];
        let d2 = ca[2] - cb[2];
        let d3 = ca[3] - cb[3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    let mut sum = (s0 + s1) + (s2 + s3);
    for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Inner product using a 4-way unrolled accumulator for auto-vectorization.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    let (chunks_a, rem_a) = a.as_chunks::<4>();
    let (chunks_b, rem_b) = b.as_chunks::<4>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b.iter()) {
        s0 += ca[0] * cb[0];
        s1 += ca[1] * cb[1];
        s2 += ca[2] * cb[2];
        s3 += ca[3] * cb[3];
    }

    let mut sum = (s0 + s1) + (s2 + s3);
    for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
        sum += x * y;
    }
    sum
}

/// Search distance where smaller means more similar.
///
/// For `DotProduct`, returns `-dot(a, b)` so larger inner products sort first.
#[inline]
pub fn search_distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::L2Squared => l2_squared(a, b),
        DistanceMetric::Cosine => (1.0 - dot_product(a, b)).max(0.0),
        DistanceMetric::DotProduct => -dot_product(a, b),
    }
}

/// Spatial distance used for clustering, routing inserts, drift checks, and LIRE.
///
/// Uses L2² for `DotProduct` so clusters remain spatially compact and drift is zero when unchanged.
#[inline]
pub fn cluster_distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::L2Squared | DistanceMetric::DotProduct => l2_squared(a, b),
        DistanceMetric::Cosine => (1.0 - dot_product(a, b)).max(0.0),
    }
}

/// Compute the unweighted centroid of a slice of vectors.
pub fn compute_mean_centroid(items: &[&[f32]], dim: usize, metric: DistanceMetric) -> Vec<f32> {
    if items.is_empty() {
        return vec![0.0; dim];
    }
    let mut acc = vec![0.0f64; dim];
    for v in items {
        for (a, &x) in acc.iter_mut().zip(v.iter()) {
            *a += x as f64;
        }
    }
    let inv = 1.0 / (items.len() as f64);
    let mut out: Vec<f32> = acc.into_iter().map(|s| (s * inv) as f32).collect();
    if metric == DistanceMetric::Cosine {
        normalize_in_place(&mut out);
    }
    out
}

/// Compute the subtree-weighted centroid of `(centroid, weight)` pairs.
///
/// Falls back to unweighted mean if total weight is 0.
pub fn compute_weighted_centroid(
    items: &[(&[f32], u64)],
    dim: usize,
    metric: DistanceMetric,
) -> Vec<f32> {
    if items.is_empty() {
        return vec![0.0; dim];
    }
    let total_weight: u64 = items.iter().map(|&(_, w)| w).sum();
    if total_weight == 0 {
        let refs: Vec<&[f32]> = items.iter().map(|&(v, _)| v).collect();
        return compute_mean_centroid(&refs, dim, metric);
    }
    let mut acc = vec![0.0f64; dim];
    for &(v, w) in items {
        if w == 0 {
            continue;
        }
        let wf = w as f64;
        for (a, &x) in acc.iter_mut().zip(v.iter()) {
            *a += (x as f64) * wf;
        }
    }
    let inv = 1.0 / (total_weight as f64);
    let mut out: Vec<f32> = acc.into_iter().map(|s| (s * inv) as f32).collect();
    if metric == DistanceMetric::Cosine {
        normalize_in_place(&mut out);
    }
    out
}

/// Incrementally update a centroid after adding one vector (`new_count` includes the new vector).
pub fn incremental_mean(
    old_centroid: &[f32],
    new_vec: &[f32],
    new_count: usize,
    metric: DistanceMetric,
) -> Vec<f32> {
    if new_count <= 1 {
        let mut out = new_vec.to_vec();
        if metric == DistanceMetric::Cosine {
            normalize_in_place(&mut out);
        }
        return out;
    }
    let n = new_count as f64;
    let mut out: Vec<f32> = old_centroid
        .iter()
        .zip(new_vec.iter())
        .map(|(&c, &x)| (c as f64 + (x as f64 - c as f64) / n) as f32)
        .collect();
    if metric == DistanceMetric::Cosine {
        normalize_in_place(&mut out);
    }
    out
}
