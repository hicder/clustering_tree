use std::borrow::Cow;

use crate::config::DistanceMetric;
use crate::error::{Result, TreeError};
use crate::simd;

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

/// Validate and borrow `v` when no normalization is needed, or allocate only for `Cosine`.
#[inline]
pub fn prepare_vector_cow<'a>(
    v: &'a [f32],
    dim: usize,
    metric: DistanceMetric,
) -> Result<Cow<'a, [f32]>> {
    validate_vector(v, dim)?;
    if metric == DistanceMetric::Cosine {
        let mut out = v.to_vec();
        normalize_in_place(&mut out);
        Ok(Cow::Owned(out))
    } else {
        Ok(Cow::Borrowed(v))
    }
}

/// Squared Euclidean distance using SIMD.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    simd::l2_squared(a, b)
}

/// Inner product using SIMD.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    simd::dot_product(a, b)
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

/// Search distance against an unaligned little-endian `f32` byte slice.
#[inline]
pub fn search_distance_bytes(metric: DistanceMetric, a: &[f32], b_bytes: &[u8]) -> f32 {
    match metric {
        DistanceMetric::L2Squared => simd::l2_squared_bytes(a, b_bytes),
        DistanceMetric::Cosine => (1.0 - simd::dot_product_bytes(a, b_bytes)).max(0.0),
        DistanceMetric::DotProduct => -simd::dot_product_bytes(a, b_bytes),
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

/// Compute the unweighted centroid of `count` packed vectors of length `dim` stored contiguously.
pub fn compute_mean_centroid_flat(vecs: &[f32], count: usize, dim: usize, metric: DistanceMetric) -> Vec<f32> {
    if count == 0 {
        return vec![0.0; dim];
    }
    let mut acc = vec![0.0f64; dim];
    for v in vecs.chunks_exact(dim) {
        for (a, &x) in acc.iter_mut().zip(v.iter()) {
            *a += x as f64;
        }
    }
    let inv = 1.0 / (count as f64);
    let mut out: Vec<f32> = acc.into_iter().map(|s| (s * inv) as f32).collect();
    if metric == DistanceMetric::Cosine {
        normalize_in_place(&mut out);
    }
    out
}

/// Compute the subtree-weighted centroid of packed `centroids` (`count * dim`) with `weights`.
pub fn compute_weighted_centroid_flat(
    centroids: &[f32],
    weights: &[u64],
    dim: usize,
    metric: DistanceMetric,
) -> Vec<f32> {
    if weights.is_empty() {
        return vec![0.0; dim];
    }
    let total_weight: u64 = weights.iter().sum();
    if total_weight == 0 {
        return compute_mean_centroid_flat(centroids, weights.len(), dim, metric);
    }
    let mut acc = vec![0.0f64; dim];
    for (v, &w) in centroids.chunks_exact(dim).zip(weights.iter()) {
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
