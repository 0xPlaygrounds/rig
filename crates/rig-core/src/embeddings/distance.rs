//! Similarity and distance of embedding vectors.
//!
//! [`Embedding`](crate::embeddings::Embedding) reductions use fixed chunks and
//! left-to-right summation for reproducible ordering.
//!
//! ```
//! use rig_core::embeddings::{Embedding, distance::VectorDistance};
//!
//! let vector = Embedding { document: String::new(), vec: vec![1.0, 0.0] };
//! assert_eq!(vector.cosine_similarity(&vector, true), 1.0);
//! assert_eq!(vector.euclidean_distance(&vector), 0.0);
//! ```

/// Similarity and distance metrics of embedding vectors.
/// Supply equal-length vectors; the embedding implementation pairs only their
/// shared prefix. Unnormalized cosine requires nonzero magnitudes.
pub trait VectorDistance {
    /// The dot product.
    fn dot_product(&self, other: &Self) -> f64;

    /// Cosine similarity, from -1 to 1.
    /// If `normalized` is true, the vectors are taken as unit length and the
    /// dot product is returned.
    fn cosine_similarity(&self, other: &Self, normalized: bool) -> f64;

    /// The angle between the vectors as a fraction of pi, from 0 (same
    /// direction) to 1 (opposite). `normalized` is as for
    /// [`Self::cosine_similarity`].
    fn angular_distance(&self, other: &Self, normalized: bool) -> f64 {
        // Roundoff can push a valid cosine beyond the domain of acos.
        self.cosine_similarity(other, normalized)
            .clamp(-1.0, 1.0)
            .acos()
            / std::f64::consts::PI
    }

    /// The straight-line (L2) distance.
    fn euclidean_distance(&self, other: &Self) -> f64;

    /// The sum of the coordinates' absolute differences (L1).
    fn manhattan_distance(&self, other: &Self) -> f64;

    /// The largest absolute difference of a coordinate (L-infinity).
    fn chebyshev_distance(&self, other: &Self) -> f64;
}

/// Reduction chunk size. Preserve left-to-right summation within and across
/// chunks to keep floating-point results reproducible.
const CHUNK: usize = 256;

/// `term` of each pair of `a` and `b` summed in fixed chunks, left to right
/// within and across them.
fn pair_sum(a: &[f64], b: &[f64], term: impl Fn(f64, f64) -> f64) -> f64 {
    a.chunks(CHUNK)
        .zip(b.chunks(CHUNK))
        .map(|(a, b)| a.iter().zip(b).map(|(x, y)| term(*x, *y)).sum::<f64>())
        .sum()
}

impl VectorDistance for crate::embeddings::Embedding {
    fn dot_product(&self, other: &Self) -> f64 {
        dot(&self.vec, &other.vec)
    }

    fn cosine_similarity(&self, other: &Self, normalized: bool) -> f64 {
        let dot_product = dot(&self.vec, &other.vec);
        if normalized {
            return dot_product;
        }
        dot_product / (magnitude(&self.vec) * magnitude(&other.vec))
    }

    fn euclidean_distance(&self, other: &Self) -> f64 {
        pair_sum(&self.vec, &other.vec, |x, y| (x - y).powi(2)).sqrt()
    }

    fn manhattan_distance(&self, other: &Self) -> f64 {
        pair_sum(&self.vec, &other.vec, |x, y| (x - y).abs())
    }

    fn chebyshev_distance(&self, other: &Self) -> f64 {
        self.vec
            .iter()
            .zip(&other.vec)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }
}

/// The dot product of `a` and `b`, summed as [`VectorDistance`] sums it.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    pair_sum(a, b, |x, y| x * y)
}

/// The length of `vec`, summed as [`VectorDistance`] sums it.
pub(crate) fn magnitude(vec: &[f64]) -> f64 {
    pair_sum(vec, vec, |x, _| x.powi(2)).sqrt()
}

#[cfg(test)]
mod tests;
