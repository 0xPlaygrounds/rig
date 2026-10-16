//! Similarity of embedding vectors.
//!
//! [`Embedding`](crate::embeddings::Embedding) reductions use fixed chunks and
//! left-to-right summation for reproducible ordering.
//!
//! ```
//! use rig_core::embeddings::{Embedding, distance::VectorDistance};
//!
//! let vector = Embedding { document: String::new(), vec: vec![1.0, 0.0] };
//! assert_eq!(vector.cosine_similarity(&vector, true), 1.0);
//! ```

/// Similarity of embedding vectors.
/// Supply equal-length vectors; the embedding implementation pairs only their
/// shared prefix. Unnormalized cosine requires nonzero magnitudes.
pub trait VectorDistance {
    /// Get cosine similarity of two embedding vectors.
    /// If `normalized` is true, the dot product is returned.
    fn cosine_similarity(&self, other: &Self, normalized: bool) -> f64;
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
    fn cosine_similarity(&self, other: &Self, normalized: bool) -> f64 {
        let dot_product = dot(&self.vec, &other.vec);
        if normalized {
            return dot_product;
        }
        dot_product / (magnitude(&self.vec) * magnitude(&other.vec))
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
