//! Distance and similarity helpers for embedding vectors.
//!
//! The [`VectorDistance`] implementation for [`Embedding`](crate::embeddings::Embedding)
//! uses iterator-based calculations by default and switches to Rayon-backed
//! parallel iterators when the `rayon` feature is enabled.

/// Distance and similarity metrics for embedding vectors.
pub trait VectorDistance {
    /// Get dot product of two embedding vectors
    fn dot_product(&self, other: &Self) -> f64;

    /// Get cosine similarity of two embedding vectors.
    /// If `normalized` is true, the dot product is returned.
    fn cosine_similarity(&self, other: &Self, normalized: bool) -> f64;

    /// Get angular distance of two embedding vectors.
    fn angular_distance(&self, other: &Self, normalized: bool) -> f64;

    /// Get euclidean distance of two embedding vectors.
    fn euclidean_distance(&self, other: &Self) -> f64;

    /// Get manhattan distance of two embedding vectors.
    fn manhattan_distance(&self, other: &Self) -> f64;

    /// Get chebyshev distance of two embedding vectors.
    fn chebyshev_distance(&self, other: &Self) -> f64;
}

/// Generates the [`VectorDistance`] method bodies for [`Embedding`](crate::embeddings::Embedding).
///
/// The math is identical between the sequential and Rayon-backed implementations;
/// only the iterator constructor (`iter` vs `par_iter`) and the max-reduction
/// differ (`Iterator::fold` vs `ParallelIterator::reduce`), so both are supplied
/// by the caller. Keeping one source prevents the two copies from drifting.
macro_rules! impl_vector_distance {
    ($iter:ident, $($max_reduce:tt)+) => {
        fn dot_product(&self, other: &Self) -> f64 {
            self.vec
                .$iter()
                .zip(other.vec.$iter())
                .map(|(x, y)| x * y)
                .sum()
        }

        fn cosine_similarity(&self, other: &Self, normalized: bool) -> f64 {
            let dot_product = self.dot_product(other);

            if normalized {
                dot_product
            } else {
                let magnitude1: f64 = self.vec.$iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                let magnitude2: f64 = other.vec.$iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

                dot_product / (magnitude1 * magnitude2)
            }
        }

        fn angular_distance(&self, other: &Self, normalized: bool) -> f64 {
            let cosine_sim = self.cosine_similarity(other, normalized);
            cosine_sim.acos() / std::f64::consts::PI
        }

        fn euclidean_distance(&self, other: &Self) -> f64 {
            self.vec
                .$iter()
                .zip(other.vec.$iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
        }

        fn manhattan_distance(&self, other: &Self) -> f64 {
            self.vec
                .$iter()
                .zip(other.vec.$iter())
                .map(|(x, y)| (x - y).abs())
                .sum()
        }

        fn chebyshev_distance(&self, other: &Self) -> f64 {
            self.vec
                .$iter()
                .zip(other.vec.$iter())
                .map(|(x, y)| (x - y).abs())
                .$($max_reduce)+
        }
    };
}

#[cfg(not(feature = "rayon"))]
impl VectorDistance for crate::embeddings::Embedding {
    impl_vector_distance!(iter, fold(0.0, f64::max));
}

#[cfg(feature = "rayon")]
mod rayon {
    use crate::embeddings::{Embedding, distance::VectorDistance};
    use rayon::prelude::*;

    impl VectorDistance for Embedding {
        // `ParallelIterator` has no scalar `fold`; use `reduce` (0.0 is a valid
        // identity for `max` since every mapped value is a non-negative abs diff).
        impl_vector_distance!(par_iter, reduce(|| 0.0, f64::max));
    }
}

#[cfg(test)]
mod tests;
