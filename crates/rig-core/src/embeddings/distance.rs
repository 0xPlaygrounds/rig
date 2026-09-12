//! Distance and similarity helpers for embedding vectors.
//!
//! The [`VectorDistance`] implementation for [`Embedding`](crate::embeddings::Embedding)
//! sums in fixed chunks on one thread, so every metric is the same bits on
//! every run.

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

/// The sums behind every metric are taken in fixed chunks: each chunk is
/// summed left to right, then the chunk sums are summed left to right, so
/// the result is the same bits on every run. (A parallel `sum()` that
/// combined partial sums in whatever order the threads finished once gave
/// a cosine score that differed in its last digit from one run to the
/// next, and turned a recorded retrieval into a record that was never the
/// same twice; the chunking is kept so a parallel build, if one returns,
/// has the order it must reproduce.)
const CHUNK: usize = 256;

/// Generates the [`VectorDistance`] method bodies for [`Embedding`](crate::embeddings::Embedding)
/// from one pairwise sum, one unary sum and one max-reduction.
macro_rules! impl_vector_distance {
    ($pair_sum:ident, $unary_sum:ident, $pair_max:ident) => {
        fn dot_product(&self, other: &Self) -> f64 {
            $pair_sum(&self.vec, &other.vec, |x, y| x * y)
        }

        fn cosine_similarity(&self, other: &Self, normalized: bool) -> f64 {
            let dot_product = self.dot_product(other);

            if normalized {
                dot_product
            } else {
                let magnitude1: f64 = $unary_sum(&self.vec, |x| x.powi(2)).sqrt();
                let magnitude2: f64 = $unary_sum(&other.vec, |x| x.powi(2)).sqrt();

                dot_product / (magnitude1 * magnitude2)
            }
        }

        fn angular_distance(&self, other: &Self, normalized: bool) -> f64 {
            let cosine_sim = self.cosine_similarity(other, normalized);
            cosine_sim.acos() / std::f64::consts::PI
        }

        fn euclidean_distance(&self, other: &Self) -> f64 {
            $pair_sum(&self.vec, &other.vec, |x, y| (x - y).powi(2)).sqrt()
        }

        fn manhattan_distance(&self, other: &Self) -> f64 {
            $pair_sum(&self.vec, &other.vec, |x, y| (x - y).abs())
        }

        fn chebyshev_distance(&self, other: &Self) -> f64 {
            $pair_max(&self.vec, &other.vec, |x, y| (x - y).abs())
        }
    };
}

mod sequential {
    use super::{CHUNK, VectorDistance};
    use crate::embeddings::Embedding;

    /// Fixed chunks, two left-to-right sums, one thread.
    fn pair_sum(a: &[f64], b: &[f64], term: impl Fn(f64, f64) -> f64) -> f64 {
        a.chunks(CHUNK)
            .zip(b.chunks(CHUNK))
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| term(*x, *y)).sum::<f64>())
            .sum()
    }

    fn unary_sum(a: &[f64], term: impl Fn(f64) -> f64) -> f64 {
        a.chunks(CHUNK)
            .map(|a| a.iter().map(|x| term(*x)).sum::<f64>())
            .sum()
    }

    fn pair_max(a: &[f64], b: &[f64], term: impl Fn(f64, f64) -> f64) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| term(*x, *y))
            .fold(0.0, f64::max)
    }

    impl VectorDistance for Embedding {
        impl_vector_distance!(pair_sum, unary_sum, pair_max);
    }
}

#[cfg(test)]
mod tests;
