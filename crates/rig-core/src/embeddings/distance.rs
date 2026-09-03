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

/// The sums behind every metric are taken in fixed chunks: each chunk is
/// summed left to right, then the chunk sums are summed left to right.
/// With the `rayon` feature the chunks are computed in parallel and their
/// sums combined in index order, so the result is the same bits on every
/// run and the same bits as the sequential build. (A `par_iter().sum()`
/// combines partial sums in whatever order the threads finish, and a
/// cosine score that differed in its last digit from one run to the next
/// turned a recorded retrieval into a record that was never the same
/// twice.)
const CHUNK: usize = 256;

/// The sum of `terms`, in `CHUNK`s, in order.
#[cfg(not(feature = "rayon"))]
fn chunked_sum(terms: impl ExactSizeIterator<Item = f64>) -> f64 {
    let mut chunks = Vec::with_capacity(terms.len().div_ceil(CHUNK).max(1));
    let mut chunk = 0.0;
    for (index, term) in terms.enumerate() {
        chunk += term;
        if (index + 1) % CHUNK == 0 {
            chunks.push(chunk);
            chunk = 0.0;
        }
    }
    chunks.push(chunk);
    chunks.into_iter().sum()
}

/// Generates the [`VectorDistance`] method bodies for [`Embedding`](crate::embeddings::Embedding)
/// from one pairwise sum, one unary sum and one max-reduction, so the
/// sequential and the Rayon-backed implementations share their math.
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

#[cfg(not(feature = "rayon"))]
mod sequential {
    use super::{VectorDistance, chunked_sum};
    use crate::embeddings::Embedding;

    fn pair_sum(a: &[f64], b: &[f64], term: impl Fn(f64, f64) -> f64) -> f64 {
        chunked_sum(a.iter().zip(b).map(|(x, y)| term(*x, *y)))
    }

    fn unary_sum(a: &[f64], term: impl Fn(f64) -> f64) -> f64 {
        chunked_sum(a.iter().map(|x| term(*x)))
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

#[cfg(feature = "rayon")]
mod rayon {
    use super::{CHUNK, VectorDistance};
    use crate::embeddings::Embedding;
    use rayon::prelude::*;

    /// Each chunk summed left to right on its thread; the chunk sums
    /// combined in index order.
    fn pair_sum(a: &[f64], b: &[f64], term: impl Fn(f64, f64) -> f64 + Sync) -> f64 {
        a.par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| term(*x, *y)).sum::<f64>())
            .collect::<Vec<f64>>()
            .into_iter()
            .sum()
    }

    fn unary_sum(a: &[f64], term: impl Fn(f64) -> f64 + Sync) -> f64 {
        a.par_chunks(CHUNK)
            .map(|a| a.iter().map(|x| term(*x)).sum::<f64>())
            .collect::<Vec<f64>>()
            .into_iter()
            .sum()
    }

    fn pair_max(a: &[f64], b: &[f64], term: impl Fn(f64, f64) -> f64 + Sync) -> f64 {
        // 0.0 is a valid identity for `max`: every term is a non-negative
        // absolute difference.
        a.par_iter()
            .zip(b.par_iter())
            .map(|(x, y)| term(*x, *y))
            .reduce(|| 0.0, f64::max)
    }

    impl VectorDistance for Embedding {
        impl_vector_distance!(pair_sum, unary_sum, pair_max);
    }
}

#[cfg(test)]
mod tests;
