//! Distance and similarity helpers for embedding vectors.
//!
//! [`Embedding`](crate::embeddings::Embedding) reductions use fixed chunks and
//! left-to-right summation for reproducible ordering.
//!
//! ```
//! use rig_core::embeddings::{Embedding, distance::{VectorDistance, VectorDistanceError}};
//!
//! # fn main() -> Result<(), VectorDistanceError> {
//! let vector = Embedding { document: String::new(), vec: vec![1.0, 0.0] };
//! let score = vector.dot_product(&vector)?;
//! assert_eq!(score, 1.0);
//! # Ok(())
//! # }
//! ```

/// Invalid input to a vector distance or similarity calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VectorDistanceError {
    /// The vectors have different lengths.
    #[error("vector dimensions differ: left has {left}, right has {right}")]
    DimensionMismatch {
        /// Number of components in `self`.
        left: usize,
        /// Number of components in `other`.
        right: usize,
    },
}

/// Distance and similarity metrics for embedding vectors.
/// All methods return [`VectorDistanceError::DimensionMismatch`] for unequal
/// lengths. Equal-length inputs retain IEEE floating-point behavior, including
/// non-finite results. Unnormalized cosine requires nonzero magnitudes.
pub trait VectorDistance {
    /// Returns the dot product, or an error if dimensions differ.
    fn dot_product(&self, other: &Self) -> Result<f64, VectorDistanceError>;

    /// Returns cosine similarity, or an error if dimensions differ.
    /// If `normalized` is true, the caller must supply unit vectors and the
    /// dot product is returned without computing magnitudes.
    fn cosine_similarity(&self, other: &Self, normalized: bool)
    -> Result<f64, VectorDistanceError>;

    /// Returns angular distance, or an error if dimensions differ.
    /// If `normalized` is true, the caller must supply unit vectors.
    fn angular_distance(&self, other: &Self, normalized: bool) -> Result<f64, VectorDistanceError>;

    /// Returns Euclidean distance, or an error if dimensions differ.
    fn euclidean_distance(&self, other: &Self) -> Result<f64, VectorDistanceError>;

    /// Returns Manhattan distance, or an error if dimensions differ.
    fn manhattan_distance(&self, other: &Self) -> Result<f64, VectorDistanceError>;

    /// Returns Chebyshev distance, or an error if dimensions differ.
    fn chebyshev_distance(&self, other: &Self) -> Result<f64, VectorDistanceError>;
}

fn check_dimensions(a: &[f64], b: &[f64]) -> Result<(), VectorDistanceError> {
    if a.len() != b.len() {
        return Err(VectorDistanceError::DimensionMismatch {
            left: a.len(),
            right: b.len(),
        });
    }
    Ok(())
}

/// Reduction chunk size. Preserve left-to-right summation within and across
/// chunks to keep floating-point results reproducible.
const CHUNK: usize = 256;

/// Generates the [`VectorDistance`] method bodies for [`Embedding`](crate::embeddings::Embedding)
/// from one pairwise sum, one unary sum and one max-reduction.
macro_rules! impl_vector_distance {
    ($pair_sum:ident, $unary_sum:ident, $pair_max:ident) => {
        fn dot_product(&self, other: &Self) -> Result<f64, VectorDistanceError> {
            check_dimensions(&self.vec, &other.vec)?;
            Ok($pair_sum(&self.vec, &other.vec, |x, y| x * y))
        }

        fn cosine_similarity(
            &self,
            other: &Self,
            normalized: bool,
        ) -> Result<f64, VectorDistanceError> {
            let dot_product = self.dot_product(other)?;

            if normalized {
                Ok(dot_product)
            } else {
                let magnitude1: f64 = $unary_sum(&self.vec, |x| x.powi(2)).sqrt();
                let magnitude2: f64 = $unary_sum(&other.vec, |x| x.powi(2)).sqrt();

                Ok(dot_product / (magnitude1 * magnitude2))
            }
        }

        fn angular_distance(
            &self,
            other: &Self,
            normalized: bool,
        ) -> Result<f64, VectorDistanceError> {
            let cosine_sim = self.cosine_similarity(other, normalized)?;
            // Roundoff can push a valid cosine beyond the domain of acos.
            Ok(cosine_sim.clamp(-1.0, 1.0).acos() / std::f64::consts::PI)
        }

        fn euclidean_distance(&self, other: &Self) -> Result<f64, VectorDistanceError> {
            check_dimensions(&self.vec, &other.vec)?;
            Ok($pair_sum(&self.vec, &other.vec, |x, y| (x - y).powi(2)).sqrt())
        }

        fn manhattan_distance(&self, other: &Self) -> Result<f64, VectorDistanceError> {
            check_dimensions(&self.vec, &other.vec)?;
            Ok($pair_sum(&self.vec, &other.vec, |x, y| (x - y).abs()))
        }

        fn chebyshev_distance(&self, other: &Self) -> Result<f64, VectorDistanceError> {
            check_dimensions(&self.vec, &other.vec)?;
            Ok($pair_max(&self.vec, &other.vec, |x, y| (x - y).abs()))
        }
    };
}

mod sequential {
    use super::{CHUNK, VectorDistance, VectorDistanceError, check_dimensions};
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
