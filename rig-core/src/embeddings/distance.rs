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

#[cfg(not(feature = "rayon"))]
impl VectorDistance for crate::embeddings::Embedding {
    fn dot_product(&self, other: &Self) -> f64 {
        self.vec
            .iter()
            .zip(other.vec.iter())
            .map(|(x, y)| x * y)
            .sum()
    }

    fn cosine_similarity(&self, other: &Self, normalized: bool) -> f64 {
        let dot_product = self.dot_product(other);

        if normalized {
            dot_product
        } else {
            let magnitude1: f64 = self.vec.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            let magnitude2: f64 = other.vec.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

            dot_product / (magnitude1 * magnitude2)
        }
    }

    fn angular_distance(&self, other: &Self, normalized: bool) -> f64 {
        let cosine_sim = self.cosine_similarity(other, normalized);
        cosine_sim.acos() / std::f64::consts::PI
    }

    fn euclidean_distance(&self, other: &Self) -> f64 {
        self.vec
            .iter()
            .zip(other.vec.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn manhattan_distance(&self, other: &Self) -> f64 {
        self.vec
            .iter()
            .zip(other.vec.iter())
            .map(|(x, y)| (x - y).abs())
            .sum()
    }

    fn chebyshev_distance(&self, other: &Self) -> f64 {
        self.vec
            .iter()
            .zip(other.vec.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }
}

#[cfg(feature = "rayon")]
mod rayon {
    use crate::embeddings::{distance::VectorDistance, Embedding};
    use rayon::prelude::*;

    impl VectorDistance for Embedding {
        fn dot_product(&self, other: &Self) -> f64 {
            self.vec
                .par_iter()
                .zip(other.vec.par_iter())
                .map(|(x, y)| x * y)
                .sum()
        }

        fn cosine_similarity(&self, other: &Self, normalized: bool) -> f64 {
            let dot_product = self.dot_product(other);

            if normalized {
                dot_product
            } else {
                let magnitude1: f64 = self.vec.par_iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                let magnitude2: f64 = other.vec.par_iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

                dot_product / (magnitude1 * magnitude2)
            }
        }

        fn angular_distance(&self, other: &Self, normalized: bool) -> f64 {
            let cosine_sim = self.cosine_similarity(other, normalized);
            cosine_sim.acos() / std::f64::consts::PI
        }

        fn euclidean_distance(&self, other: &Self) -> f64 {
            self.vec
                .par_iter()
                .zip(other.vec.par_iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
        }

        fn manhattan_distance(&self, other: &Self) -> f64 {
            self.vec
                .par_iter()
                .zip(other.vec.par_iter())
                .map(|(x, y)| (x - y).abs())
                .sum()
        }

        fn chebyshev_distance(&self, other: &Self) -> f64 {
            self.vec
                .iter()
                .zip(other.vec.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::VectorDistance;
    use crate::embeddings::Embedding;

    fn embeddings() -> (Embedding, Embedding) {
        let embedding_1 = Embedding {
            document: "test".to_string(),
            vec: vec![1.0, 2.0, 3.0],
        };

        let embedding_2 = Embedding {
            document: "test".to_string(),
            vec: vec![1.0, 5.0, 7.0],
        };

        (embedding_1, embedding_2)
    }

    #[test]
    fn test_dot_product() {
        let (embedding_1, embedding_2) = embeddings();

        assert_eq!(embedding_1.dot_product(&embedding_2), 32.0)
    }

    #[test]
    fn test_cosine_similarity() {
        let (embedding_1, embedding_2) = embeddings();

        assert_eq!(
            embedding_1.cosine_similarity(&embedding_2, false),
            0.9875414397573881
        )
    }

    #[test]
    fn test_angular_distance() {
        let (embedding_1, embedding_2) = embeddings();

        assert_eq!(
            embedding_1.angular_distance(&embedding_2, false),
            0.0502980301830343
        )
    }

    #[test]
    fn test_euclidean_distance() {
        let (embedding_1, embedding_2) = embeddings();

        assert_eq!(embedding_1.euclidean_distance(&embedding_2), 5.0)
    }

    #[test]
    fn test_manhattan_distance() {
        let (embedding_1, embedding_2) = embeddings();

        assert_eq!(embedding_1.manhattan_distance(&embedding_2), 7.0)
    }

    #[test]
    fn test_chebyshev_distance() {
        let (embedding_1, embedding_2) = embeddings();

        assert_eq!(embedding_1.chebyshev_distance(&embedding_2), 4.0)
    }
}
