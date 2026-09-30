use super::{VectorDistance, VectorDistanceError};
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

    assert_eq!(embedding_1.dot_product(&embedding_2), Ok(32.0));
}

#[test]
fn test_cosine_similarity() {
    let (embedding_1, embedding_2) = embeddings();

    assert_eq!(
        embedding_1.cosine_similarity(&embedding_2, false),
        Ok(0.9875414397573881)
    );
}

#[test]
fn test_angular_distance() {
    let (embedding_1, embedding_2) = embeddings();

    assert_eq!(
        embedding_1.angular_distance(&embedding_2, false),
        Ok(0.0502980301830343)
    );
}

#[test]
fn angular_distance_handles_rounded_cosine_endpoints() {
    let vector = Embedding {
        document: "same".into(),
        vec: vec![1.0, 1.0, 1.0],
    };
    let opposite = Embedding {
        document: "opposite".into(),
        vec: vec![-1.0, -1.0, -1.0],
    };

    assert_eq!(vector.angular_distance(&vector, false), Ok(0.0));
    assert_eq!(vector.angular_distance(&opposite, false), Ok(1.0));
}

#[test]
fn test_euclidean_distance() {
    let (embedding_1, embedding_2) = embeddings();

    assert_eq!(embedding_1.euclidean_distance(&embedding_2), Ok(5.0));
}

#[test]
fn test_manhattan_distance() {
    let (embedding_1, embedding_2) = embeddings();

    assert_eq!(embedding_1.manhattan_distance(&embedding_2), Ok(7.0));
}

#[test]
fn test_chebyshev_distance() {
    let (embedding_1, embedding_2) = embeddings();

    assert_eq!(embedding_1.chebyshev_distance(&embedding_2), Ok(4.0));
}

/// Fixed chunks and index-order sums keep the same bits on every run.
#[test]
fn a_metric_is_the_same_bits_on_every_run() -> anyhow::Result<()> {
    let a = Embedding {
        document: "a".into(),
        vec: (0..5000)
            .map(|i| ((i * 7919) % 1000) as f64 / 997.0 - 0.5)
            .collect(),
    };
    let b = Embedding {
        document: "b".into(),
        vec: (0..5000)
            .map(|i| ((i * 104729) % 1000) as f64 / 991.0 - 0.5)
            .collect(),
    };
    let first = (
        a.dot_product(&b)?,
        a.cosine_similarity(&b, false)?,
        a.euclidean_distance(&b)?,
        a.manhattan_distance(&b)?,
    );
    for _ in 0..64 {
        let again = (
            a.dot_product(&b)?,
            a.cosine_similarity(&b, false)?,
            a.euclidean_distance(&b)?,
            a.manhattan_distance(&b)?,
        );
        anyhow::ensure!(first.0.to_bits() == again.0.to_bits());
        anyhow::ensure!(first.1.to_bits() == again.1.to_bits());
        anyhow::ensure!(first.2.to_bits() == again.2.to_bits());
        anyhow::ensure!(first.3.to_bits() == again.3.to_bits());
    }
    // And the same bits as the chunked reference the sequential build
    // computes: 256-term chunks summed left to right, then the chunk sums.
    let reference: f64 = a
        .vec
        .chunks(256)
        .zip(b.vec.chunks(256))
        .map(|(x, y)| x.iter().zip(y).map(|(x, y)| x * y).sum::<f64>())
        .collect::<Vec<f64>>()
        .into_iter()
        .sum();
    anyhow::ensure!(first.0.to_bits() == reference.to_bits());
    Ok(())
}

#[test]
fn every_metric_rejects_mismatched_dimensions() {
    for (left, right) in [(2, 3), (3, 2), (0, 1), (1, 0), (256, 257), (257, 256)] {
        let a = Embedding {
            document: String::new(),
            vec: vec![1.0; left],
        };
        let b = Embedding {
            document: String::new(),
            vec: vec![1.0; right],
        };
        let expected = Err(VectorDistanceError::DimensionMismatch { left, right });
        for result in [
            a.dot_product(&b),
            a.cosine_similarity(&b, false),
            a.cosine_similarity(&b, true),
            a.angular_distance(&b, false),
            a.angular_distance(&b, true),
            a.euclidean_distance(&b),
            a.manhattan_distance(&b),
            a.chebyshev_distance(&b),
        ] {
            assert_eq!(result, expected);
        }
    }
}

#[test]
fn normalized_metrics_accept_equal_dimensions() {
    let a = Embedding {
        document: String::new(),
        vec: vec![1.0, 0.0],
    };
    let b = Embedding {
        document: String::new(),
        vec: vec![0.0, 1.0],
    };
    assert_eq!(a.cosine_similarity(&b, true), Ok(0.0));
    assert_eq!(a.angular_distance(&b, true), Ok(0.5));
}

#[test]
fn equal_empty_and_zero_vectors_keep_arithmetic_behavior() -> anyhow::Result<()> {
    for vec in [vec![], vec![0.0, 0.0]] {
        let embedding = Embedding {
            document: String::new(),
            vec,
        };
        anyhow::ensure!(embedding.dot_product(&embedding)? == 0.0);
        anyhow::ensure!(embedding.euclidean_distance(&embedding)? == 0.0);
        anyhow::ensure!(embedding.manhattan_distance(&embedding)? == 0.0);
        anyhow::ensure!(embedding.chebyshev_distance(&embedding)? == 0.0);
        anyhow::ensure!(embedding.cosine_similarity(&embedding, false)?.is_nan());
        anyhow::ensure!(embedding.angular_distance(&embedding, false)?.is_nan());
    }
    Ok(())
}
