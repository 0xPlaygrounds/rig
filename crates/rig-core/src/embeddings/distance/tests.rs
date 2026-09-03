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

    assert_eq!(embedding_1.dot_product(&embedding_2), 32.0);
}

#[test]
fn test_cosine_similarity() {
    let (embedding_1, embedding_2) = embeddings();

    assert_eq!(
        embedding_1.cosine_similarity(&embedding_2, false),
        0.9875414397573881
    );
}

#[test]
fn test_angular_distance() {
    let (embedding_1, embedding_2) = embeddings();

    assert_eq!(
        embedding_1.angular_distance(&embedding_2, false),
        0.0502980301830343
    );
}

#[test]
fn test_euclidean_distance() {
    let (embedding_1, embedding_2) = embeddings();

    assert_eq!(embedding_1.euclidean_distance(&embedding_2), 5.0);
}

#[test]
fn test_manhattan_distance() {
    let (embedding_1, embedding_2) = embeddings();

    assert_eq!(embedding_1.manhattan_distance(&embedding_2), 7.0);
}

#[test]
fn test_chebyshev_distance() {
    let (embedding_1, embedding_2) = embeddings();

    assert_eq!(embedding_1.chebyshev_distance(&embedding_2), 4.0);
}

/// Every sum is taken in fixed chunks in index order, so a metric is the
/// same bits on every run — under `rayon`, whatever order the threads
/// finish in. (A recorded retrieval score that differed in its last digit
/// between two replays of one cassette is how this was found.)
#[test]
fn a_metric_is_the_same_bits_on_every_run() {
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
        a.dot_product(&b),
        a.cosine_similarity(&b, false),
        a.euclidean_distance(&b),
        a.manhattan_distance(&b),
    );
    for _ in 0..64 {
        let again = (
            a.dot_product(&b),
            a.cosine_similarity(&b, false),
            a.euclidean_distance(&b),
            a.manhattan_distance(&b),
        );
        assert_eq!(first.0.to_bits(), again.0.to_bits());
        assert_eq!(first.1.to_bits(), again.1.to_bits());
        assert_eq!(first.2.to_bits(), again.2.to_bits());
        assert_eq!(first.3.to_bits(), again.3.to_bits());
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
    assert_eq!(first.0.to_bits(), reference.to_bits());
}
