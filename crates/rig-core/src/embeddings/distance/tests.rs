use super::VectorDistance;
use crate::embeddings::Embedding;

/// Every sum is taken in fixed chunks in index order, so a score is the
/// same bits on every run. (A recorded retrieval score that differed in its last digit
/// between two replays of one cassette is how this was found.)
#[test]
fn a_score_is_the_same_bits_on_every_run() {
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
        a.cosine_similarity(&b, true),
        a.cosine_similarity(&b, false),
    );
    for _ in 0..64 {
        let again = (
            a.cosine_similarity(&b, true),
            a.cosine_similarity(&b, false),
        );
        assert_eq!(first.0.to_bits(), again.0.to_bits());
        assert_eq!(first.1.to_bits(), again.1.to_bits());
    }
    // And the same bits as the chunked reference: 256-term chunks summed
    // left to right, then the chunk sums.
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
