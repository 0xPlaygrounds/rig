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
