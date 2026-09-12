use std::cmp::Reverse;

use crate::{embeddings::embedding::Embedding, vector_store::IndexStrategy};

use super::{InMemoryVectorStore, RankingItem};

#[test]
fn equal_score_cutoff_is_independent_of_candidate_order() {
    let query = Embedding {
        document: "query".into(),
        vec: vec![1.0, 1.0],
    };
    let store =
        InMemoryVectorStore::from_documents_with_ids(["a", "b", "c"].into_iter().map(|id| {
            (
                id,
                id.to_owned(),
                vec![Embedding {
                    document: id.into(),
                    vec: query.vec.clone(),
                }],
            )
        }));
    for order in [
        ["a", "b", "c"],
        ["c", "b", "a"],
        ["b", "a", "c"],
        ["a", "c", "b"],
    ] {
        for n in 0..=3 {
            let ranking = store
                .rank_candidates(order, &query, n, None, None)
                .expect("ranked");
            let selected: Vec<_> = super::ranked(ranking)
                .into_iter()
                .map(|item| item.1.as_str())
                .collect();
            assert_eq!(
                selected,
                ["a", "b", "c"][..n],
                "cutoff {n}, input {order:?}"
            );
        }
    }
}

#[test]
fn test_auto_ids() {
    let mut vector_store = InMemoryVectorStore::builder()
        .index_strategy(IndexStrategy::LSH {
            num_tables: 5,
            num_hyperplanes: 10,
        })
        .documents(vec![
            (
                "glarb-garb",
                vec![Embedding {
                    document: "glarb-garb".to_string(),
                    vec: vec![0.1, 0.1, 0.5],
                }],
            ),
            (
                "marble-marble",
                vec![Embedding {
                    document: "marble-marble".to_string(),
                    vec: vec![0.7, -0.3, 0.0],
                }],
            ),
            (
                "flumb-flumb",
                vec![Embedding {
                    document: "flumb-flumb".to_string(),
                    vec: vec![0.3, 0.7, 0.1],
                }],
            ),
        ])
        .build();

    vector_store.add_documents(vec![
        (
            "brotato",
            vec![Embedding {
                document: "brotato".to_string(),
                vec: vec![0.3, 0.7, 0.1],
            }],
        ),
        (
            "ping-pong",
            vec![Embedding {
                document: "ping-pong".to_string(),
                vec: vec![0.7, -0.3, 0.0],
            }],
        ),
    ]);

    let mut store = vector_store.embeddings.into_iter().collect::<Vec<_>>();
    store.sort_by_key(|(id, _)| id.clone());

    assert_eq!(
        store,
        vec![
            (
                "doc0".to_string(),
                (
                    "glarb-garb",
                    vec![Embedding {
                        document: "glarb-garb".to_string(),
                        vec: vec![0.1, 0.1, 0.5],
                    }]
                )
            ),
            (
                "doc1".to_string(),
                (
                    "marble-marble",
                    vec![Embedding {
                        document: "marble-marble".to_string(),
                        vec: vec![0.7, -0.3, 0.0],
                    }]
                )
            ),
            (
                "doc2".to_string(),
                (
                    "flumb-flumb",
                    vec![Embedding {
                        document: "flumb-flumb".to_string(),
                        vec: vec![0.3, 0.7, 0.1],
                    }]
                )
            ),
            (
                "doc3".to_string(),
                (
                    "brotato",
                    vec![Embedding {
                        document: "brotato".to_string(),
                        vec: vec![0.3, 0.7, 0.1],
                    }]
                )
            ),
            (
                "doc4".to_string(),
                (
                    "ping-pong",
                    vec![Embedding {
                        document: "ping-pong".to_string(),
                        vec: vec![0.7, -0.3, 0.0],
                    }]
                )
            )
        ]
    );
}

#[test]
fn test_single_embedding() {
    let vector_store = InMemoryVectorStore::builder()
        .index_strategy(IndexStrategy::LSH {
            num_tables: 5,
            num_hyperplanes: 10,
        })
        .documents_with_ids(vec![
            (
                "doc1",
                "glarb-garb",
                vec![Embedding {
                    document: "glarb-garb".to_string(),
                    vec: vec![0.1, 0.1, 0.5],
                }],
            ),
            (
                "doc2",
                "marble-marble",
                vec![Embedding {
                    document: "marble-marble".to_string(),
                    vec: vec![0.7, -0.3, 0.0],
                }],
            ),
            (
                "doc3",
                "flumb-flumb",
                vec![Embedding {
                    document: "flumb-flumb".to_string(),
                    vec: vec![0.3, 0.7, 0.1],
                }],
            ),
        ])
        .build();

    let ranking = vector_store
        .vector_search(
            &Embedding {
                document: "glarby-glarble".to_string(),
                vec: vec![0.0, 0.1, 0.6],
            },
            1,
            None,
            None,
        )
        .unwrap();

    assert_eq!(
        ranking
            .into_iter()
            .map(|Reverse(RankingItem(distance, id, doc, _))| {
                (
                    distance.0,
                    id.clone(),
                    serde_json::from_str(&serde_json::to_string(doc).unwrap()).unwrap(),
                )
            })
            .collect::<Vec<(_, _, String)>>(),
        vec![(
            0.9807965956109156,
            "doc1".to_string(),
            "glarb-garb".to_string()
        )]
    );
}

#[test]
fn test_multiple_embeddings() {
    let vector_store = InMemoryVectorStore::builder()
        .index_strategy(IndexStrategy::LSH {
            num_tables: 5,
            num_hyperplanes: 10,
        })
        .documents_with_ids(vec![
            (
                "doc1",
                "glarb-garb",
                vec![
                    Embedding {
                        document: "glarb-garb".to_string(),
                        vec: vec![0.1, 0.1, 0.5],
                    },
                    Embedding {
                        document: "don't-choose-me".to_string(),
                        vec: vec![-0.5, 0.9, 0.1],
                    },
                ],
            ),
            (
                "doc2",
                "marble-marble",
                vec![
                    Embedding {
                        document: "marble-marble".to_string(),
                        vec: vec![0.7, -0.3, 0.0],
                    },
                    Embedding {
                        document: "sandwich".to_string(),
                        vec: vec![0.5, 0.5, -0.7],
                    },
                ],
            ),
            (
                "doc3",
                "flumb-flumb",
                vec![
                    Embedding {
                        document: "flumb-flumb".to_string(),
                        vec: vec![0.3, 0.7, 0.1],
                    },
                    Embedding {
                        document: "banana".to_string(),
                        vec: vec![0.1, -0.5, -0.5],
                    },
                ],
            ),
        ])
        .build();

    let ranking = vector_store
        .vector_search(
            &Embedding {
                document: "glarby-glarble".to_string(),
                vec: vec![0.0, 0.1, 0.6],
            },
            1,
            None,
            None,
        )
        .unwrap();

    assert_eq!(
        ranking
            .into_iter()
            .map(|Reverse(RankingItem(distance, id, doc, _))| {
                (
                    distance.0,
                    id.clone(),
                    serde_json::from_str(&serde_json::to_string(doc).unwrap()).unwrap(),
                )
            })
            .collect::<Vec<(_, _, String)>>(),
        vec![(
            0.9807965956109156,
            "doc1".to_string(),
            "glarb-garb".to_string()
        )]
    );
}

#[tokio::test]
async fn top_n_honors_filter_and_threshold() {
    use crate::test_utils::MockEmbeddingModel;
    use crate::vector_store::VectorStoreIndex;
    use crate::vector_store::request::{Filter, SearchFilter, VectorSearchRequest};
    use serde::Serialize;
    use serde_json::json;

    // Document payloads carry metadata alongside content, like real backends.
    #[derive(Clone, Serialize, PartialEq, Eq)]
    struct Item {
        category: String,
        text: String,
    }

    fn item(category: &str, text: &str) -> Item {
        Item {
            category: category.to_string(),
            text: text.to_string(),
        }
    }

    // `MockEmbeddingModel` embeds every query as this fixed 10-dim vector; give
    // every document the same embedding so all cosine similarities are 1.0 and
    // only the filter/threshold decide the result set.
    let vec = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let embedding = |doc: &str| {
        vec![Embedding {
            document: doc.to_string(),
            vec: vec.clone(),
        }]
    };

    let index = InMemoryVectorStore::from_documents_with_ids(vec![
        ("a", item("fruit", "banana"), embedding("banana")),
        ("b", item("veg", "carrot"), embedding("carrot")),
        ("c", item("fruit", "apple"), embedding("apple")),
    ])
    .index(MockEmbeddingModel);

    let ids = |req| async {
        let mut out: Vec<String> = index
            .top_n_ids(req)
            .await
            .unwrap()
            .into_iter()
            .map(|(_, id)| id)
            .collect();
        out.sort();
        out
    };

    // No filter: every document is returned.
    let all = ids(VectorSearchRequest::builder()
        .query("q")
        .samples(10)
        .build())
    .await;
    assert_eq!(all, vec!["a", "b", "c"]);

    // Metadata filter: only documents whose `category` field is `fruit`.
    let fruit = ids(VectorSearchRequest::builder()
        .query("q")
        .samples(10)
        .filter(Filter::eq("category", json!("fruit")))
        .build())
    .await;
    assert_eq!(fruit, vec!["a", "c"]);

    // Threshold above the maximum similarity (1.0): nothing qualifies.
    let none = ids(VectorSearchRequest::builder()
        .query("q")
        .samples(10)
        .threshold(2.0)
        .build())
    .await;
    assert!(none.is_empty());

    // Threshold at or below the similarity keeps all matches.
    let kept = ids(VectorSearchRequest::builder()
        .query("q")
        .samples(10)
        .threshold(0.5)
        .build())
    .await;
    assert_eq!(kept, vec!["a", "b", "c"]);
}

#[tokio::test]
async fn top_n_excludes_non_finite_similarity() {
    use crate::test_utils::MockEmbeddingModel;
    use crate::vector_store::VectorStoreIndex;
    use crate::vector_store::request::VectorSearchRequest;

    let embedding = |doc: &str, vec: Vec<f64>| {
        vec![Embedding {
            document: doc.to_string(),
            vec,
        }]
    };

    // The zero-magnitude embedding produces a NaN cosine similarity, which
    // sorts as the maximum under OrderedFloat. It must not rank first (or
    // appear at all), even with no threshold set.
    let index = InMemoryVectorStore::from_documents_with_ids(vec![
        (
            "good",
            "good".to_string(),
            embedding(
                "good",
                vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
        ),
        (
            "degenerate",
            "degenerate".to_string(),
            embedding("degenerate", vec![0.0; 10]),
        ),
    ])
    .index(MockEmbeddingModel);

    let ids: Vec<String> = index
        .top_n_ids(
            VectorSearchRequest::builder()
                .query("q")
                .samples(10)
                .build(),
        )
        .await
        .unwrap()
        .into_iter()
        .map(|(_, id)| id)
        .collect();
    assert_eq!(ids, vec!["good".to_string()]);
}

#[tokio::test]
async fn top_n_ranks_document_by_best_finite_embedding() {
    use crate::test_utils::MockEmbeddingModel;
    use crate::vector_store::VectorStoreIndex;
    use crate::vector_store::request::VectorSearchRequest;

    // A document that owns both a strong finite embedding and a degenerate
    // zero-magnitude (NaN) one must still be returned, ranked by the finite
    // embedding — not dropped because NaN sorts as the OrderedFloat maximum.
    let index = InMemoryVectorStore::from_documents_with_ids(vec![(
        "mixed",
        "mixed".to_string(),
        vec![
            Embedding {
                document: "good-chunk".to_string(),
                vec: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            },
            Embedding {
                document: "empty-chunk".to_string(),
                vec: vec![0.0; 10],
            },
        ],
    )])
    .index(MockEmbeddingModel);

    let results = index
        .top_n_ids(
            VectorSearchRequest::builder()
                .query("q")
                .samples(10)
                .build(),
        )
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, "mixed");
    assert!(results[0].0.is_finite());
}

/// A search returns its documents best first, and documents of one score in
/// id order, whatever order the store iterated them in: the ranking heap
/// is sorted before it is read, so a request built from the results is the
/// same request on every run.
#[tokio::test]
async fn top_n_returns_documents_best_first_then_by_id() {
    use crate::test_utils::MockEmbeddingModel;
    use crate::vector_store::VectorStoreIndex;
    use crate::vector_store::request::VectorSearchRequest;

    // The mock embeds every query as this fixed vector: a document with the
    // same vector scores 1.0, the reversed one scores less.
    let query = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let far = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
    let embedding = |doc: &str, vec: &[f64]| {
        vec![Embedding {
            document: doc.to_string(),
            vec: vec.to_vec(),
        }]
    };
    for _ in 0..8 {
        let index = InMemoryVectorStore::from_documents_with_ids(vec![
            ("c", "c".to_string(), embedding("c", &query)),
            ("a", "a".to_string(), embedding("a", &query)),
            ("z", "z".to_string(), embedding("z", &far)),
            ("b", "b".to_string(), embedding("b", &query)),
        ])
        .index(MockEmbeddingModel);
        let request = VectorSearchRequest::builder()
            .query("q")
            .samples(10)
            .build();
        let ids: Vec<String> = index
            .top_n_ids(request)
            .await
            .expect("a search")
            .into_iter()
            .map(|(_, id)| id)
            .collect();
        assert_eq!(ids, ["a", "b", "c", "z"], "best first, ties by id");
    }
}
