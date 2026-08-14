//! In-memory implementation of a vector store.
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use super::{IndexStrategy, VectorStoreError, VectorStoreIndex, request::VectorSearchRequest};
use crate::{
    embeddings::{Embedding, EmbeddingModel, distance::VectorDistance},
    vector_store::request::Filter,
};

use super::lsh::LSHIndex;

pub use super::builder::InMemoryVectorStoreBuilder;

/// [InMemoryVectorStore] is a simple in-memory vector store that stores embeddings
/// in-memory using a HashMap.
#[derive(Clone, Default)]
pub struct InMemoryVectorStore<D: Serialize> {
    /// The embeddings are stored in a HashMap.
    /// Hashmap key is the document id.
    /// Hashmap value is a tuple of the serializable document and its corresponding embeddings.
    embeddings: HashMap<String, (D, Vec<Embedding>)>,

    index_strategy: IndexStrategy,

    lsh_index: Option<LSHIndex>,
}

impl<D: Serialize + Eq> InMemoryVectorStore<D> {
    /// Create a new builder for configuring an [InMemoryVectorStore].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rig_core::vector_store::InMemoryVectorStore;
    ///
    /// let store = InMemoryVectorStore::<String>::builder()
    ///     .with_lsh()
    ///     .documents(documents)
    ///     .build();
    /// ```
    pub fn builder() -> InMemoryVectorStoreBuilder<D> {
        InMemoryVectorStoreBuilder::new()
    }

    /// Internal constructor used by the builder.
    pub(super) fn from_builder(
        embeddings: HashMap<String, (D, Vec<Embedding>)>,
        index_strategy: IndexStrategy,
    ) -> Self {
        let mut vector_store = Self {
            embeddings,
            index_strategy: index_strategy.clone(),
            lsh_index: None,
        };

        // Initialize LSH index if needed
        if let IndexStrategy::LSH {
            num_tables,
            num_hyperplanes,
        } = index_strategy
        {
            vector_store.initialize_lsh_index(num_tables, num_hyperplanes);
        }

        vector_store
    }

    /// Create a new [InMemoryVectorStore] from documents and their corresponding embeddings.
    /// Ids are automatically generated have will have the form `"doc{n}"` where `n`
    /// is the index of the document.
    ///
    /// Uses BruteForce index strategy by default. For custom index strategies, use [InMemoryVectorStore::builder].
    pub fn from_documents(documents: impl IntoIterator<Item = (D, Vec<Embedding>)>) -> Self {
        let mut store = Self::from_builder(HashMap::new(), IndexStrategy::default());
        store.add_documents(documents);
        store
    }

    /// Create a new [InMemoryVectorStore] from documents and their corresponding embeddings with ids.
    ///
    /// Uses BruteForce index strategy by default. For custom index strategies, use [InMemoryVectorStore::builder].
    pub fn from_documents_with_ids(
        documents: impl IntoIterator<Item = (impl ToString, D, Vec<Embedding>)>,
    ) -> Self {
        let mut store = Self::from_builder(HashMap::new(), IndexStrategy::default());
        store.add_documents_with_ids(documents);
        store
    }

    /// Create a new [InMemoryVectorStore] from documents and their corresponding embeddings.
    /// Document ids are generated using the provided function.
    ///
    /// Uses BruteForce index strategy by default. For custom index strategies, use [InMemoryVectorStore::builder].
    pub fn from_documents_with_id_f(
        documents: impl IntoIterator<Item = (D, Vec<Embedding>)>,
        f: fn(&D) -> String,
    ) -> Self {
        let mut store = Self::from_builder(HashMap::new(), IndexStrategy::default());
        store.add_documents_with_id_f(documents, f);
        store
    }

    /// Insert a single document, keeping the LSH index (when enabled) in sync.
    fn insert_document(&mut self, id: String, doc: D, embeddings: Vec<Embedding>) {
        if let Some(ref mut lsh_index) = self.lsh_index {
            for embedding in embeddings.iter() {
                lsh_index.insert(id.clone(), &embedding.vec);
            }
        }
        self.embeddings.insert(id, (doc, embeddings));
    }

    /// Tests whether a document satisfies the (optional) metadata filter.
    ///
    /// Documents are serialized to JSON on demand and matched with
    /// [`Filter::satisfies`]. Returns `Ok(true)` when no filter is set so the
    /// serialization cost is only paid for filtered queries.
    fn satisfies_filter(
        doc: &D,
        filter: Option<&Filter<serde_json::Value>>,
    ) -> Result<bool, VectorStoreError> {
        match filter {
            None => Ok(true),
            Some(filter) => {
                let value = serde_json::to_value(doc).map_err(VectorStoreError::JsonError)?;
                Ok(filter.satisfies(&value))
            }
        }
    }

    /// Scores one candidate document against the query prompt.
    ///
    /// Returns the best similarity across the document's embeddings together with
    /// the matching embedding text, or `None` when the document is filtered out,
    /// has no finite-similarity embedding, or scores below the threshold. Shared
    /// by the brute-force and LSH scans so the filter, threshold, and NaN
    /// handling live in exactly one place.
    fn score_candidate<'a>(
        doc: &D,
        embeddings: &'a [Embedding],
        prompt_embedding: &Embedding,
        filter: Option<&Filter<serde_json::Value>>,
        threshold: Option<f64>,
    ) -> Result<Option<(OrderedFloat<f64>, &'a String)>, VectorStoreError> {
        if !Self::satisfies_filter(doc, filter)? {
            return Ok(None);
        }

        // Best (highest-similarity) embedding for this document.
        //
        // A zero-magnitude embedding yields a NaN similarity, which sorts as the
        // maximum under `OrderedFloat` and slips past `distance < threshold`
        // (every comparison with NaN is false). Drop non-finite similarities
        // *before* selecting the max so a document still ranks by its best
        // finite embedding; the document is skipped only when it has no finite
        // similarity at all.
        let Some((distance, embed_doc)) = embeddings
            .iter()
            .map(|embedding| {
                (
                    OrderedFloat(embedding.cosine_similarity(prompt_embedding, false)),
                    &embedding.document,
                )
            })
            .filter(|(distance, _)| distance.0.is_finite())
            .max_by(|a, b| a.0.cmp(&b.0))
        else {
            return Ok(None);
        };

        // Skip documents below the similarity threshold.
        if threshold.is_some_and(|t| distance.0 < t) {
            return Ok(None);
        }

        Ok(Some((distance, embed_doc)))
    }

    /// Implement vector search on [InMemoryVectorStore].
    /// To be used by implementations of [VectorStoreIndex::top_n] and [VectorStoreIndex::top_n_ids] methods.
    ///
    /// The metadata `filter` and similarity `threshold` are applied *during* the
    /// scan, before the top-`n` selection, so results match backends that filter
    /// server-side rather than returning the unfiltered top-`n`.
    fn vector_search(
        &self,
        prompt_embedding: &Embedding,
        n: usize,
        filter: Option<&Filter<serde_json::Value>>,
        threshold: Option<f64>,
    ) -> Result<EmbeddingRanking<'_, D>, VectorStoreError> {
        match &self.index_strategy {
            IndexStrategy::BruteForce => self.rank_candidates(
                self.embeddings.keys(),
                prompt_embedding,
                n,
                filter,
                threshold,
            ),
            IndexStrategy::LSH { .. } => {
                // If we don't have an LSH index yet, fall back to brute force
                let Some(lsh_index) = self.lsh_index.as_ref() else {
                    tracing::warn!("LSH index not initialized, falling back to brute force search");
                    return self.rank_candidates(
                        self.embeddings.keys(),
                        prompt_embedding,
                        n,
                        filter,
                        threshold,
                    );
                };
                self.rank_candidates(
                    lsh_index.query(&prompt_embedding.vec),
                    prompt_embedding,
                    n,
                    filter,
                    threshold,
                )
            }
        }
    }

    /// Ranks candidate documents by best embedding similarity, keeping the top `n`.
    ///
    /// Shared by the brute-force scan (which passes every stored id) and the LSH
    /// scan (which passes only its candidate ids); unknown ids are skipped.
    fn rank_candidates(
        &self,
        candidate_ids: impl IntoIterator<Item = impl AsRef<str>>,
        prompt_embedding: &Embedding,
        n: usize,
        filter: Option<&Filter<serde_json::Value>>,
        threshold: Option<f64>,
    ) -> Result<EmbeddingRanking<'_, D>, VectorStoreError> {
        // Sort documents by best embedding distance
        let mut docs = BinaryHeap::new();

        for candidate_id in candidate_ids {
            let Some((id, (doc, embeddings))) =
                self.embeddings.get_key_value(candidate_id.as_ref())
            else {
                continue;
            };
            let Some((distance, embed_doc)) =
                Self::score_candidate(doc, embeddings, prompt_embedding, filter, threshold)?
            else {
                continue;
            };

            docs.push(Reverse(RankingItem(distance, id, doc, embed_doc)));

            // If the heap size exceeds n, pop the least old element.
            if docs.len() > n {
                docs.pop();
            }
        }

        // Log selected documents with their distances (the joined string is only
        // built when INFO logging is actually enabled for the "rig" target).
        if tracing::enabled!(target: "rig", tracing::Level::INFO) {
            tracing::info!(target: "rig",
                "Selected documents: {}",
                docs.iter()
                    .map(|Reverse(RankingItem(distance, id, _, _))| format!("{id} ({distance})"))
                    .collect::<Vec<String>>()
                    .join(", ")
            );
        }

        Ok(docs)
    }

    /// Initialize LSH index from existing embeddings
    fn initialize_lsh_index(&mut self, num_tables: usize, num_hyperplanes: usize) {
        if self.embeddings.is_empty() {
            return;
        }

        // Get the dimension from the first embedding
        let first_embedding = self
            .embeddings
            .values()
            .next()
            .and_then(|(_, embeddings)| embeddings.iter().next())
            .map(|e| e.vec.len())
            .unwrap_or(0);

        if first_embedding == 0 {
            return;
        }

        let mut lsh_index = LSHIndex::new(first_embedding, num_tables, num_hyperplanes);

        // Insert all existing embeddings into the LSH index
        for (id, (_, embeddings)) in self.embeddings.iter() {
            for embedding in embeddings.iter() {
                lsh_index.insert(id.clone(), &embedding.vec);
            }
        }

        self.lsh_index = Some(lsh_index);
    }

    /// Add documents and their corresponding embeddings to the store.
    /// Ids are automatically generated have will have the form `"doc{n}"` where `n`
    /// is the index of the document.
    pub fn add_documents(&mut self, documents: impl IntoIterator<Item = (D, Vec<Embedding>)>) {
        let current_index = self.embeddings.len();
        for (index, (doc, embeddings)) in documents.into_iter().enumerate() {
            self.insert_document(format!("doc{}", index + current_index), doc, embeddings);
        }
    }

    /// Add documents and their corresponding embeddings to the store with ids.
    pub fn add_documents_with_ids(
        &mut self,
        documents: impl IntoIterator<Item = (impl ToString, D, Vec<Embedding>)>,
    ) {
        for (id, doc, embeddings) in documents {
            self.insert_document(id.to_string(), doc, embeddings);
        }
    }

    /// Add documents and their corresponding embeddings to the store.
    /// Document ids are generated using the provided function.
    pub fn add_documents_with_id_f(
        &mut self,
        documents: impl IntoIterator<Item = (D, Vec<Embedding>)>,
        f: fn(&D) -> String,
    ) {
        for (doc, embeddings) in documents {
            self.insert_document(f(&doc), doc, embeddings);
        }
    }
}

/// RankingItem(distance, document_id, serializable document, embeddings document)
#[derive(Eq, PartialEq)]
struct RankingItem<'a, D: Serialize>(OrderedFloat<f64>, &'a String, &'a D, &'a String);

impl<D: Serialize + Eq> Ord for RankingItem<'_, D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<D: Serialize + Eq> PartialOrd for RankingItem<'_, D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

type EmbeddingRanking<'a, D> = BinaryHeap<Reverse<RankingItem<'a, D>>>;

impl<D: Serialize> InMemoryVectorStore<D> {
    pub fn index<M: EmbeddingModel>(self, model: M) -> InMemoryVectorIndex<M, D> {
        InMemoryVectorIndex::new(model, self)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &(D, Vec<Embedding>))> {
        self.embeddings.iter()
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

pub struct InMemoryVectorIndex<M: EmbeddingModel, D: Serialize> {
    model: M,
    pub store: InMemoryVectorStore<D>,
}

impl<M: EmbeddingModel, D: Serialize> InMemoryVectorIndex<M, D> {
    pub fn new(model: M, store: InMemoryVectorStore<D>) -> Self {
        Self { model, store }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &(D, Vec<Embedding>))> {
        self.store.iter()
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}

impl<M: EmbeddingModel + Sync, D: Serialize + Sync + Send + Eq> VectorStoreIndex
    for InMemoryVectorIndex<M, D>
{
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: for<'a> Deserialize<'a>>(
        &self,
        req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let prompt_embedding = &self.model.embed_text(req.query()).await?;

        let docs = self.store.vector_search(
            prompt_embedding,
            req.samples() as usize,
            req.filter().as_ref(),
            req.threshold(),
        )?;

        // Return n best
        docs.into_iter()
            // The distance should always be between 0 and 1, so distance should be fine to use as an absolute value
            .map(|Reverse(RankingItem(distance, id, doc, _))| {
                Ok((
                    distance.0,
                    id.clone(),
                    serde_json::from_str(
                        &serde_json::to_string(doc).map_err(VectorStoreError::JsonError)?,
                    )
                    .map_err(VectorStoreError::JsonError)?,
                ))
            })
            .collect::<Result<Vec<_>, _>>()
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let prompt_embedding = &self.model.embed_text(req.query()).await?;

        let docs = self.store.vector_search(
            prompt_embedding,
            req.samples() as usize,
            req.filter().as_ref(),
            req.threshold(),
        )?;

        docs.into_iter()
            .map(|Reverse(RankingItem(distance, id, _, _))| Ok((distance.0, id.clone())))
            .collect::<Result<Vec<_>, _>>()
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Reverse;

    use crate::{embeddings::embedding::Embedding, vector_store::IndexStrategy};

    use super::{InMemoryVectorStore, RankingItem};

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
        )
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
        )
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
}
