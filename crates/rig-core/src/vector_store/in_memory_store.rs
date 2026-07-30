//! In-memory implementation of a vector store.
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    sync::{PoisonError, RwLock},
};

use ordered_float::OrderedFloat;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use super::{
    IndexStrategy, SearchHit, StoreRecord, VectorStoreError, request::VectorSearchRequest,
};
use crate::{
    OneOrMany,
    embeddings::{Embedding, distance::VectorDistance},
    vector_store::request::Filter,
};

use super::lsh::LSHIndex;

pub use super::builder::InMemoryVectorStoreBuilder;

/// Internal, single-threaded state of the store.
#[derive(Clone, Default)]
struct Inner {
    /// The embeddings are stored in a HashMap.
    /// Hashmap key is the document id.
    /// Hashmap value is a tuple of the JSON document payload and its corresponding embeddings.
    embeddings: HashMap<String, (Value, OneOrMany<Embedding>)>,

    index_strategy: IndexStrategy,

    lsh_index: Option<LSHIndex>,
}

/// [InMemoryVectorStore] is a simple in-memory vector store that stores JSON
/// document payloads and their embeddings in a HashMap.
///
/// Queries arrive pre-embedded via [`VectorSearchRequest`]; the store never
/// embeds text itself.
#[derive(Default)]
pub struct InMemoryVectorStore {
    inner: RwLock<Inner>,
}

impl Clone for InMemoryVectorStore {
    fn clone(&self) -> Self {
        Self {
            inner: RwLock::new(self.read().clone()),
        }
    }
}

/// RankingItem(distance, document_id, document payload)
#[derive(PartialEq)]
struct RankingItem<'a>(OrderedFloat<f64>, &'a String, &'a Value);

impl Eq for RankingItem<'_> {}

impl Ord for RankingItem<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for RankingItem<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

type EmbeddingRanking<'a> = BinaryHeap<Reverse<RankingItem<'a>>>;

impl InMemoryVectorStore {
    /// Create a new, empty [InMemoryVectorStore] with the default (brute-force) index strategy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new builder for configuring an [InMemoryVectorStore].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rig_core::vector_store::in_memory_store::InMemoryVectorStore;
    ///
    /// let store = InMemoryVectorStore::builder()
    ///     .documents(documents)
    ///     .build()?;
    /// ```
    pub fn builder() -> InMemoryVectorStoreBuilder {
        InMemoryVectorStoreBuilder::new()
    }

    /// Internal constructor used by the builder.
    pub(super) fn from_builder(
        embeddings: HashMap<String, (Value, OneOrMany<Embedding>)>,
        index_strategy: IndexStrategy,
    ) -> Self {
        let mut inner = Inner {
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
            Self::initialize_lsh_index(&mut inner, num_tables, num_hyperplanes);
        }

        Self {
            inner: RwLock::new(inner),
        }
    }

    /// Create a new [InMemoryVectorStore] from documents and their corresponding embeddings.
    /// Ids are automatically generated have will have the form `"doc{n}"` where `n`
    /// is the index of the document.
    ///
    /// Uses BruteForce index strategy by default. For custom index strategies, use [InMemoryVectorStore::builder].
    pub fn from_documents<D: Serialize>(
        documents: impl IntoIterator<Item = (D, OneOrMany<Embedding>)>,
    ) -> Result<Self, VectorStoreError> {
        Self::builder().documents(documents).build()
    }

    /// Create a new [InMemoryVectorStore] from documents and their corresponding embeddings with ids.
    ///
    /// Uses BruteForce index strategy by default. For custom index strategies, use [InMemoryVectorStore::builder].
    pub fn from_documents_with_ids<D: Serialize>(
        documents: impl IntoIterator<Item = (impl ToString, D, OneOrMany<Embedding>)>,
    ) -> Result<Self, VectorStoreError> {
        Self::builder().documents_with_ids(documents).build()
    }

    /// Create a new [InMemoryVectorStore] from documents and their corresponding embeddings.
    /// Document ids are generated using the provided function.
    ///
    /// Uses BruteForce index strategy by default. For custom index strategies, use [InMemoryVectorStore::builder].
    pub fn from_documents_with_id_f<D: Serialize>(
        documents: impl IntoIterator<Item = (D, OneOrMany<Embedding>)>,
        f: fn(&D) -> String,
    ) -> Result<Self, VectorStoreError> {
        Self::builder().documents_with_id_f(documents, f).build()
    }

    /// Insert precomputed records into the store.
    ///
    /// Records with an existing id replace the previous entry.
    pub async fn insert(&self, records: Vec<StoreRecord>) -> Result<(), VectorStoreError> {
        let mut inner = self.inner.write().unwrap_or_else(PoisonError::into_inner);

        for StoreRecord {
            id,
            payload,
            embeddings,
        } in records
        {
            // Update LSH index if it exists
            if let Some(ref mut lsh_index) = inner.lsh_index {
                for embedding in embeddings.iter() {
                    lsh_index.insert(id.clone(), &embedding.vec);
                }
            }

            inner.embeddings.insert(id, (payload, embeddings));
        }

        Ok(())
    }

    /// Serializes each document and inserts it. Sugar over [`Self::insert`].
    pub async fn insert_as<T: Serialize>(
        &self,
        docs: Vec<(String, T, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let records = docs
            .into_iter()
            .map(|(id, doc, embeddings)| StoreRecord::new(id, &doc, embeddings))
            .collect::<Result<Vec<_>, _>>()?;
        self.insert(records).await
    }

    /// Returns the top N most similar documents for a pre-embedded query.
    ///
    /// Scores are cosine similarities: higher is better. Unlike the external
    /// backends, all query embeddings are considered: each document's score is
    /// the maximum cosine similarity over all (query embedding, document
    /// embedding) pairs. Results are sorted by descending score.
    pub async fn top_n(
        &self,
        req: VectorSearchRequest,
    ) -> Result<Vec<SearchHit>, VectorStoreError> {
        let inner = self.read();
        Self::vector_search(
            &inner,
            req.query(),
            req.samples() as usize,
            req.filter().as_ref(),
            req.threshold(),
        )
    }

    /// Returns the top N most similar document IDs as `(score, id)` tuples.
    ///
    /// Results are sorted by descending similarity score.
    pub async fn top_n_ids(
        &self,
        req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(self
            .top_n(req)
            .await?
            .into_iter()
            .map(|hit| (hit.score, hit.id))
            .collect())
    }

    /// Returns the top N most similar documents deserialized into `T` as
    /// `(score, id, document)` tuples. Sugar over [`Self::top_n`].
    pub async fn top_n_as<T: DeserializeOwned>(
        &self,
        req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.top_n(req)
            .await?
            .into_iter()
            .map(|hit| {
                let doc = serde_json::from_value(hit.payload)?;
                Ok((hit.score, hit.id, doc))
            })
            .collect()
    }

    /// Get the document by its id and deserialize it into the given type.
    pub fn get_document<T: DeserializeOwned>(
        &self,
        id: &str,
    ) -> Result<Option<T>, VectorStoreError> {
        self.read()
            .embeddings
            .get(id)
            .map(|(doc, _)| serde_json::from_value(doc.clone()))
            .transpose()
            .map_err(VectorStoreError::JsonError)
    }

    /// Returns a snapshot of the stored `(id, payload, embeddings)` entries.
    pub fn entries(&self) -> Vec<(String, Value, OneOrMany<Embedding>)> {
        self.read()
            .embeddings
            .iter()
            .map(|(id, (doc, embeddings))| (id.clone(), doc.clone(), embeddings.clone()))
            .collect()
    }

    /// Number of documents in the store.
    pub fn len(&self) -> usize {
        self.read().embeddings.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.read().embeddings.is_empty()
    }

    fn read(&self) -> impl std::ops::Deref<Target = Inner> + '_ {
        self.inner.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Tests whether a document satisfies the (optional) metadata filter.
    fn satisfies_filter(doc: &Value, filter: Option<&Filter<serde_json::Value>>) -> bool {
        match filter {
            None => true,
            Some(filter) => filter.satisfies(doc),
        }
    }

    /// Scores one candidate document against the pre-embedded query.
    ///
    /// Returns the best similarity across all (query embedding, document
    /// embedding) pairs, or `None` when the document is filtered out, has no
    /// finite-similarity embedding, or scores below the threshold. Shared by
    /// the brute-force and LSH scans so the filter, threshold, and NaN
    /// handling live in exactly one place.
    fn score_candidate(
        doc: &Value,
        embeddings: &OneOrMany<Embedding>,
        queries: &OneOrMany<Embedding>,
        filter: Option<&Filter<serde_json::Value>>,
        threshold: Option<f64>,
    ) -> Option<OrderedFloat<f64>> {
        if !Self::satisfies_filter(doc, filter) {
            return None;
        }

        // Best (highest-similarity) embedding for this document, across all
        // query embeddings.
        //
        // A zero-magnitude embedding yields a NaN similarity, which sorts as the
        // maximum under `OrderedFloat` and slips past `distance < threshold`
        // (every comparison with NaN is false). Drop non-finite similarities
        // *before* selecting the max so a document still ranks by its best
        // finite embedding; the document is skipped only when it has no finite
        // similarity at all.
        let distance = embeddings
            .iter()
            .flat_map(|embedding| {
                queries
                    .iter()
                    .map(|query| OrderedFloat(embedding.cosine_similarity(query, false)))
            })
            .filter(|distance| distance.0.is_finite())
            .max()?;

        // Skip documents below the similarity threshold.
        if threshold.is_some_and(|t| distance.0 < t) {
            return None;
        }

        Some(distance)
    }

    /// Implement vector search on [InMemoryVectorStore].
    ///
    /// The metadata `filter` and similarity `threshold` are applied *during* the
    /// scan, before the top-`n` selection, so results match backends that filter
    /// server-side rather than returning the unfiltered top-`n`.
    fn vector_search(
        inner: &Inner,
        queries: &OneOrMany<Embedding>,
        n: usize,
        filter: Option<&Filter<serde_json::Value>>,
        threshold: Option<f64>,
    ) -> Result<Vec<SearchHit>, VectorStoreError> {
        let ranking = match &inner.index_strategy {
            IndexStrategy::BruteForce => {
                Self::vector_search_brute_force(inner, queries, n, filter, threshold)
            }
            IndexStrategy::LSH { .. } => {
                Self::vector_search_lsh(inner, queries, n, filter, threshold)
            }
        };

        // `into_sorted_vec` sorts ascending by `Reverse`, i.e. descending by
        // similarity score, which is exactly the order callers expect.
        Ok(ranking
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(RankingItem(distance, id, doc))| SearchHit {
                id: id.clone(),
                score: distance.0,
                payload: doc.clone(),
            })
            .collect())
    }

    /// Brute force vector search - checks all documents
    fn vector_search_brute_force<'a>(
        inner: &'a Inner,
        queries: &OneOrMany<Embedding>,
        n: usize,
        filter: Option<&Filter<serde_json::Value>>,
        threshold: Option<f64>,
    ) -> EmbeddingRanking<'a> {
        // Sort documents by best embedding distance
        let mut docs = BinaryHeap::new();

        for (id, (doc, embeddings)) in inner.embeddings.iter() {
            let Some(distance) = Self::score_candidate(doc, embeddings, queries, filter, threshold)
            else {
                continue;
            };

            docs.push(Reverse(RankingItem(distance, id, doc)));

            // If the heap size exceeds n, pop the least old element.
            if docs.len() > n {
                docs.pop();
            }
        }

        // Log selected documents with their distances
        tracing::info!(target: "rig",
            "Selected documents: {}",
            docs.iter()
                .map(|Reverse(RankingItem(distance, id, _))| format!("{id} ({distance})"))
                .collect::<Vec<String>>()
                .join(", ")
        );

        docs
    }

    /// LSH-based vector search - uses LSH to find candidates then computes exact distances
    fn vector_search_lsh<'a>(
        inner: &'a Inner,
        queries: &OneOrMany<Embedding>,
        n: usize,
        filter: Option<&Filter<serde_json::Value>>,
        threshold: Option<f64>,
    ) -> EmbeddingRanking<'a> {
        // If we don't have an LSH index yet, fall back to brute force
        let Some(lsh_index) = inner.lsh_index.as_ref() else {
            tracing::warn!("LSH index not initialized, falling back to brute force search");
            return Self::vector_search_brute_force(inner, queries, n, filter, threshold);
        };

        // Collect candidates across all query embeddings.
        let mut candidates = queries
            .iter()
            .flat_map(|query| lsh_index.query(&query.vec))
            .collect::<Vec<_>>();
        candidates.sort();
        candidates.dedup();

        let mut docs = BinaryHeap::new();

        for candidate_id in candidates {
            if let Some((id_ref, (doc, embeddings))) = inner.embeddings.get_key_value(&candidate_id)
                && let Some(distance) =
                    Self::score_candidate(doc, embeddings, queries, filter, threshold)
            {
                docs.push(Reverse(RankingItem(distance, id_ref, doc)));

                if docs.len() > n {
                    docs.pop();
                }
            }
        }

        // Log selected documents with their distances
        tracing::info!(target: "rig",
            "Selected documents (LSH): {}",
            docs.iter()
                .map(|Reverse(RankingItem(distance, id, _))| format!("{id} ({distance})"))
                .collect::<Vec<String>>()
                .join(", ")
        );

        docs
    }

    /// Initialize LSH index from existing embeddings
    fn initialize_lsh_index(inner: &mut Inner, num_tables: usize, num_hyperplanes: usize) {
        if inner.embeddings.is_empty() {
            return;
        }

        // Get the dimension from the first embedding
        let first_embedding = inner
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
        for (id, (_, embeddings)) in inner.embeddings.iter() {
            for embedding in embeddings.iter() {
                lsh_index.insert(id.clone(), &embedding.vec);
            }
        }

        inner.lsh_index = Some(lsh_index);
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        OneOrMany,
        embeddings::embedding::Embedding,
        vector_store::{IndexStrategy, StoreRecord, request::VectorSearchRequest},
    };

    use super::InMemoryVectorStore;

    fn embedding(doc: &str, vec: Vec<f64>) -> OneOrMany<Embedding> {
        OneOrMany::one(Embedding {
            document: doc.to_string(),
            vec,
        })
    }

    #[test]
    fn test_auto_ids() {
        let vector_store = InMemoryVectorStore::builder()
            .index_strategy(IndexStrategy::LSH {
                num_tables: 5,
                num_hyperplanes: 10,
            })
            .documents(vec![
                ("glarb-garb", embedding("glarb-garb", vec![0.1, 0.1, 0.5])),
                (
                    "marble-marble",
                    embedding("marble-marble", vec![0.7, -0.3, 0.0]),
                ),
                ("flumb-flumb", embedding("flumb-flumb", vec![0.3, 0.7, 0.1])),
            ])
            .documents(vec![
                ("brotato", embedding("brotato", vec![0.3, 0.7, 0.1])),
                ("ping-pong", embedding("ping-pong", vec![0.7, -0.3, 0.0])),
            ])
            .build()
            .unwrap();

        let mut store = vector_store.entries();
        store.sort_by_key(|(id, _, _)| id.clone());

        assert_eq!(
            store,
            vec![
                (
                    "doc0".to_string(),
                    json!("glarb-garb"),
                    embedding("glarb-garb", vec![0.1, 0.1, 0.5])
                ),
                (
                    "doc1".to_string(),
                    json!("marble-marble"),
                    embedding("marble-marble", vec![0.7, -0.3, 0.0])
                ),
                (
                    "doc2".to_string(),
                    json!("flumb-flumb"),
                    embedding("flumb-flumb", vec![0.3, 0.7, 0.1])
                ),
                (
                    "doc3".to_string(),
                    json!("brotato"),
                    embedding("brotato", vec![0.3, 0.7, 0.1])
                ),
                (
                    "doc4".to_string(),
                    json!("ping-pong"),
                    embedding("ping-pong", vec![0.7, -0.3, 0.0])
                )
            ]
        );
    }

    #[tokio::test]
    async fn test_single_embedding() {
        let vector_store = InMemoryVectorStore::builder()
            .index_strategy(IndexStrategy::LSH {
                num_tables: 5,
                num_hyperplanes: 10,
            })
            .documents_with_ids(vec![
                (
                    "doc1",
                    "glarb-garb",
                    embedding("glarb-garb", vec![0.1, 0.1, 0.5]),
                ),
                (
                    "doc2",
                    "marble-marble",
                    embedding("marble-marble", vec![0.7, -0.3, 0.0]),
                ),
                (
                    "doc3",
                    "flumb-flumb",
                    embedding("flumb-flumb", vec![0.3, 0.7, 0.1]),
                ),
            ])
            .build()
            .unwrap();

        let hits = vector_store
            .top_n(VectorSearchRequest::new(
                OneOrMany::one(Embedding {
                    document: "glarby-glarble".to_string(),
                    vec: vec![0.0, 0.1, 0.6],
                }),
                1,
            ))
            .await
            .unwrap();

        assert_eq!(
            hits.into_iter()
                .map(|hit| (hit.score, hit.id, hit.payload))
                .collect::<Vec<_>>(),
            vec![(0.9807965956109156, "doc1".to_string(), json!("glarb-garb"))]
        );
    }

    #[tokio::test]
    async fn test_multiple_embeddings() {
        let vector_store = InMemoryVectorStore::builder()
            .index_strategy(IndexStrategy::LSH {
                num_tables: 5,
                num_hyperplanes: 10,
            })
            .documents_with_ids(vec![
                (
                    "doc1",
                    "glarb-garb",
                    OneOrMany::many(vec![
                        Embedding {
                            document: "glarb-garb".to_string(),
                            vec: vec![0.1, 0.1, 0.5],
                        },
                        Embedding {
                            document: "don't-choose-me".to_string(),
                            vec: vec![-0.5, 0.9, 0.1],
                        },
                    ])
                    .unwrap(),
                ),
                (
                    "doc2",
                    "marble-marble",
                    OneOrMany::many(vec![
                        Embedding {
                            document: "marble-marble".to_string(),
                            vec: vec![0.7, -0.3, 0.0],
                        },
                        Embedding {
                            document: "sandwich".to_string(),
                            vec: vec![0.5, 0.5, -0.7],
                        },
                    ])
                    .unwrap(),
                ),
                (
                    "doc3",
                    "flumb-flumb",
                    OneOrMany::many(vec![
                        Embedding {
                            document: "flumb-flumb".to_string(),
                            vec: vec![0.3, 0.7, 0.1],
                        },
                        Embedding {
                            document: "banana".to_string(),
                            vec: vec![0.1, -0.5, -0.5],
                        },
                    ])
                    .unwrap(),
                ),
            ])
            .build()
            .unwrap();

        let hits = vector_store
            .top_n(VectorSearchRequest::new(
                OneOrMany::one(Embedding {
                    document: "glarby-glarble".to_string(),
                    vec: vec![0.0, 0.1, 0.6],
                }),
                1,
            ))
            .await
            .unwrap();

        assert_eq!(
            hits.into_iter()
                .map(|hit| (hit.score, hit.id, hit.payload))
                .collect::<Vec<_>>(),
            vec![(0.9807965956109156, "doc1".to_string(), json!("glarb-garb"))]
        );
    }

    #[tokio::test]
    async fn top_n_honors_filter_and_threshold() {
        use crate::vector_store::request::{Filter, SearchFilter};
        use serde::Serialize;

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

        // Embed every query as this fixed 10-dim vector; give every document the
        // same embedding so all cosine similarities are 1.0 and only the
        // filter/threshold decide the result set.
        let vec = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let doc_embedding = |doc: &str| embedding(doc, vec.clone());
        let query_embedding = || Embedding {
            document: "q".to_string(),
            vec: vec.clone(),
        };

        let store = InMemoryVectorStore::from_documents_with_ids(vec![
            ("a", item("fruit", "banana"), doc_embedding("banana")),
            ("b", item("veg", "carrot"), doc_embedding("carrot")),
            ("c", item("fruit", "apple"), doc_embedding("apple")),
        ])
        .unwrap();

        let ids = |req| async {
            let mut out: Vec<String> = store
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
        let all = ids(VectorSearchRequest::new(
            OneOrMany::one(query_embedding()),
            10,
        ))
        .await;
        assert_eq!(all, vec!["a", "b", "c"]);

        // Metadata filter: only documents whose `category` field is `fruit`.
        let fruit = ids(
            VectorSearchRequest::new(OneOrMany::one(query_embedding()), 10)
                .with_filter(Filter::eq("category", json!("fruit"))),
        )
        .await;
        assert_eq!(fruit, vec!["a", "c"]);

        // Threshold above the maximum similarity (1.0): nothing qualifies.
        let none = ids(
            VectorSearchRequest::new(OneOrMany::one(query_embedding()), 10).with_threshold(2.0),
        )
        .await;
        assert!(none.is_empty());

        // Threshold at or below the similarity keeps all matches.
        let kept = ids(
            VectorSearchRequest::new(OneOrMany::one(query_embedding()), 10).with_threshold(0.5),
        )
        .await;
        assert_eq!(kept, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn top_n_excludes_non_finite_similarity() {
        // The zero-magnitude embedding produces a NaN cosine similarity, which
        // sorts as the maximum under OrderedFloat. It must not rank first (or
        // appear at all), even with no threshold set.
        let store = InMemoryVectorStore::from_documents_with_ids(vec![
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
        .unwrap();

        let ids: Vec<String> = store
            .top_n_ids(VectorSearchRequest::new(
                OneOrMany::one(Embedding {
                    document: "q".to_string(),
                    vec: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                }),
                10,
            ))
            .await
            .unwrap()
            .into_iter()
            .map(|(_, id)| id)
            .collect();
        assert_eq!(ids, vec!["good".to_string()]);
    }

    #[tokio::test]
    async fn top_n_ranks_document_by_best_finite_embedding() {
        // A document that owns both a strong finite embedding and a degenerate
        // zero-magnitude (NaN) one must still be returned, ranked by the finite
        // embedding — not dropped because NaN sorts as the OrderedFloat maximum.
        let store = InMemoryVectorStore::from_documents_with_ids(vec![(
            "mixed",
            "mixed".to_string(),
            OneOrMany::many(vec![
                Embedding {
                    document: "good-chunk".to_string(),
                    vec: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                },
                Embedding {
                    document: "empty-chunk".to_string(),
                    vec: vec![0.0; 10],
                },
            ])
            .unwrap(),
        )])
        .unwrap();

        let results = store
            .top_n_ids(VectorSearchRequest::new(
                OneOrMany::one(Embedding {
                    document: "q".to_string(),
                    vec: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                }),
                10,
            ))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results.first().unwrap().1, "mixed");
        assert!(results.first().unwrap().0.is_finite());
    }

    #[tokio::test]
    async fn insert_adds_records_and_updates_lsh_index() {
        let store = InMemoryVectorStore::builder()
            .index_strategy(IndexStrategy::LSH {
                num_tables: 5,
                num_hyperplanes: 10,
            })
            .documents_with_ids(vec![(
                "doc1",
                "glarb-garb",
                embedding("glarb-garb", vec![0.1, 0.1, 0.5]),
            )])
            .build()
            .unwrap();

        store
            .insert(vec![
                StoreRecord::new(
                    "doc2",
                    &"marble-marble",
                    embedding("marble-marble", vec![0.7, -0.3, 0.0]),
                )
                .unwrap(),
            ])
            .await
            .unwrap();

        assert_eq!(store.len(), 2);

        // The freshly inserted record is discoverable through the LSH index.
        let hits = store
            .top_n(VectorSearchRequest::new(
                OneOrMany::one(Embedding {
                    document: "marbly".to_string(),
                    vec: vec![0.7, -0.3, 0.0],
                }),
                1,
            ))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits.first().unwrap().id, "doc2");
        assert_eq!(hits.first().unwrap().payload, json!("marble-marble"));
    }

    #[tokio::test]
    async fn top_n_as_deserializes_payloads() {
        use serde::Deserialize;

        #[derive(Deserialize, Debug, PartialEq, serde::Serialize)]
        struct Doc {
            text: String,
        }

        let store = InMemoryVectorStore::new();
        store
            .insert_as(vec![(
                "doc1".to_string(),
                Doc {
                    text: "banana".to_string(),
                },
                embedding("banana", vec![0.1, 0.2, 0.3]),
            )])
            .await
            .unwrap();

        let results: Vec<(f64, String, Doc)> = store
            .top_n_as(VectorSearchRequest::new(
                OneOrMany::one(Embedding {
                    document: "q".to_string(),
                    vec: vec![0.1, 0.2, 0.3],
                }),
                1,
            ))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        let (score, id, doc) = results.into_iter().next().unwrap();
        assert!(score > 0.99);
        assert_eq!(id, "doc1");
        assert_eq!(
            doc,
            Doc {
                text: "banana".to_string()
            }
        );
    }
}
