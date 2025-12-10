//! In-memory implementation of a vector store.
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use super::{IndexStrategy, VectorStoreError, VectorStoreIndex, request::VectorSearchRequest};
use crate::{
    OneOrMany,
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
    embeddings: HashMap<String, (D, OneOrMany<Embedding>)>,

    index_strategy: IndexStrategy,

    lsh_index: Option<LSHIndex>,
}

impl<D: Serialize + Eq> InMemoryVectorStore<D> {
    /// Create a new builder for configuring an [InMemoryVectorStore].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rig::vector_store::InMemoryVectorStore;
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
        embeddings: HashMap<String, (D, OneOrMany<Embedding>)>,
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
    pub fn from_documents(documents: impl IntoIterator<Item = (D, OneOrMany<Embedding>)>) -> Self {
        let mut store = HashMap::new();
        documents
            .into_iter()
            .enumerate()
            .for_each(|(i, (doc, embeddings))| {
                store.insert(format!("doc{i}"), (doc, embeddings));
            });

        Self {
            embeddings: store,
            index_strategy: IndexStrategy::default(),
            lsh_index: None,
        }
    }

    /// Create a new [InMemoryVectorStore] from documents and their corresponding embeddings with ids.
    ///
    /// Uses BruteForce index strategy by default. For custom index strategies, use [InMemoryVectorStore::builder].
    pub fn from_documents_with_ids(
        documents: impl IntoIterator<Item = (impl ToString, D, OneOrMany<Embedding>)>,
    ) -> Self {
        let mut store = HashMap::new();
        documents.into_iter().for_each(|(i, doc, embeddings)| {
            store.insert(i.to_string(), (doc, embeddings));
        });

        Self {
            embeddings: store,
            index_strategy: IndexStrategy::default(),
            lsh_index: None,
        }
    }

    /// Create a new [InMemoryVectorStore] from documents and their corresponding embeddings.
    /// Document ids are generated using the provided function.
    ///
    /// Uses BruteForce index strategy by default. For custom index strategies, use [InMemoryVectorStore::builder].
    pub fn from_documents_with_id_f(
        documents: impl IntoIterator<Item = (D, OneOrMany<Embedding>)>,
        f: fn(&D) -> String,
    ) -> Self {
        let mut store = HashMap::new();
        documents.into_iter().for_each(|(doc, embeddings)| {
            store.insert(f(&doc), (doc, embeddings));
        });

        Self {
            embeddings: store,
            index_strategy: IndexStrategy::default(),
            lsh_index: None,
        }
    }

    /// Implement vector search on [InMemoryVectorStore].
    /// To be used by implementations of [VectorStoreIndex::top_n] and [VectorStoreIndex::top_n_ids] methods.
    fn vector_search(&self, prompt_embedding: &Embedding, n: usize) -> EmbeddingRanking<'_, D> {
        match &self.index_strategy {
            IndexStrategy::BruteForce => self.vector_search_brute_force(prompt_embedding, n),
            IndexStrategy::LSH {
                num_tables,
                num_hyperplanes,
            } => self.vector_search_lsh(prompt_embedding, n, *num_tables, *num_hyperplanes),
        }
    }

    /// Brute force vector search - checks all documents
    fn vector_search_brute_force(
        &self,
        prompt_embedding: &Embedding,
        n: usize,
    ) -> EmbeddingRanking<'_, D> {
        // Sort documents by best embedding distance
        let mut docs = BinaryHeap::new();

        for (id, (doc, embeddings)) in self.embeddings.iter() {
            // Get the best context for the document given the prompt
            if let Some((distance, embed_doc)) = embeddings
                .iter()
                .map(|embedding| {
                    (
                        OrderedFloat(embedding.cosine_similarity(prompt_embedding, false)),
                        &embedding.document,
                    )
                })
                .max_by(|a, b| a.0.cmp(&b.0))
            {
                docs.push(Reverse(RankingItem(distance, id, doc, embed_doc)));
            };

            // If the heap size exceeds n, pop the least old element.
            if docs.len() > n {
                docs.pop();
            }
        }

        // Log selected tools with their distances
        tracing::info!(target: "rig",
            "Selected documents: {}",
            docs.iter()
                .map(|Reverse(RankingItem(distance, id, _, _))| format!("{id} ({distance})"))
                .collect::<Vec<String>>()
                .join(", ")
        );

        docs
    }

    /// LSH-based vector search - uses LSH to find candidates then computes exact distances
    fn vector_search_lsh(
        &self,
        prompt_embedding: &Embedding,
        n: usize,
        _num_tables: usize,
        _num_hyperplanes: usize,
    ) -> EmbeddingRanking<'_, D> {
        // If we don't have an LSH index yet, fall back to brute force
        if self.lsh_index.is_none() {
            tracing::warn!("LSH index not initialized, falling back to brute force search");
            return self.vector_search_brute_force(prompt_embedding, n);
        }

        let lsh_index = self.lsh_index.as_ref().unwrap();
        let candidates = lsh_index.query(&prompt_embedding.vec);

        // Sort documents by best embedding distance, but only check candidates
        let mut docs = BinaryHeap::new();

        // Collect all matching documents with their scores first
        let mut scored_docs = Vec::new();

        for candidate_id in candidates {
            if let Some((doc, embeddings)) = self.embeddings.get(&candidate_id) {
                // Get the best context for the document given the prompt
                if let Some((distance, embed_doc)) = embeddings
                    .iter()
                    .map(|embedding| {
                        (
                            OrderedFloat(embedding.cosine_similarity(prompt_embedding, false)),
                            &embedding.document,
                        )
                    })
                    .max_by(|a, b| a.0.cmp(&b.0))
                {
                    scored_docs.push((distance, candidate_id, doc, embed_doc));
                }
            }
        }

        // Sort by distance and take top n
        scored_docs.sort_by(|a, b| b.0.cmp(&a.0)); // Sort in descending order (highest similarity first)
        scored_docs.truncate(n);

        // Convert to BinaryHeap format using the original HashMap keys
        for (distance, candidate_id, doc, embed_doc) in scored_docs {
            if let Some((id_ref, _)) = self.embeddings.iter().find(|(k, _)| **k == candidate_id) {
                docs.push(Reverse(RankingItem(distance, id_ref, doc, embed_doc)));
            }
        }

        // Log selected tools with their distances
        tracing::info!(target: "rig",
            "Selected documents (LSH): {}",
            docs.iter()
                .map(|Reverse(RankingItem(distance, id, _, _))| format!("{id} ({distance})"))
                .collect::<Vec<String>>()
                .join(", ")
        );

        docs
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
    pub fn add_documents(
        &mut self,
        documents: impl IntoIterator<Item = (D, OneOrMany<Embedding>)>,
    ) {
        let current_index = self.embeddings.len();
        documents
            .into_iter()
            .enumerate()
            .for_each(|(index, (doc, embeddings))| {
                let id = format!("doc{}", index + current_index);
                self.embeddings
                    .insert(id.clone(), (doc, embeddings.clone()));

                // Update LSH index if it exists
                if let Some(ref mut lsh_index) = self.lsh_index {
                    for embedding in embeddings.iter() {
                        lsh_index.insert(id.clone(), &embedding.vec);
                    }
                }
            });
    }

    /// Add documents and their corresponding embeddings to the store with ids.
    pub fn add_documents_with_ids(
        &mut self,
        documents: impl IntoIterator<Item = (impl ToString, D, OneOrMany<Embedding>)>,
    ) {
        documents.into_iter().for_each(|(id, doc, embeddings)| {
            let id_str = id.to_string();
            self.embeddings
                .insert(id_str.clone(), (doc, embeddings.clone()));

            // Update LSH index if it exists
            if let Some(ref mut lsh_index) = self.lsh_index {
                for embedding in embeddings.iter() {
                    lsh_index.insert(id_str.clone(), &embedding.vec);
                }
            }
        });
    }

    /// Add documents and their corresponding embeddings to the store.
    /// Document ids are generated using the provided function.
    pub fn add_documents_with_id_f(
        &mut self,
        documents: Vec<(D, OneOrMany<Embedding>)>,
        f: fn(&D) -> String,
    ) {
        for (doc, embeddings) in documents {
            let id = f(&doc);
            self.embeddings
                .insert(id.clone(), (doc, embeddings.clone()));

            // Update LSH index if it exists
            if let Some(ref mut lsh_index) = self.lsh_index {
                for embedding in embeddings.iter() {
                    lsh_index.insert(id.clone(), &embedding.vec);
                }
            }
        }
    }

    /// Get the document by its id and deserialize it into the given type.
    pub fn get_document<T: for<'a> Deserialize<'a>>(
        &self,
        id: &str,
    ) -> Result<Option<T>, VectorStoreError> {
        Ok(self
            .embeddings
            .get(id)
            .map(|(doc, _)| serde_json::from_str(&serde_json::to_string(doc)?))
            .transpose()?)
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

    pub fn iter(&self) -> impl Iterator<Item = (&String, &(D, OneOrMany<Embedding>))> {
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

    pub fn iter(&self) -> impl Iterator<Item = (&String, &(D, OneOrMany<Embedding>))> {
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

        let docs = self
            .store
            .vector_search(prompt_embedding, req.samples() as usize);

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

        let docs = self
            .store
            .vector_search(prompt_embedding, req.samples() as usize);

        docs.into_iter()
            .map(|Reverse(RankingItem(distance, id, _, _))| Ok((distance.0, id.clone())))
            .collect::<Result<Vec<_>, _>>()
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Reverse;

    use crate::{OneOrMany, embeddings::embedding::Embedding, vector_store::IndexStrategy};

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
                    OneOrMany::one(Embedding {
                        document: "glarb-garb".to_string(),
                        vec: vec![0.1, 0.1, 0.5],
                    }),
                ),
                (
                    "marble-marble",
                    OneOrMany::one(Embedding {
                        document: "marble-marble".to_string(),
                        vec: vec![0.7, -0.3, 0.0],
                    }),
                ),
                (
                    "flumb-flumb",
                    OneOrMany::one(Embedding {
                        document: "flumb-flumb".to_string(),
                        vec: vec![0.3, 0.7, 0.1],
                    }),
                ),
            ])
            .build();

        vector_store.add_documents(vec![
            (
                "brotato",
                OneOrMany::one(Embedding {
                    document: "brotato".to_string(),
                    vec: vec![0.3, 0.7, 0.1],
                }),
            ),
            (
                "ping-pong",
                OneOrMany::one(Embedding {
                    document: "ping-pong".to_string(),
                    vec: vec![0.7, -0.3, 0.0],
                }),
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
                        OneOrMany::one(Embedding {
                            document: "glarb-garb".to_string(),
                            vec: vec![0.1, 0.1, 0.5],
                        })
                    )
                ),
                (
                    "doc1".to_string(),
                    (
                        "marble-marble",
                        OneOrMany::one(Embedding {
                            document: "marble-marble".to_string(),
                            vec: vec![0.7, -0.3, 0.0],
                        })
                    )
                ),
                (
                    "doc2".to_string(),
                    (
                        "flumb-flumb",
                        OneOrMany::one(Embedding {
                            document: "flumb-flumb".to_string(),
                            vec: vec![0.3, 0.7, 0.1],
                        })
                    )
                ),
                (
                    "doc3".to_string(),
                    (
                        "brotato",
                        OneOrMany::one(Embedding {
                            document: "brotato".to_string(),
                            vec: vec![0.3, 0.7, 0.1],
                        })
                    )
                ),
                (
                    "doc4".to_string(),
                    (
                        "ping-pong",
                        OneOrMany::one(Embedding {
                            document: "ping-pong".to_string(),
                            vec: vec![0.7, -0.3, 0.0],
                        })
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
                    OneOrMany::one(Embedding {
                        document: "glarb-garb".to_string(),
                        vec: vec![0.1, 0.1, 0.5],
                    }),
                ),
                (
                    "doc2",
                    "marble-marble",
                    OneOrMany::one(Embedding {
                        document: "marble-marble".to_string(),
                        vec: vec![0.7, -0.3, 0.0],
                    }),
                ),
                (
                    "doc3",
                    "flumb-flumb",
                    OneOrMany::one(Embedding {
                        document: "flumb-flumb".to_string(),
                        vec: vec![0.3, 0.7, 0.1],
                    }),
                ),
            ])
            .build();

        let ranking = vector_store.vector_search(
            &Embedding {
                document: "glarby-glarble".to_string(),
                vec: vec![0.0, 0.1, 0.6],
            },
            1,
        );

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
            .build();

        let ranking = vector_store.vector_search(
            &Embedding {
                document: "glarby-glarble".to_string(),
                vec: vec![0.0, 0.1, 0.6],
            },
            1,
        );

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
}
