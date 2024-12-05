//! In-memory implementation of a vector store.
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use super::{VectorStoreError, VectorStoreIndex};
use crate::{
    embeddings::{distance::VectorDistance, Embedding, EmbeddingModel},
    OneOrMany,
};

/// [InMemoryVectorStore] is a simple in-memory vector store that stores embeddings
/// in-memory using a HashMap.
#[derive(Clone, Default)]
pub struct InMemoryVectorStore<D: Serialize> {
    /// The embeddings are stored in a HashMap.
    /// Hashmap key is the document id.
    /// Hashmap value is a tuple of the serializable document and its corresponding embeddings.
    embeddings: HashMap<String, (D, OneOrMany<Embedding>)>,
}

impl<D: Serialize + Eq> InMemoryVectorStore<D> {
    /// Create a new [InMemoryVectorStore] from documents and their corresponding embeddings.
    /// Ids are automatically generated have will have the form `"doc{n}"` where `n`
    /// is the index of the document.
    pub fn from_documents(documents: impl IntoIterator<Item = (D, OneOrMany<Embedding>)>) -> Self {
        let mut store = HashMap::new();
        documents
            .into_iter()
            .enumerate()
            .for_each(|(i, (doc, embeddings))| {
                store.insert(format!("doc{i}"), (doc, embeddings));
            });

        Self { embeddings: store }
    }

    /// Create a new [InMemoryVectorStore] from documents and and their corresponding embeddings with ids.
    pub fn from_documents_with_ids(
        documents: impl IntoIterator<Item = (impl ToString, D, OneOrMany<Embedding>)>,
    ) -> Self {
        let mut store = HashMap::new();
        documents.into_iter().for_each(|(i, doc, embeddings)| {
            store.insert(i.to_string(), (doc, embeddings));
        });

        Self { embeddings: store }
    }

    /// Create a new [InMemoryVectorStore] from documents and their corresponding embeddings.
    /// Document ids are generated using the provided function.
    pub fn from_documents_with_id_f(
        documents: impl IntoIterator<Item = (D, OneOrMany<Embedding>)>,
        f: fn(&D) -> String,
    ) -> Self {
        let mut store = HashMap::new();
        documents.into_iter().for_each(|(doc, embeddings)| {
            store.insert(f(&doc), (doc, embeddings));
        });

        Self { embeddings: store }
    }

    /// Implement vector search on [InMemoryVectorStore].
    /// To be used by implementations of [VectorStoreIndex::top_n] and [VectorStoreIndex::top_n_ids] methods.
    fn vector_search(&self, prompt_embedding: &Embedding, n: usize) -> EmbeddingRanking<D> {
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
                .map(|Reverse(RankingItem(distance, id, _, _))| format!("{} ({})", id, distance))
                .collect::<Vec<String>>()
                .join(", ")
        );

        docs
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
                self.embeddings
                    .insert(format!("doc{}", index + current_index), (doc, embeddings));
            });
    }

    /// Add documents and their corresponding embeddings to the store with ids.
    pub fn add_documents_with_ids(
        &mut self,
        documents: impl IntoIterator<Item = (impl ToString, D, OneOrMany<Embedding>)>,
    ) {
        documents.into_iter().for_each(|(id, doc, embeddings)| {
            self.embeddings.insert(id.to_string(), (doc, embeddings));
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
            self.embeddings.insert(id, (doc, embeddings));
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
    async fn top_n<T: for<'a> Deserialize<'a>>(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let prompt_embedding = &self.model.embed_text(query).await?;

        let docs = self.store.vector_search(prompt_embedding, n);

        // Return n best
        docs.into_iter()
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
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let prompt_embedding = &self.model.embed_text(query).await?;

        let docs = self.store.vector_search(prompt_embedding, n);

        // Return n best
        docs.into_iter()
            .map(|Reverse(RankingItem(distance, id, _, _))| Ok((distance.0, id.clone())))
            .collect::<Result<Vec<_>, _>>()
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Reverse;

    use crate::{embeddings::embedding::Embedding, OneOrMany};

    use super::{InMemoryVectorStore, RankingItem};

    #[test]
    fn test_auto_ids() {
        let mut vector_store = InMemoryVectorStore::from_documents(vec![
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
        ]);

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
        let vector_store = InMemoryVectorStore::from_documents_with_ids(vec![
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
        ]);

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
        let vector_store = InMemoryVectorStore::from_documents_with_ids(vec![
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
        ]);

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
