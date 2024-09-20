//! In-memory implementation of a vector store.
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use super::{VectorStore, VectorStoreError, VectorStoreIndex};
use crate::embeddings::{DocumentEmbeddings, Embedding, EmbeddingModel, EmbeddingsBuilder};

/// InMemoryVectorStore is a simple in-memory vector store that stores embeddings
/// in-memory using a HashMap.
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct InMemoryVectorStore {
    /// The embeddings are stored in a HashMap with the document ID as the key.
    embeddings: HashMap<String, DocumentEmbeddings>,
}

/// RankingItem(distance, document_id, document, embed_doc)
#[derive(Eq, PartialEq)]
struct RankingItem<'a>(
    OrderedFloat<f64>,
    &'a String,
    &'a DocumentEmbeddings,
    &'a String,
);

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

impl VectorStore for InMemoryVectorStore {
    type Q = ();

    async fn add_documents(
        &mut self,
        documents: Vec<DocumentEmbeddings>,
    ) -> Result<(), VectorStoreError> {
        for doc in documents {
            self.embeddings.insert(doc.id.clone(), doc);
        }

        Ok(())
    }

    async fn get_document<T: for<'a> Deserialize<'a>>(
        &self,
        id: &str,
    ) -> Result<Option<T>, VectorStoreError> {
        Ok(self
            .embeddings
            .get(id)
            .map(|document| serde_json::from_value(document.document.clone()))
            .transpose()?)
    }

    async fn get_document_embeddings(
        &self,
        id: &str,
    ) -> Result<Option<DocumentEmbeddings>, VectorStoreError> {
        Ok(self.embeddings.get(id).cloned())
    }

    async fn get_document_by_query(
        &self,
        _query: Self::Q,
    ) -> Result<Option<DocumentEmbeddings>, VectorStoreError> {
        Ok(None)
    }
}

impl InMemoryVectorStore {
    pub fn index<M: EmbeddingModel>(self, model: M) -> InMemoryVectorIndex<M> {
        InMemoryVectorIndex::new(model, self)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &DocumentEmbeddings)> {
        self.embeddings.iter()
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Uitilty method to create an InMemoryVectorStore from a list of embeddings.
    pub async fn from_embeddings(
        embeddings: Vec<DocumentEmbeddings>,
    ) -> Result<Self, VectorStoreError> {
        let mut store = Self::default();
        store.add_documents(embeddings).await?;
        Ok(store)
    }

    /// Create an InMemoryVectorStore from a list of documents.
    /// The documents are serialized to JSON and embedded using the provided embedding model.
    /// The resulting embeddings are stored in an InMemoryVectorStore created by the method.
    pub async fn from_documents<M: EmbeddingModel, T: Serialize>(
        embedding_model: M,
        documents: &[(String, T)],
    ) -> Result<Self, VectorStoreError> {
        let embeddings = documents
            .iter()
            .fold(
                EmbeddingsBuilder::new(embedding_model),
                |builder, (id, doc)| {
                    builder.json_document(
                        id,
                        serde_json::to_value(doc).expect("Document should be serializable"),
                        vec![serde_json::to_string(doc).expect("Document should be serializable")],
                    )
                },
            )
            .build()
            .await?;

        let store = Self::from_embeddings(embeddings).await?;
        Ok(store)
    }
}

pub struct InMemoryVectorIndex<M: EmbeddingModel> {
    model: M,
    pub store: InMemoryVectorStore,
}

impl<M: EmbeddingModel> InMemoryVectorIndex<M> {
    pub fn new(model: M, store: InMemoryVectorStore) -> Self {
        Self { model, store }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &DocumentEmbeddings)> {
        self.store.iter()
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Create an InMemoryVectorIndex from a list of documents.
    /// The documents are serialized to JSON and embedded using the provided embedding model.
    /// The resulting embeddings are stored in an InMemoryVectorStore created by the method.
    /// The InMemoryVectorIndex is then created from the store and the provided query model.
    pub async fn from_documents<T: Serialize>(
        embedding_model: M,
        query_model: M,
        documents: &[(String, T)],
    ) -> Result<Self, VectorStoreError> {
        let mut store = InMemoryVectorStore::default();

        let embeddings = documents
            .iter()
            .fold(
                EmbeddingsBuilder::new(embedding_model),
                |builder, (id, doc)| {
                    builder.json_document(
                        id,
                        serde_json::to_value(doc).expect("Document should be serializable"),
                        vec![serde_json::to_string(doc).expect("Document should be serializable")],
                    )
                },
            )
            .build()
            .await?;

        store.add_documents(embeddings).await?;
        Ok(store.index(query_model))
    }

    /// Utility method to create an InMemoryVectorIndex from a list of embeddings
    /// and an embedding model.
    pub async fn from_embeddings(
        query_model: M,
        embeddings: Vec<DocumentEmbeddings>,
    ) -> Result<Self, VectorStoreError> {
        let store = InMemoryVectorStore::from_embeddings(embeddings).await?;
        Ok(store.index(query_model))
    }
}

impl<M: EmbeddingModel + std::marker::Sync> VectorStoreIndex for InMemoryVectorIndex<M> {
    async fn top_n_from_query(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, DocumentEmbeddings)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_document(query).await?;
        self.top_n_from_embedding(&prompt_embedding, n).await
    }

    async fn top_n_from_embedding(
        &self,
        query_embedding: &Embedding,
        n: usize,
    ) -> Result<Vec<(f64, DocumentEmbeddings)>, VectorStoreError> {
        // Sort documents by best embedding distance
        let mut docs: EmbeddingRanking = BinaryHeap::new();

        for (id, doc_embeddings) in self.store.embeddings.iter() {
            // Get the best context for the document given the prompt
            if let Some((distance, embed_doc)) = doc_embeddings
                .embeddings
                .iter()
                .map(|embedding| {
                    (
                        OrderedFloat(embedding.distance(query_embedding)),
                        &embedding.document,
                    )
                })
                .min_by(|a, b| a.0.cmp(&b.0))
            {
                docs.push(Reverse(RankingItem(
                    distance,
                    id,
                    doc_embeddings,
                    embed_doc,
                )));
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

        // Return n best
        Ok(docs
            .into_iter()
            .map(|Reverse(RankingItem(distance, _, doc, _))| (distance.0, doc.clone()))
            .collect())
    }
}
