//! In-memory vector storage with exact cosine scoring.
//!
//! ```
//! use rig_core::vector_store::in_memory_store::InMemoryVectorStore;
//!
//! let namespace = String::from("prices");
//! let store = InMemoryVectorStore::from_documents_with_id_f(
//!     [(1.5_f64, Vec::new())],
//!     |price| format!("{namespace}:{price}"),
//! );
//! assert_eq!(store.len(), 1);
//! ```
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};

use ordered_float::OrderedFloat;
use serde::{Serialize, de::DeserializeOwned};

use super::{
    VectorSearchIdResult, VectorSearchResult, VectorStoreError, VectorStoreIndex,
    request::VectorSearchRequest,
};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::{
    driver::DynModel,
    embeddings::{
        Embedding,
        distance::{dot, magnitude},
    },
    operation::Embedding as EmbeddingOp,
    vector_store::request::Filter,
};

/// Stores serializable documents and their embeddings by document ID.
/// Documents do not need equality or ordering implementations.
#[derive(Clone)]
pub struct InMemoryVectorStore<D: Serialize> {
    /// Each document and its embeddings, by document id.
    embeddings: HashMap<String, Stored<D>>,
}

/// A document, its embeddings and their lengths, measured once at insert so a
/// query scores each embedding with one dot product.
#[derive(Clone)]
struct Stored<D> {
    doc: D,
    embeddings: Vec<Embedding>,
    magnitudes: Vec<f64>,
}

impl<D: Serialize> Default for InMemoryVectorStore<D> {
    fn default() -> Self {
        Self {
            embeddings: HashMap::new(),
        }
    }
}

impl<D: Serialize> InMemoryVectorStore<D> {
    /// Create a new [InMemoryVectorStore] from documents and their corresponding embeddings.
    /// Ids are automatically generated have will have the form `"doc{n}"` where `n`
    /// is the index of the document.
    pub fn from_documents(documents: impl IntoIterator<Item = (D, Vec<Embedding>)>) -> Self {
        let mut store = Self::default();
        store.add_documents(documents);
        store
    }

    /// Create a new [InMemoryVectorStore] from documents and their corresponding embeddings with ids.
    pub fn from_documents_with_ids(
        documents: impl IntoIterator<Item = (impl ToString, D, Vec<Embedding>)>,
    ) -> Self {
        let mut store = Self::default();
        store.add_documents_with_ids(documents);
        store
    }

    /// Creates a store, calling `f` once per document in input order.
    /// The callback may capture and mutate state. Duplicate IDs replace earlier documents.
    pub fn from_documents_with_id_f(
        documents: impl IntoIterator<Item = (D, Vec<Embedding>)>,
        f: impl FnMut(&D) -> String,
    ) -> Self {
        let mut store = Self::default();
        store.add_documents_with_id_f(documents, f);
        store
    }

    /// Inserts or replaces the document `id`.
    fn insert(&mut self, id: String, doc: D, embeddings: Vec<Embedding>) {
        let magnitudes = embeddings.iter().map(|e| magnitude(&e.vec)).collect();
        let stored = Stored {
            doc,
            embeddings,
            magnitudes,
        };
        self.embeddings.insert(id, stored);
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
        let mut docs = BinaryHeap::new();
        let prompt = prompt_embedding.vec.as_slice();
        let prompt_magnitude = magnitude(prompt);

        for (id, stored) in &self.embeddings {
            // The document is serialized only for a filtered query.
            if let Some(filter) = filter
                && !filter.satisfies(&serde_json::to_value(&stored.doc)?)
            {
                continue;
            }
            // The best embedding's score, the same bits as the unnormalized
            // `VectorDistance::cosine_similarity`. Non-finite scores go first
            // so NaN cannot outrank valid embeddings or bypass the threshold.
            let Some(distance) = stored
                .embeddings
                .iter()
                .zip(&stored.magnitudes)
                .map(|(e, length)| OrderedFloat(dot(&e.vec, prompt) / (length * prompt_magnitude)))
                .filter(|distance| distance.0.is_finite())
                .max()
                .filter(|distance| !threshold.is_some_and(|t| distance.0 < t))
            else {
                continue;
            };

            docs.push(Reverse(RankingItem(distance, id, &stored.doc)));

            // Evict the worst score, or the greatest id at an equal score.
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
                    .map(|Reverse(RankingItem(distance, id, _))| format!("{id} ({distance})"))
                    .collect::<Vec<String>>()
                    .join(", ")
            );
        }

        Ok(docs)
    }

    /// Add documents and their corresponding embeddings to the store.
    /// IDs have the form `"doc{n}"`, starting at the current document count and
    /// skipping occupied IDs so existing documents are never overwritten.
    pub fn add_documents(&mut self, documents: impl IntoIterator<Item = (D, Vec<Embedding>)>) {
        let mut index = self.embeddings.len();
        for (doc, embeddings) in documents {
            let mut id = format!("doc{index}");
            while self.embeddings.contains_key(&id) {
                index += 1;
                id = format!("doc{index}");
            }
            self.insert(id, doc, embeddings);
            index += 1;
        }
    }

    /// Add documents and their corresponding embeddings to the store with ids.
    pub fn add_documents_with_ids(
        &mut self,
        documents: impl IntoIterator<Item = (impl ToString, D, Vec<Embedding>)>,
    ) {
        for (id, doc, embeddings) in documents {
            self.insert(id.to_string(), doc, embeddings);
        }
    }

    /// Adds documents, calling `f` once per document in input order.
    /// The callback may capture and mutate state. Duplicate IDs replace existing documents.
    pub fn add_documents_with_id_f(
        &mut self,
        documents: impl IntoIterator<Item = (D, Vec<Embedding>)>,
        mut f: impl FnMut(&D) -> String,
    ) {
        for (doc, embeddings) in documents {
            self.insert(f(&doc), doc, embeddings);
        }
    }
}

/// A score and document ID define the ranking key; the payload does not.
struct RankingItem<'a, D: Serialize>(OrderedFloat<f64>, &'a String, &'a D);

impl<D: Serialize> PartialEq for RankingItem<'_, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<D: Serialize> Eq for RankingItem<'_, D> {}

/// Orders results by descending score and ascending document ID for ties.
/// Explicit ordering keeps retrieval independent of hash-map iteration order.
fn ranked<D: Serialize>(docs: EmbeddingRanking<'_, D>) -> Vec<RankingItem<'_, D>> {
    let mut items: Vec<RankingItem<'_, D>> = docs.into_iter().map(|Reverse(item)| item).collect();
    items.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(right.1)));
    items
}

impl<D: Serialize> Ord for RankingItem<'_, D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0).then_with(|| other.1.cmp(self.1))
    }
}

impl<D: Serialize> PartialOrd for RankingItem<'_, D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

type EmbeddingRanking<'a, D> = BinaryHeap<Reverse<RankingItem<'a, D>>>;

impl<D: Serialize> InMemoryVectorStore<D> {
    pub fn index(self, model: impl Into<DynModel<EmbeddingOp>>) -> InMemoryVectorIndex<D> {
        InMemoryVectorIndex::new(model, self)
    }

    /// Every document id with its document and embeddings.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &D, &[Embedding])> {
        self.embeddings
            .iter()
            .map(|(id, stored)| (id, &stored.doc, stored.embeddings.as_slice()))
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

/// A vector store paired with a query embedding model. Stored vectors must use
/// the same embedding space and dimensions as query vectors. Results are ordered
/// by descending best finite cosine similarity, then ascending document ID.
pub struct InMemoryVectorIndex<D: Serialize> {
    model: DynModel<EmbeddingOp>,
    pub store: InMemoryVectorStore<D>,
}

impl<D: Serialize> InMemoryVectorIndex<D> {
    pub fn new(model: impl Into<DynModel<EmbeddingOp>>, store: InMemoryVectorStore<D>) -> Self {
        Self {
            model: model.into(),
            store,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &D, &[Embedding])> {
        self.store.iter()
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}

impl<D> VectorStoreIndex for InMemoryVectorIndex<D>
where
    D: Serialize + WasmCompatSend + WasmCompatSync,
{
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: DeserializeOwned>(
        &self,
        req: VectorSearchRequest,
    ) -> Result<Vec<VectorSearchResult<T>>, VectorStoreError> {
        let prompt_embedding = &self.model.embed_text(req.query()).await?;

        let docs = self.store.vector_search(
            prompt_embedding,
            req.samples() as usize,
            req.filter(),
            req.threshold(),
        )?;

        ranked(docs)
            .into_iter()
            .map(|RankingItem(distance, id, doc)| {
                Ok(VectorSearchResult {
                    score: distance.0,
                    id: id.clone(),
                    document: serde_json::from_value(serde_json::to_value(doc)?)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest,
    ) -> Result<Vec<VectorSearchIdResult>, VectorStoreError> {
        let prompt_embedding = &self.model.embed_text(req.query()).await?;

        let docs = self.store.vector_search(
            prompt_embedding,
            req.samples() as usize,
            req.filter(),
            req.threshold(),
        )?;

        Ok(ranked(docs)
            .into_iter()
            .map(|RankingItem(distance, id, _)| VectorSearchIdResult {
                score: distance.0,
                id: id.clone(),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests;
