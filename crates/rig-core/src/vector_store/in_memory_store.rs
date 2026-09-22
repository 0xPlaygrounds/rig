//! In-memory vector storage with cosine scoring and optional approximate candidates.
//!
//! ```
//! use rig_core::vector_store::in_memory_store::InMemoryVectorStore;
//!
//! let store = InMemoryVectorStore::<String>::builder().build();
//! assert!(store.is_empty());
//! ```
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};

use ordered_float::OrderedFloat;
use serde::{Serialize, de::DeserializeOwned};

use super::{IndexStrategy, VectorStoreError, VectorStoreIndex, request::VectorSearchRequest};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
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
    /// Creates an empty builder with brute-force search as the default strategy.
    pub fn builder() -> InMemoryVectorStoreBuilder<D> {
        InMemoryVectorStoreBuilder::new()
    }

    /// Internal constructor used by the builder.
    pub(super) fn from_builder(
        embeddings: HashMap<String, (D, Vec<Embedding>)>,
        index_strategy: IndexStrategy,
    ) -> Self {
        let lsh_params = match &index_strategy {
            IndexStrategy::LSH {
                num_tables,
                num_hyperplanes,
            } => Some((*num_tables, *num_hyperplanes)),
            IndexStrategy::BruteForce => None,
        };

        let mut vector_store = Self {
            embeddings,
            index_strategy,
            lsh_index: None,
        };

        if let Some((num_tables, num_hyperplanes)) = lsh_params {
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

    /// Inserts or replaces a document and adds its embeddings to any existing LSH index.
    fn insert_document(&mut self, id: String, doc: D, embeddings: Vec<Embedding>) {
        if let Some(ref mut lsh_index) = self.lsh_index {
            for embedding in embeddings.iter() {
                lsh_index.insert(&id, &embedding.vec);
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

        // Filter non-finite scores before selecting the maximum so NaN cannot
        // outrank valid embeddings or bypass the threshold.
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

        let first_embedding = self
            .embeddings
            .values()
            .next()
            .and_then(|(_, embeddings)| embeddings.iter().next())
            .map_or(0, |e| e.vec.len());

        if first_embedding == 0 {
            return;
        }

        let mut lsh_index = LSHIndex::new(first_embedding, num_tables, num_hyperplanes);

        for (id, (_, embeddings)) in self.embeddings.iter() {
            for embedding in embeddings.iter() {
                lsh_index.insert(id, &embedding.vec);
            }
        }

        self.lsh_index = Some(lsh_index);
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
            self.insert_document(id, doc, embeddings);
            index += 1;
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

/// Orders results by descending score and ascending document ID for ties.
/// Explicit ordering keeps retrieval independent of hash-map iteration order.
fn ranked<D: Serialize + Eq>(docs: EmbeddingRanking<'_, D>) -> Vec<RankingItem<'_, D>> {
    let mut items: Vec<RankingItem<'_, D>> = docs.into_iter().map(|Reverse(item)| item).collect();
    items.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(right.1)));
    items
}

impl<D: Serialize + Eq> Ord for RankingItem<'_, D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0).then_with(|| other.1.cmp(self.1))
    }
}

impl<D: Serialize + Eq> PartialOrd for RankingItem<'_, D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

type EmbeddingRanking<'a, D> = BinaryHeap<Reverse<RankingItem<'a, D>>>;

impl<D: Serialize> InMemoryVectorStore<D> {
    pub fn index<M: EmbeddingModel>(self, model: M) -> InMemoryVectorIndex<D, M> {
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

/// A vector store paired with a query embedding model. Stored vectors must use
/// the same embedding space and dimensions as query vectors. Results are ordered
/// by descending best finite cosine similarity, then ascending document ID.
pub struct InMemoryVectorIndex<D: Serialize, M> {
    model: M,
    pub store: InMemoryVectorStore<D>,
}

impl<D: Serialize, M> InMemoryVectorIndex<D, M> {
    pub fn new(model: M, store: InMemoryVectorStore<D>) -> Self {
        Self { model, store }
    }

    /// The embedding model used for queries.
    pub fn model(&self) -> &M {
        &self.model
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

impl<D: Serialize + WasmCompatSend + WasmCompatSync + Eq, M: EmbeddingModel> VectorStoreIndex
    for InMemoryVectorIndex<D, M>
{
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: DeserializeOwned>(
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

        ranked(docs)
            .into_iter()
            .map(|RankingItem(distance, id, doc, _)| {
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

        ranked(docs)
            .into_iter()
            .map(|RankingItem(distance, id, _, _)| Ok((distance.0, id.clone())))
            .collect::<Result<Vec<_>, _>>()
    }
}

#[cfg(test)]
mod tests;
