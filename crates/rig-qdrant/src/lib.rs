//! Qdrant vector store integration for Rig.
//!
//! This crate provides [`QdrantVectorStore`], a Rig vector store backed by
//! Qdrant collections. It supports dense vector search and Qdrant filter
//! expressions through [`QdrantFilter`].
//!
//! Queries arrive pre-embedded via
//! [`VectorSearchRequest`](rig_core::vector_store::request::VectorSearchRequest);
//! the store never embeds text itself. There is no store trait: the store
//! exposes concrete inherent async methods (`top_n`, `top_n_ids`, `top_n_as`,
//! `insert`, `insert_as`).
//!
//! The root `rig` facade re-exports this crate as `rig::qdrant` when the
//! `qdrant` feature is enabled.

mod filter;

pub use filter::QdrantFilter;
use qdrant_client::{
    Payload, Qdrant,
    qdrant::{
        PointId, PointStruct, Query, QueryPoints, UpsertPointsBuilder, point_id::PointIdOptions,
    },
};
use rig_core::vector_store::{
    SearchHit, StoreRecord, VectorStoreError, request::VectorSearchRequest,
};
use serde::{Serialize, de::DeserializeOwned};
use uuid::Uuid;

/// Represents a vector store implementation using Qdrant - <https://qdrant.tech/> as the backend.
///
/// Queries and records arrive pre-embedded, so the store holds no embedding
/// model: it only stores and searches vectors.
pub struct QdrantVectorStore {
    /// Client instance for Qdrant server communication
    client: Qdrant,
    /// Default search parameters
    query_params: QueryPoints,
}

impl QdrantVectorStore {
    /// Creates a new instance of `QdrantVectorStore`.
    ///
    /// # Arguments
    /// * `client` - Qdrant client instance
    /// * `query_params` - Search parameters for vector queries
    ///   Reference: <https://api.qdrant.tech/v-1-12-x/api-reference/search/query-points>
    pub fn new(client: Qdrant, query_params: QueryPoints) -> Self {
        Self {
            client,
            query_params,
        }
    }

    pub fn client(&self) -> &Qdrant {
        &self.client
    }

    /// Fill in query parameters from the pre-embedded search request.
    ///
    /// A `query` preconfigured on the store's [`QueryPoints`] takes precedence;
    /// otherwise the first query embedding of the request is used (Qdrant
    /// queries a single vector).
    fn prepare_query_params(
        &self,
        req: &VectorSearchRequest<QdrantFilter>,
    ) -> Result<QueryPoints, VectorStoreError> {
        let query = match self.query_params.query {
            Some(ref q) => q.clone(),
            None => {
                let vec_f32: Vec<f32> = req.query().first().vec.iter().map(|&x| x as f32).collect();
                Query::new_nearest(vec_f32)
            }
        };

        let filter = req
            .filter()
            .as_ref()
            .cloned()
            .map(QdrantFilter::interpret)
            .transpose()?
            .flatten();

        let mut params = self.query_params.clone();
        params.query = Some(query);
        params.limit = Some(req.samples());
        params.score_threshold = req.threshold().map(|x| x as f32);
        params.filter = filter;
        Ok(params)
    }

    /// Insert precomputed records into the Qdrant collection configured on the
    /// store's [`QueryPoints`].
    ///
    /// Each embedding of a record becomes one Qdrant point carrying the
    /// record's payload. A record with a single embedding keeps its `id` as the
    /// point id (Qdrant requires ids to be UUIDs or unsigned integers); records
    /// with multiple embeddings get a freshly generated UUID per point, since a
    /// Qdrant point holds exactly one vector.
    pub async fn insert(&self, records: Vec<StoreRecord>) -> Result<(), VectorStoreError> {
        let collection_name = self.query_params.collection_name.clone();

        for record in records {
            let doc_as_payload = Payload::try_from(record.payload)
                .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

            let single = record.embeddings.len() == 1;
            let embeddings_as_point_structs = record
                .embeddings
                .into_iter()
                .map(|embedding| {
                    let embedding_as_f32: Vec<f32> =
                        embedding.vec.into_iter().map(|x| x as f32).collect();
                    let id = if single {
                        record.id.clone()
                    } else {
                        Uuid::new_v4().to_string()
                    };
                    PointStruct::new(id, embedding_as_f32, doc_as_payload.clone())
                })
                .collect::<Vec<PointStruct>>();

            let request =
                UpsertPointsBuilder::new(&collection_name, embeddings_as_point_structs).wait(true);
            self.client.upsert_points(request).await.map_err(|err| {
                VectorStoreError::DatastoreError(format!("Error while upserting: {err}").into())
            })?;
        }

        Ok(())
    }

    /// Serializes each document and inserts it. Sugar over [`Self::insert`].
    pub async fn insert_as<T: Serialize>(
        &self,
        docs: Vec<(
            String,
            T,
            rig_core::OneOrMany<rig_core::embeddings::Embedding>,
        )>,
    ) -> Result<(), VectorStoreError> {
        let records = docs
            .into_iter()
            .map(|(id, doc, embeddings)| StoreRecord::new(id, &doc, embeddings))
            .collect::<Result<Vec<_>, _>>()?;
        self.insert(records).await
    }

    /// Search for the top `n` nearest neighbors to the given pre-embedded query
    /// within the Qdrant vector store.
    ///
    /// Scores are Qdrant similarity scores: higher is better. Only the first
    /// query embedding is used (unless a `query` was preconfigured on the
    /// store's [`QueryPoints`]). Results are sorted by descending score.
    pub async fn top_n(
        &self,
        req: VectorSearchRequest<QdrantFilter>,
    ) -> Result<Vec<SearchHit>, VectorStoreError> {
        let params = self.prepare_query_params(&req)?;

        let result = self
            .client
            .query(params)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        result
            .result
            .into_iter()
            .map(|item| {
                let id =
                    stringify_id(item.id.ok_or_else(|| {
                        VectorStoreError::DatastoreError("Missing point ID".into())
                    })?)?;
                let score = item.score as f64;
                let payload = serde_json::to_value(item.payload)?;
                Ok(SearchHit { id, score, payload })
            })
            .collect()
    }

    /// Search for the top `n` nearest neighbors to the given pre-embedded query
    /// within the Qdrant vector store.
    /// Returns a vector of `(score, id)` tuples sorted by descending score.
    pub async fn top_n_ids(
        &self,
        req: VectorSearchRequest<QdrantFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let params = self.prepare_query_params(&req)?;

        let points = self
            .client
            .query(params)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?
            .result;

        points
            .into_iter()
            .map(|point| {
                let id =
                    stringify_id(point.id.ok_or_else(|| {
                        VectorStoreError::DatastoreError("Missing point ID".into())
                    })?)?;
                Ok((point.score as f64, id))
            })
            .collect()
    }

    /// Search for the top `n` nearest neighbors and deserialize each payload
    /// into `T`, as `(score, id, document)` tuples. Sugar over [`Self::top_n`].
    pub async fn top_n_as<T: DeserializeOwned>(
        &self,
        req: VectorSearchRequest<QdrantFilter>,
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
}

/// Converts a `PointId` to its string representation.
fn stringify_id(id: PointId) -> Result<String, VectorStoreError> {
    match id.point_id_options {
        Some(PointIdOptions::Num(num)) => Ok(num.to_string()),
        Some(PointIdOptions::Uuid(uuid)) => Ok(uuid.to_string()),
        None => Err(VectorStoreError::DatastoreError(
            "Invalid point ID format".into(),
        )),
    }
}
