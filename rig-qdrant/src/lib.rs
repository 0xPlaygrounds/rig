mod filter;

use filter::*;
use qdrant_client::{
    Payload, Qdrant,
    qdrant::{
        Filter, PointId, PointStruct, Query, QueryPoints, UpsertPointsBuilder,
        point_id::PointIdOptions,
    },
};
use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex, request::VectorSearchRequest,
    },
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Represents a vector store implementation using Qdrant - <https://qdrant.tech/> as the backend.
pub struct QdrantVectorStore<M: EmbeddingModel> {
    /// Model used to generate embeddings for the vector store
    model: M,
    /// Client instance for Qdrant server communication
    client: Qdrant,
    /// Default search parameters
    query_params: QueryPoints,
}

impl<M> QdrantVectorStore<M>
where
    M: EmbeddingModel,
{
    /// Creates a new instance of `QdrantVectorStore`.
    ///
    /// # Arguments
    /// * `client` - Qdrant client instance
    /// * `model` - Embedding model instance
    /// * `query_params` - Search parameters for vector queries
    ///   Reference: <https://api.qdrant.tech/v-1-12-x/api-reference/search/query-points>
    pub fn new(client: Qdrant, model: M, query_params: QueryPoints) -> Self {
        Self {
            client,
            model,
            query_params,
        }
    }

    pub fn client(&self) -> &Qdrant {
        &self.client
    }

    /// Embed query based on `QdrantVectorStore` model and modify the vector in the required format.
    async fn generate_query_vector(&self, query: &str) -> Result<Vec<f32>, VectorStoreError> {
        let embedding = self.model.embed_text(query).await?;
        Ok(embedding.vec.iter().map(|&x| x as f32).collect())
    }

    /// Fill in query parameters with the given query and limit.
    fn prepare_query_params(
        &self,
        query: Option<Query>,
        limit: usize,
        threshold: Option<f64>,
        filter: Option<Filter>,
    ) -> QueryPoints {
        let mut params = self.query_params.clone();
        params.query = query;
        params.limit = Some(limit as u64);
        params.score_threshold = threshold.map(|x| x as f32);
        params.filter = filter;
        params
    }
}

impl<Model> InsertDocuments for QdrantVectorStore<Model>
where
    Model: EmbeddingModel + Send + Sync,
{
    async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let collection_name = self.query_params.collection_name.clone();

        for (document, embeddings) in documents {
            let json_document = serde_json::to_value(&document).unwrap();
            let doc_as_payload = Payload::try_from(json_document).unwrap();

            let embeddings_as_point_structs = embeddings
                .into_iter()
                .map(|embedding| {
                    let embedding_as_f32: Vec<f32> =
                        embedding.vec.into_iter().map(|x| x as f32).collect();
                    PointStruct::new(
                        Uuid::new_v4().to_string(),
                        embedding_as_f32,
                        doc_as_payload.clone(),
                    )
                })
                .collect::<Vec<PointStruct>>();

            let request = UpsertPointsBuilder::new(&collection_name, embeddings_as_point_structs);
            self.client.upsert_points(request).await.map_err(|err| {
                VectorStoreError::DatastoreError(format!("Error while upserting: {err}").into())
            })?;
        }

        Ok(())
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

impl<M> VectorStoreIndex for QdrantVectorStore<M>
where
    M: EmbeddingModel + std::marker::Sync + Send,
{
    type Filter = QdrantFilter;

    /// Search for the top `n` nearest neighbors to the given query within the Qdrant vector store.
    /// Returns a vector of tuples containing the score, ID, and payload of the nearest neighbors.
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let query = match self.query_params.query {
            Some(ref q) => Some(q.clone()),
            None => Some(Query::new_nearest(
                self.generate_query_vector(req.query()).await?,
            )),
        };

        let filter = req
            .filter()
            .as_ref()
            .cloned()
            .map(QdrantFilter::interpret)
            .transpose()?
            .flatten();

        let params =
            self.prepare_query_params(query, req.samples() as usize, req.threshold(), filter);

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
                let payload = serde_json::from_value(serde_json::to_value(item.payload)?)?;
                Ok((score, id, payload))
            })
            .collect()
    }

    /// Search for the top `n` nearest neighbors to the given query within the Qdrant vector store.
    /// Returns a vector of tuples containing the score and ID of the nearest neighbors.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let query = match self.query_params.query {
            Some(ref q) => Some(q.clone()),
            None => Some(Query::new_nearest(
                self.generate_query_vector(req.query()).await?,
            )),
        };

        let filter = req
            .filter()
            .as_ref()
            .cloned()
            .map(QdrantFilter::interpret)
            .transpose()?
            .flatten();

        let params =
            self.prepare_query_params(query, req.samples() as usize, req.threshold(), filter);

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
}
