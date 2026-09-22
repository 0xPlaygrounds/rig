//! Qdrant vector store for Rig.
//!
//! [`QdrantVectorStore`] runs dense vector search against a Qdrant collection,
//! optionally narrowed by a [`QdrantFilter`]. The `rig` facade re-exports this
//! crate as `rig::qdrant` under the `qdrant` feature.

mod filter;

pub use filter::QdrantFilter;
use qdrant_client::{
    Payload, Qdrant,
    qdrant::{
        Filter, PointId, PointStruct, Query, QueryPoints, UpsertPointsBuilder,
        point_id::PointIdOptions,
    },
};
use rig_core::{
    Embed,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex, request::VectorSearchRequest,
    },
    wasm_compat::WasmCompatSend,
};
use serde::{Serialize, de::DeserializeOwned};
use uuid::Uuid;

/// Vector store backed by a Qdrant collection.
///
/// Queries are embedded with the same model `M` that populated the collection,
/// so results are meaningless under another model.
pub struct QdrantVectorStore<M> {
    model: M,
    client: Qdrant,
    query_params: QueryPoints,
}

impl<M: EmbeddingModel> QdrantVectorStore<M> {
    /// Creates a store over the collection named by `query_params`. Each search
    /// clones `query_params` and overrides its query, limit, threshold, and filter.
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

    /// Embeds the query and narrows each component to `f32` for Qdrant.
    async fn generate_query_vector(&self, query: &str) -> Result<Vec<f32>, VectorStoreError> {
        let embedding = self.model.embed_text(query).await?;
        Ok(embedding.vec.iter().map(|&x| x as f32).collect())
    }

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

    /// Runs the search and returns scored points. A query preset in
    /// `query_params` takes precedence over embedding the request text.
    async fn run_query(
        &self,
        req: &VectorSearchRequest<QdrantFilter>,
    ) -> Result<Vec<qdrant_client::qdrant::ScoredPoint>, VectorStoreError> {
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

        Ok(self
            .client
            .query(params)
            .await
            .map_err(VectorStoreError::datastore)?
            .result)
    }
}

impl<M: EmbeddingModel> InsertDocuments for QdrantVectorStore<M> {
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let collection_name = self.query_params.collection_name.clone();

        for (document, embeddings) in documents {
            let json_document = serde_json::to_value(&document)?;
            let doc_as_payload =
                Payload::try_from(json_document).map_err(VectorStoreError::datastore)?;

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

            let request =
                UpsertPointsBuilder::new(&collection_name, embeddings_as_point_structs).wait(true);
            self.client
                .upsert_points(request)
                .await
                .map_err(VectorStoreError::datastore)?;
        }

        Ok(())
    }
}

/// Renders a point id as a string, erroring when the id is absent.
fn stringify_id(id: PointId) -> Result<String, VectorStoreError> {
    match id.point_id_options {
        Some(PointIdOptions::Num(num)) => Ok(num.to_string()),
        Some(PointIdOptions::Uuid(uuid)) => Ok(uuid),
        None => Err(VectorStoreError::MissingIdError(
            "Qdrant point carries no id".to_string(),
        )),
    }
}

fn missing_point_id() -> VectorStoreError {
    VectorStoreError::MissingIdError("Qdrant search result carries no point id".to_string())
}

impl<M: EmbeddingModel> VectorStoreIndex for QdrantVectorStore<M> {
    type Filter = QdrantFilter;

    /// Returns the nearest points as `(score, id, payload)`. Errors when a point
    /// lacks an id or its payload does not deserialize into `T`.
    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.run_query(&req)
            .await?
            .into_iter()
            .map(|item| {
                let id = stringify_id(item.id.ok_or_else(missing_point_id)?)?;
                let score = item.score as f64;
                let payload = serde_json::from_value(serde_json::to_value(item.payload)?)?;
                Ok((score, id, payload))
            })
            .collect()
    }

    /// Like `top_n` but returns `(score, id)` without deserializing payloads.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        self.run_query(&req)
            .await?
            .into_iter()
            .map(|point| {
                let id = stringify_id(point.id.ok_or_else(missing_point_id)?)?;
                Ok((point.score as f64, id))
            })
            .collect()
    }
}
