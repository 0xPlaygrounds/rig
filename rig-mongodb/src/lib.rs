use futures::StreamExt;
use mongodb::bson::{self, doc};

use rig::{
    embeddings::embedding::{Embedding, EmbeddingModel},
    vector_store::{VectorStoreError, VectorStoreIndex},
};
use serde::Deserialize;

/// A MongoDB vector store.
pub struct MongoDbVectorStore<C> {
    collection: mongodb::Collection<C>,
}

fn mongodb_to_rig_error(e: mongodb::error::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

impl<C> MongoDbVectorStore<C> {
    /// Create a new `MongoDbVectorStore` from a MongoDB collection.
    pub fn new(collection: mongodb::Collection<C>) -> Self {
        Self { collection }
    }

    /// Create a new `MongoDbVectorIndex` from an existing `MongoDbVectorStore`.
    ///
    /// The index (of type "vector") must already exist for the MongoDB collection.
    /// See the MongoDB [documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/) for more information on creating indexes.
    pub fn index<M: EmbeddingModel>(
        &self,
        model: M,
        index_name: &str,
        search_params: SearchParams,
    ) -> MongoDbVectorIndex<M, C> {
        MongoDbVectorIndex::new(self.collection.clone(), model, index_name, search_params)
    }
}

/// A vector index for a MongoDB collection.
pub struct MongoDbVectorIndex<M: EmbeddingModel, C> {
    collection: mongodb::Collection<C>,
    model: M,
    index_name: String,
    search_params: SearchParams,
}

impl<M: EmbeddingModel, C> MongoDbVectorIndex<M, C> {
    /// Vector search stage of aggregation pipeline of mongoDB collection.
    /// To be used by implementations of top_n and top_n_ids methods on VectorStoreIndex trait for MongoDbVectorIndex.
    fn pipeline_search_stage(&self, prompt_embedding: &Embedding, n: usize) -> bson::Document {
        let SearchParams {
            filter,
            exact,
            num_candidates,
            path,
        } = &self.search_params;

        doc! {
          "$vectorSearch": {
            "index": &self.index_name,
            "path": path,
            "queryVector": &prompt_embedding.vec,
            "numCandidates": num_candidates.unwrap_or((n * 10) as u32),
            "limit": n as u32,
            "filter": filter,
            "exact": exact.unwrap_or(false)
          }
        }
    }

    /// Score declaration stage of aggregation pipeline of mongoDB collection.
    /// /// To be used by implementations of top_n and top_n_ids methods on VectorStoreIndex trait for MongoDbVectorIndex.
    fn pipeline_score_stage(&self) -> bson::Document {
        doc! {
          "$addFields": {
            "score": { "$meta": "vectorSearchScore" }
          }
        }
    }
}

impl<M: EmbeddingModel, C> MongoDbVectorIndex<M, C> {
    pub fn new(
        collection: mongodb::Collection<C>,
        model: M,
        index_name: &str,
        search_params: SearchParams,
    ) -> Self {
        Self {
            collection,
            model,
            index_name: index_name.to_string(),
            search_params,
        }
    }
}

/// See [MongoDB Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/) for more information
/// on each of the fields
pub struct SearchParams {
    filter: mongodb::bson::Document,
    path: String,
    exact: Option<bool>,
    num_candidates: Option<u32>,
}

impl SearchParams {
    /// Initializes a new `SearchParams` with default values.
    pub fn new(path: &str) -> Self {
        Self {
            filter: doc! {},
            exact: None,
            num_candidates: None,
            path: path.to_string(),
        }
    }

    /// Sets the pre-filter field of the search params.
    /// See [MongoDB vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/) for more information.
    pub fn filter(mut self, filter: mongodb::bson::Document) -> Self {
        self.filter = filter;
        self
    }

    /// Sets the exact field of the search params.
    /// If exact is true, an ENN vector search will be performed, otherwise, an ANN search will be performed.
    /// By default, exact is false.
    /// See [MongoDB vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/) for more information.
    pub fn exact(mut self, exact: bool) -> Self {
        self.exact = Some(exact);
        self
    }

    /// Sets the num_candidates field of the search params.
    /// Only set this field if exact is set to false.
    /// Number of nearest neighbors to use during the search.
    /// See [MongoDB vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/) for more information.
    pub fn num_candidates(mut self, num_candidates: u32) -> Self {
        self.num_candidates = Some(num_candidates);
        self
    }
}

impl<M: EmbeddingModel + Sync + Send, C: Sync + Send> VectorStoreIndex
    for MongoDbVectorIndex<M, C>
{
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_document(query).await?;

        let mut cursor = self
            .collection
            .aggregate(
                [
                    self.pipeline_search_stage(&prompt_embedding, n),
                    self.pipeline_score_stage(),
                ],
                None,
            )
            .await
            .map_err(mongodb_to_rig_error)?
            .with_type::<serde_json::Value>();

        let mut results = Vec::new();
        while let Some(doc) = cursor.next().await {
            let doc = doc.map_err(mongodb_to_rig_error)?;
            let score = doc.get("score").expect("score").as_f64().expect("f64");
            let id = doc.get("_id").expect("_id").to_string();
            let doc_t: T = serde_json::from_value(doc).map_err(VectorStoreError::JsonError)?;
            results.push((score, id, doc_t));
        }

        tracing::info!(target: "rig",
            "Selected documents: {}",
            results.iter()
                .map(|(distance, id, _)| format!("{} ({})", id, distance))
                .collect::<Vec<String>>()
                .join(", ")
        );

        Ok(results)
    }

    async fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_document(query).await?;

        let mut cursor = self
            .collection
            .aggregate(
                [
                    self.pipeline_search_stage(&prompt_embedding, n),
                    self.pipeline_score_stage(),
                    doc! {
                        "$project": {
                            "_id": 1,
                            "score": 1
                        },
                    },
                ],
                None,
            )
            .await
            .map_err(mongodb_to_rig_error)?
            .with_type::<serde_json::Value>();

        let mut results = Vec::new();
        while let Some(doc) = cursor.next().await {
            let doc = doc.map_err(mongodb_to_rig_error)?;
            let score = doc.get("score").expect("score").as_f64().expect("f64");
            let id = doc.get("_id").expect("_id").to_string();
            results.push((score, id));
        }

        tracing::info!(target: "rig",
            "Selected documents: {}",
            results.iter()
                .map(|(distance, id)| format!("{} ({})", id, distance))
                .collect::<Vec<String>>()
                .join(", ")
        );

        Ok(results)
    }
}
