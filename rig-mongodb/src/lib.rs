use futures::StreamExt;
use mongodb::bson::{self, doc};

use rig::{
    embeddings::{DocumentEmbeddings, Embedding, EmbeddingModel},
    vector_store::{VectorStore, VectorStoreError, VectorStoreIndex},
};
use serde::Deserialize;

/// A MongoDB vector store.
pub struct MongoDbVectorStore {
    collection: mongodb::Collection<DocumentEmbeddings>,
}

fn mongodb_to_rig_error(e: mongodb::error::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

impl VectorStore for MongoDbVectorStore {
    type Q = mongodb::bson::Document;

    async fn add_documents(
        &mut self,
        documents: Vec<DocumentEmbeddings>,
    ) -> Result<(), VectorStoreError> {
        self.collection
            .insert_many(documents, None)
            .await
            .map_err(mongodb_to_rig_error)?;
        Ok(())
    }

    async fn get_document_embeddings(
        &self,
        id: &str,
    ) -> Result<Option<DocumentEmbeddings>, VectorStoreError> {
        self.collection
            .find_one(doc! { "_id": id }, None)
            .await
            .map_err(mongodb_to_rig_error)
    }

    async fn get_document<T: for<'a> serde::Deserialize<'a>>(
        &self,
        id: &str,
    ) -> Result<Option<T>, VectorStoreError> {
        Ok(self
            .collection
            .clone_with_type::<String>()
            .aggregate(
                [
                    doc! {"$match": { "_id": id}},
                    doc! {"$project": { "document": 1 }},
                    doc! {"$replaceRoot": { "newRoot": "$document" }},
                ],
                None,
            )
            .await
            .map_err(mongodb_to_rig_error)?
            .with_type::<String>()
            .next()
            .await
            .transpose()
            .map_err(mongodb_to_rig_error)?
            .map(|doc| serde_json::from_str(&doc))
            .transpose()?)
    }

    async fn get_document_by_query(
        &self,
        query: Self::Q,
    ) -> Result<Option<DocumentEmbeddings>, VectorStoreError> {
        self.collection
            .find_one(query, None)
            .await
            .map_err(mongodb_to_rig_error)
    }
}

impl MongoDbVectorStore {
    /// Create a new `MongoDbVectorStore` from a MongoDB collection.
    pub fn new(collection: mongodb::Collection<DocumentEmbeddings>) -> Self {
        Self { collection }
    }

    /// Create a new `MongoDbVectorIndex` from an existing `MongoDbVectorStore`.
    ///
    /// The index (of type "vector") must already exist for the MongoDB collection.
    /// See the MongoDB [documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/) for more information on creating indexes.
    ///
    /// An additional filter can be provided to further restrict the documents that are
    /// considered in the search.
    pub fn index<M: EmbeddingModel>(
        &self,
        model: M,
        index_name: &str,
        filter: mongodb::bson::Document,
    ) -> MongoDbVectorIndex<M> {
        MongoDbVectorIndex::new(self.collection.clone(), model, index_name, filter)
    }
}

/// A vector index for a MongoDB collection.
pub struct MongoDbVectorIndex<M: EmbeddingModel> {
    collection: mongodb::Collection<DocumentEmbeddings>,
    model: M,
    index_name: String,
    filter: mongodb::bson::Document,
}

impl<M: EmbeddingModel> MongoDbVectorIndex<M> {
    /// Vector search stage of aggregation pipeline of mongoDB collection.
    /// To be used by implementations of top_n and top_n_ids methods on VectorStoreIndex trait for MongoDbVectorIndex.
    fn pipeline_search_stage(&self, prompt_embedding: &Embedding, n: usize) -> bson::Document {
        doc! {
          "$vectorSearch": {
            "index": &self.index_name,
            "path": "embeddings.vec",
            "queryVector": &prompt_embedding.vec,
            "numCandidates": (n * 10) as u32,
            "limit": n as u32,
            "filter": &self.filter,
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

impl<M: EmbeddingModel> MongoDbVectorIndex<M> {
    pub fn new(
        collection: mongodb::Collection<DocumentEmbeddings>,
        model: M,
        index_name: &str,
        filter: mongodb::bson::Document,
    ) -> Self {
        Self {
            collection,
            model,
            index_name: index_name.to_string(),
            filter,
        }
    }
}

impl<M: EmbeddingModel + std::marker::Sync + Send> VectorStoreIndex for MongoDbVectorIndex<M> {
    async fn top_n<T: for<'a> Deserialize<'a> + std::marker::Send>(
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
