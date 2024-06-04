use anyhow::Result;
use futures::StreamExt;
use mongodb::bson::doc;

use rig::{
    embeddings::{DocumentEmbeddings, Embedding, EmbeddingModel},
    vector_store::{VectorStore, VectorStoreIndex}
};

pub struct MongoDbVectorStore {
    collection: mongodb::Collection<DocumentEmbeddings>,
}

impl VectorStore for MongoDbVectorStore {
    type Q = mongodb::bson::Document;

    async fn add_documents(&mut self, documents: Vec<DocumentEmbeddings>) -> Result<()> {
        self.collection.insert_many(documents, None).await?;
        Ok(())
    }

    async fn get_document_embeddings(&self, id: &str) -> Result<Option<DocumentEmbeddings>> {
        Ok(self.collection.find_one(doc! { "_id": id }, None).await?)
    }

    async fn get_document<T: for<'a> serde::Deserialize<'a>>(&self, id: &str) -> Result<Option<T>> {
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
            .await?
            .with_type::<String>()
            .next()
            .await
            .transpose()?
            .map(|doc| serde_json::from_str(&doc))
            .transpose()?)
    }

    async fn get_document_by_query(&self, query: Self::Q) -> Result<Option<DocumentEmbeddings>> {
        Ok(self.collection.find_one(query, None).await?)
    }
}

impl MongoDbVectorStore {
    pub fn new(collection: mongodb::Collection<DocumentEmbeddings>) -> Self {
        Self { collection }
    }

    pub fn index<M: EmbeddingModel>(
        &self,
        model: M,
        index_name: &str,
        filter: mongodb::bson::Document,
    ) -> Result<MongoDbVectorIndex<M>> {
        MongoDbVectorIndex::new(self.collection.clone(), model, index_name, filter)
    }
}

pub struct MongoDbVectorIndex<M: EmbeddingModel> {
    collection: mongodb::Collection<DocumentEmbeddings>,
    model: M,
    index_name: String,
    filter: mongodb::bson::Document,
}

impl<M: EmbeddingModel> MongoDbVectorIndex<M> {
    pub fn new(
        collection: mongodb::Collection<DocumentEmbeddings>,
        model: M,
        index_name: &str,
        filter: mongodb::bson::Document,
    ) -> Result<Self> {
        Ok(Self {
            collection,
            model,
            index_name: index_name.to_string(),
            filter,
        })
    }
}

impl<M: EmbeddingModel + std::marker::Sync + Send> VectorStoreIndex for MongoDbVectorIndex<M> {
    async fn embed_document(&self, document: &str) -> Result<Embedding> {
        self.model.embed_document(document).await
    }

    async fn top_n_from_query(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, DocumentEmbeddings)>> {
        let prompt_embedding = self.model.embed_document(query).await?;
        self.top_n_from_embedding(&prompt_embedding, n).await
    }

    async fn top_n_from_embedding(
        &self,
        prompt_embedding: &Embedding,
        n: usize,
    ) -> Result<Vec<(f64, DocumentEmbeddings)>> {
        let mut cursor = self
            .collection
            .aggregate(
                [
                    doc! {
                      "$vectorSearch": {
                        "index": &self.index_name,
                        "path": "embeddings.vec",
                        "queryVector": &prompt_embedding.vec,
                        "numCandidates": (n * 10) as u32,
                        "limit": n as u32,
                        "filter": &self.filter,
                      }
                    },
                    doc! {
                      "$addFields": {
                        "score": { "$meta": "vectorSearchScore" }
                      }
                    },
                ],
                None,
            )
            .await?
            .with_type::<serde_json::Value>();

        let mut results = Vec::new();
        while let Some(doc) = cursor.next().await {
            let doc = doc?;
            let score = doc.get("score").expect("score").as_f64().expect("f64");
            let document: DocumentEmbeddings = serde_json::from_value(doc).expect("document");
            results.push((score, document));
        }

        tracing::info!(target: "ai",
            "Selected documents: {}",
            results.iter()
                .map(|(distance, doc)| format!("{} ({})", doc.id, distance))
                .collect::<Vec<String>>()
                .join(", ")
        );

        Ok(results)
    }
}
