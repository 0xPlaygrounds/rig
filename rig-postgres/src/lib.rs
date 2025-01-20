use futures::{Stream, StreamExt};
use rig::{
    embeddings::{self, Embedding, EmbeddingError, EmbeddingModel},
    vector_store::VectorStoreError,
    Embed, OneOrMany,
};
use serde::Serialize;
use sqlx::PgPool;
use uuid::Uuid;

pub struct PostgresVectorStore<Model: EmbeddingModel> {
    model: Model,
    pg_pool: PgPool,
    documents_table: String,
}

impl<Model: EmbeddingModel> PostgresVectorStore<Model> {
    pub fn new(model: Model, pg_pool: PgPool, documents_table: Option<String>) -> Self {
        Self {
            model,
            pg_pool,
            documents_table: documents_table.unwrap_or(String::from("documents")),
        }
    }

    pub async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        for (document, embeddings) in documents {
            let id = Uuid::new_v4();
            let json_document = serde_json::to_value(&document).unwrap();

            for embedding in embeddings {
                let embedding_text = embedding.document;
                let embedding: Vec<f64> = embedding.vec;

                sqlx::query(
                    format!(
                        "INSERT INTO {} (id, document, embedded_text, embedding) VALUES ($1, $2, $3, $4)",
                        self.documents_table
                    )
                    .as_str(),
                )
                .bind(id)
                .bind(&json_document)
                .bind(&embedding_text)
                .bind(&embedding)
                .execute(&self.pg_pool)
                .await
                .map_err(|e| VectorStoreError::DatastoreError(e.into()))?;
            }
        }

        Ok(())
    }
}
