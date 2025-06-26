use std::fmt::Display;

use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{VectorStoreError, VectorStoreIndex},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

pub struct PostgresVectorStore<Model: EmbeddingModel> {
    model: Model,
    pg_pool: PgPool,
    documents_table: String,
    distance_function: PgVectorDistanceFunction,
}

/* PgVector supported distances
<-> - L2 distance
<#> - (negative) inner product
<=> - cosine distance
<+> - L1 distance (added in 0.7.0)
<~> - Hamming distance (binary vectors, added in 0.7.0)
<%> - Jaccard distance (binary vectors, added in 0.7.0)
 */
pub enum PgVectorDistanceFunction {
    L2,
    InnerProduct,
    Cosine,
    L1,
    Hamming,
    Jaccard,
}

impl Display for PgVectorDistanceFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PgVectorDistanceFunction::L2 => write!(f, "<->"),
            PgVectorDistanceFunction::InnerProduct => write!(f, "<#>"),
            PgVectorDistanceFunction::Cosine => write!(f, "<=>"),
            PgVectorDistanceFunction::L1 => write!(f, "<+>"),
            PgVectorDistanceFunction::Hamming => write!(f, "<~>"),
            PgVectorDistanceFunction::Jaccard => write!(f, "<%>"),
        }
    }
}

#[derive(Debug, Deserialize, sqlx::FromRow)]
pub struct SearchResult {
    id: Uuid,
    document: Value,
    //embedded_text: String,
    distance: f64,
}

#[derive(Debug, Deserialize, sqlx::FromRow)]
pub struct SearchResultOnlyId {
    id: Uuid,
    distance: f64,
}

impl SearchResult {
    pub fn into_result<T: DeserializeOwned>(self) -> Result<(f64, String, T), VectorStoreError> {
        let document: T =
            serde_json::from_value(self.document).map_err(VectorStoreError::JsonError)?;
        Ok((self.distance, self.id.to_string(), document))
    }
}

impl<Model: EmbeddingModel> PostgresVectorStore<Model> {
    pub fn new(
        model: Model,
        pg_pool: PgPool,
        documents_table: Option<String>,
        distance_function: PgVectorDistanceFunction,
    ) -> Self {
        Self {
            model,
            pg_pool,
            documents_table: documents_table.unwrap_or(String::from("documents")),
            distance_function,
        }
    }

    pub fn with_defaults(model: Model, pg_pool: PgPool) -> Self {
        Self::new(model, pg_pool, None, PgVectorDistanceFunction::Cosine)
    }

    fn search_query_full(&self) -> String {
        self.search_query(true)
    }
    fn search_query_only_ids(&self) -> String {
        self.search_query(false)
    }

    fn search_query(&self, with_document: bool) -> String {
        let document = if with_document { ", document" } else { "" };
        format!(
            "
            SELECT id{}, distance FROM ( \
              SELECT DISTINCT ON (id) id{}, embedding {} $1 as distance \
              FROM {} \
              ORDER BY id, distance \
            ) as d \
            ORDER BY distance \
            LIMIT $2",
            document, document, self.distance_function, self.documents_table
        )
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

impl<Model: EmbeddingModel> VectorStoreIndex for PostgresVectorStore<Model> {
    /// Get the top n documents based on the distance to the given query.
    /// The result is a list of tuples of the form (score, id, document)
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let embedded_query: pgvector::Vector = self
            .model
            .embed_text(query)
            .await?
            .vec
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<f32>>()
            .into();

        let rows: Vec<SearchResult> = sqlx::query_as(self.search_query_full().as_str())
            .bind(embedded_query)
            .bind(n as i64)
            .fetch_all(&self.pg_pool)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows: Vec<(f64, String, T)> = rows
            .into_iter()
            .flat_map(SearchResult::into_result)
            .collect();

        Ok(rows)
    }

    /// Same as `top_n` but returns the document ids only.
    async fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let embedded_query: pgvector::Vector = self
            .model
            .embed_text(query)
            .await?
            .vec
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<f32>>()
            .into();

        let rows: Vec<SearchResultOnlyId> = sqlx::query_as(self.search_query_only_ids().as_str())
            .bind(embedded_query)
            .bind(n as i64)
            .fetch_all(&self.pg_pool)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows: Vec<(f64, String)> = rows
            .into_iter()
            .map(|row| (row.distance, row.id.to_string()))
            .collect();

        Ok(rows)
    }
}
