use std::fmt::Display;

use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{VectorStoreError, VectorStoreIndex},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use surrealdb::{Connection, Surreal, sql::Thing};

pub use surrealdb::engine::local::Mem;
pub use surrealdb::engine::remote::ws::{Ws, Wss};

pub struct SurrealVectorStore<Model: EmbeddingModel, C: Connection> {
    model: Model,
    surreal: Surreal<C>,
    documents_table: String,
    distance_function: SurrealDistanceFunction,
}

/// SurrealDB supported distances
pub enum SurrealDistanceFunction {
    Knn,
    Hamming,
    Euclidean,
    Cosine,
    Jaccard,
}

impl Display for SurrealDistanceFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SurrealDistanceFunction::Cosine => write!(f, "vector::similarity::cosine"),
            SurrealDistanceFunction::Knn => write!(f, "vector::distance::knn"),
            SurrealDistanceFunction::Euclidean => write!(f, "vector::distance::euclidean"),
            SurrealDistanceFunction::Hamming => write!(f, "vector::distance::hamming"),
            SurrealDistanceFunction::Jaccard => write!(f, "vector::similarity::jaccard"),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SearchResult {
    id: Thing,
    document: String,
    distance: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateRecord {
    document: String,
    embedded_text: String,
    embedding: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct SearchResultOnlyId {
    id: Thing,
    distance: f64,
}

impl SearchResult {
    pub fn into_result<T: DeserializeOwned>(self) -> Result<(f64, String, T), VectorStoreError> {
        let document: T =
            serde_json::from_str(&self.document).map_err(VectorStoreError::JsonError)?;

        Ok((self.distance, self.id.id.to_string(), document))
    }
}

impl<Model: EmbeddingModel, C: Connection> SurrealVectorStore<Model, C> {
    pub fn new(
        model: Model,
        surreal: Surreal<C>,
        documents_table: Option<String>,
        distance_function: SurrealDistanceFunction,
    ) -> Self {
        Self {
            model,
            surreal,
            documents_table: documents_table.unwrap_or(String::from("documents")),
            distance_function,
        }
    }

    pub fn inner_client(&self) -> &Surreal<C> {
        &self.surreal
    }

    pub fn with_defaults(model: Model, surreal: Surreal<C>) -> Self {
        Self::new(model, surreal, None, SurrealDistanceFunction::Cosine)
    }

    fn search_query_full(&self) -> String {
        self.search_query(true)
    }

    fn search_query_only_ids(&self) -> String {
        self.search_query(false)
    }

    fn search_query(&self, with_document: bool) -> String {
        let document = if with_document { ", document" } else { "" };
        let embedded_text = if with_document { ", embedded_text" } else { "" };
        let Self {
            distance_function, ..
        } = self;
        format!(
            "
               SELECT id {document} {embedded_text}, {distance_function}($vec, embedding) as distance \
              from type::table($tablename) order by distance desc \
            LIMIT $limit",
        )
    }

    pub async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        for (document, embeddings) in documents {
            let json_document: serde_json::Value = serde_json::to_value(&document).unwrap();
            let json_document_as_string = serde_json::to_string(&json_document).unwrap();

            for embedding in embeddings {
                let embedded_text = embedding.document;
                let embedding: Vec<f64> = embedding.vec;

                let record = CreateRecord {
                    document: json_document_as_string.clone(),
                    embedded_text,
                    embedding,
                };

                self.surreal
                    .create::<Option<CreateRecord>>(self.documents_table.clone())
                    .content(record)
                    .await
                    .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
            }
        }

        Ok(())
    }
}

impl<Model: EmbeddingModel, C: Connection> VectorStoreIndex for SurrealVectorStore<Model, C> {
    /// Get the top n documents based on the distance to the given query.
    /// The result is a list of tuples of the form (score, id, document)
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let embedded_query: Vec<f64> = self.model.embed_text(query).await?.vec;

        let mut response = self
            .surreal
            .query(self.search_query_full().as_str())
            .bind(("vec", embedded_query))
            .bind(("tablename", self.documents_table.clone()))
            .bind(("limit", n))
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows: Vec<SearchResult> = response
            .take(0)
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
        let embedded_query: Vec<f32> = self
            .model
            .embed_text(query)
            .await?
            .vec
            .iter()
            .map(|&x| x as f32)
            .collect();

        let mut response = self
            .surreal
            .query(self.search_query_only_ids().as_str())
            .bind(("vec", embedded_query))
            .bind(("tablename", self.documents_table.clone()))
            .bind(("limit", n))
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows: Vec<(f64, String)> = response
            .take::<Vec<SearchResultOnlyId>>(0)
            .unwrap()
            .into_iter()
            .map(|row| (row.distance, row.id.id.to_string()))
            .collect();

        Ok(rows)
    }
}
