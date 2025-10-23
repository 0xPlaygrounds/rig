use std::fmt::Display;

use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{SearchFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use surrealdb::{Connection, Surreal, sql::Thing};

pub use surrealdb::engine::local::Mem;
pub use surrealdb::engine::remote::ws::{Ws, Wss};

pub struct SurrealVectorStore<C, Model>
where
    C: Connection,
    Model: EmbeddingModel,
{
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

impl<C, Model> InsertDocuments for SurrealVectorStore<C, Model>
where
    C: Connection + Send + Sync,
    Model: EmbeddingModel + Send + Sync,
{
    async fn insert_documents<Doc: Serialize + Embed + Send>(
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealSearchFilter(String);

impl SurrealSearchFilter {
    fn inner(self) -> String {
        self.0
    }
}

impl std::fmt::Display for SurrealSearchFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl SearchFilter for SurrealSearchFilter {
    type Value = surrealdb::Value;

    fn eq(key: String, value: Self::Value) -> Self {
        Self(format!("{key} = {value}"))
    }

    fn gt(key: String, value: Self::Value) -> Self {
        Self(format!("{key} > {value}"))
    }

    fn lt(key: String, value: Self::Value) -> Self {
        Self(format!("{key} < {value}"))
    }

    fn and(self, rhs: Self) -> Self {
        Self(format!("({self}) AND ({rhs})"))
    }

    fn or(self, rhs: Self) -> Self {
        Self(format!("({self}) OR ({rhs})"))
    }
}

impl SurrealSearchFilter {
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(format!("NOT ({self})"))
    }
}

impl<C, Model> SurrealVectorStore<C, Model>
where
    C: Connection,
    Model: EmbeddingModel,
{
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
              from type::table($tablename) \
              where {distance_function}($vec, embedding) >= $threshold AND $filter \
              order by distance desc \
            LIMIT $limit",
        )
    }
}

impl<C, Model> VectorStoreIndex for SurrealVectorStore<C, Model>
where
    C: Connection,
    Model: EmbeddingModel,
{
    type Filter = SurrealSearchFilter;

    /// Get the top n documents based on the distance to the given query.
    /// The result is a list of tuples of the form (score, id, document)
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<SurrealSearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let embedded_query: Vec<f64> = self.model.embed_text(req.query()).await?.vec;

        let mut response = self
            .surreal
            .query(self.search_query_full().as_str())
            .bind(("vec", embedded_query))
            .bind(("tablename", self.documents_table.clone()))
            .bind(("threshold", req.threshold().unwrap_or(0.)))
            .bind(("limit", req.samples() as usize))
            .bind((
                "filter",
                req.filter()
                    .clone()
                    .map(SurrealSearchFilter::inner)
                    .unwrap_or("true".into()),
            ))
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
        req: VectorSearchRequest<SurrealSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let embedded_query: Vec<f32> = self
            .model
            .embed_text(req.query())
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
            .bind(("threshold", req.threshold().unwrap_or(0.)))
            .bind(("limit", req.samples() as usize))
            .bind((
                "filter",
                req.filter()
                    .clone()
                    .map(SurrealSearchFilter::inner)
                    .unwrap_or("true".into()),
            ))
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
