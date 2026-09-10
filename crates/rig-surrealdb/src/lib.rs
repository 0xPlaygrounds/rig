//! SurrealDB vector store integration for Rig.
//!
//! This crate provides [`SurrealVectorStore`], a Rig vector store backed by
//! SurrealDB. It supports local in-memory and remote WebSocket connections
//! through the re-exported SurrealDB engine types.
//!
//! The root `rig` facade re-exports this crate as `rig::surrealdb` when the
//! `surrealdb` feature is enabled.

use std::fmt::Display;

use rig_core::{
    Embed,
    embeddings::{Embedding, EmbeddingModel, EmbeddingModelHandle},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{DynamicSearchFilter, Filter, FilterError, SearchFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use surrealdb::{
    Connection, Surreal,
    types::{RecordId, RecordIdKey, SurrealValue, ToSql, Value},
};

pub use surrealdb::engine::local::Mem;
pub use surrealdb::engine::remote::ws::{Ws, Wss};

/// A SurrealDB-backed vector store.
///
/// The embedding model's concrete type is erased at construction into an
/// [`EmbeddingModelHandle`], which is fixed for the store's lifetime: an index
/// populated under one model is only meaningful when queried under that model.
pub struct SurrealVectorStore<C>
where
    C: Connection,
{
    model: EmbeddingModelHandle,
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

#[derive(Debug, Deserialize, SurrealValue)]
struct SearchResult {
    id: RecordId,
    document: String,
    distance: f64,
}

#[derive(Debug, Serialize, Deserialize, SurrealValue)]
pub struct CreateRecord {
    document: String,
    embedded_text: String,
    embedding: Vec<f64>,
}

#[derive(Debug, Deserialize, SurrealValue)]
pub struct SearchResultOnlyId {
    id: RecordId,
    distance: f64,
}

impl SearchResult {
    pub fn into_result<T: DeserializeOwned>(self) -> Result<(f64, String, T), VectorStoreError> {
        let document: T =
            serde_json::from_str(&self.document).map_err(VectorStoreError::JsonError)?;

        Ok((self.distance, record_key_to_string(&self.id.key), document))
    }
}

fn record_key_to_string(key: &RecordIdKey) -> String {
    match key {
        RecordIdKey::Number(value) => value.to_string(),
        RecordIdKey::String(value) => value.clone(),
        RecordIdKey::Uuid(value) => value.to_string(),
        RecordIdKey::Array(_) | RecordIdKey::Object(_) | RecordIdKey::Range(_) => key.to_sql(),
    }
}

impl<C> InsertDocuments for SurrealVectorStore<C>
where
    C: Connection,
{
    async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let records =
            rig_core::vector_store::flatten_embedded(documents, |json_document, embedding| {
                Ok(CreateRecord {
                    document: serde_json::to_string(json_document)?,
                    embedded_text: embedding.document,
                    embedding: embedding.vec,
                })
            })?;

        for record in records {
            self.surreal
                .create::<Option<CreateRecord>>(self.documents_table.as_str())
                .content(record)
                .await
                .map_err(VectorStoreError::datastore)?;
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

impl TryFrom<Filter<serde_json::Value>> for SurrealSearchFilter {
    type Error = FilterError;

    fn try_from(value: Filter<serde_json::Value>) -> Result<Self, Self::Error> {
        value.try_interpret(|v| Ok(Value::from_t(v)))
    }
}

impl DynamicSearchFilter for SurrealSearchFilter {
    fn from_dynamic_filter(filter: Filter<serde_json::Value>) -> Result<Self, FilterError> {
        Self::try_from(filter)
    }
}

impl std::fmt::Display for SurrealSearchFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl SearchFilter for SurrealSearchFilter {
    type Value = Value;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(format!("{} = {}", key.as_ref(), value.to_sql()))
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(format!("{} > {}", key.as_ref(), value.to_sql()))
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(format!("{} < {}", key.as_ref(), value.to_sql()))
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

    /// Test if the value at `key` contains `val`
    pub fn contains(key: &str, val: &<Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} CONTAINS {}", val.to_sql()))
    }

    /// Test if the value at `key` does *not* contain `val`
    pub fn does_not_contain(key: &str, val: &<Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} CONTAINSNOT {}", val.to_sql()))
    }

    /// Test if the value at `key` contains every element of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn all(key: &str, vals: &<Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} CONTAINSALL {}", vals.to_sql()))
    }

    /// Test if the value at `key` contains any elements of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn any(key: &str, vals: &<Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} CONTAINSANY {}", vals.to_sql()))
    }

    /// Test if the value at `key` is a member of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn member(key: &str, vals: &<Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} IN {}", vals.to_sql()))
    }

    /// Test if the value at `key` is *not* a member of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn not_member(key: &str, vals: &<Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} NOTIN {}", vals.to_sql()))
    }

    // Geospatial filters
    /// Test if the value at `key` is inside `geometry`
    pub fn inside(key: &str, geometry: &<Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} INSIDE {}", geometry.to_sql()))
    }

    /// Test if the value at `key` is outside `geometry`
    pub fn outside(key: &str, geometry: &<Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} OUTSIDE {}", geometry.to_sql()))
    }

    /// Test if the value at `key` intersects `geometry`
    pub fn intersects(key: &str, geometry: &<Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} INTERSECTS {}", geometry.to_sql()))
    }

    // String ops
    /// SurrealDB text search
    pub fn matches<'a, S: AsRef<&'a str>>(key: &str, query: S) -> Self {
        Self(format!("{key} @@ {}", query.as_ref()))
    }

    /// Check if the value at `key` matches regex `pattern`
    /// `pattern` should be a valid surrealDB regex
    pub fn regex<'a, S: AsRef<&'a str>>(key: &str, pattern: S) -> Self {
        Self(format!("{key} = /{}/", pattern.as_ref()))
    }
}

impl<C> SurrealVectorStore<C>
where
    C: Connection,
{
    pub fn new(
        model: impl EmbeddingModel + 'static,
        surreal: Surreal<C>,
        documents_table: Option<String>,
        distance_function: SurrealDistanceFunction,
    ) -> Self {
        Self {
            model: EmbeddingModelHandle::new(model),
            surreal,
            documents_table: documents_table.unwrap_or_else(|| String::from("documents")),
            distance_function,
        }
    }

    pub fn inner_client(&self) -> &Surreal<C> {
        &self.surreal
    }

    pub fn with_defaults(model: impl EmbeddingModel + 'static, surreal: Surreal<C>) -> Self {
        Self::new(model, surreal, None, SurrealDistanceFunction::Cosine)
    }

    /// Embeds the query and runs the similarity-search query, returning the raw response.
    async fn run_search_query(
        &self,
        req: &VectorSearchRequest<SurrealSearchFilter>,
        with_document: bool,
    ) -> Result<surrealdb::IndexedResults, VectorStoreError> {
        let embedded_query: Vec<f64> = self.model.embed_text(req.query()).await?.vec;

        self.surreal
            .query(self.search_query(with_document).as_str())
            .bind(("vec", embedded_query))
            .bind(("tablename", self.documents_table.clone()))
            .bind(("threshold", req.threshold().unwrap_or(0.)))
            .bind(("limit", req.samples() as usize))
            .bind((
                "filter",
                req.filter()
                    .clone()
                    .map_or("true".into(), SurrealSearchFilter::inner),
            ))
            .await
            .map_err(VectorStoreError::datastore)
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

impl<C> VectorStoreIndex for SurrealVectorStore<C>
where
    C: Connection,
{
    type Filter = SurrealSearchFilter;

    /// Get the top n documents based on the distance to the given query.
    /// The result is a list of tuples of the form (score, id, document)
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<SurrealSearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let mut response = self.run_search_query(&req, true).await?;

        let rows: Vec<SearchResult> = response.take(0).map_err(VectorStoreError::datastore)?;

        let rows: Vec<(f64, String, T)> = rows
            .into_iter()
            .map(SearchResult::into_result)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    /// Same as `top_n` but returns the document ids only.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<SurrealSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        // NOTE: this previously bound the query vector as `Vec<f32>` while
        // `top_n` bound `Vec<f64>`; both now bind `Vec<f64>`, matching the
        // stored `embedding: Vec<f64>` schema.
        let mut response = self.run_search_query(&req, false).await?;

        let rows: Vec<SearchResultOnlyId> = response
            .take::<Vec<SearchResultOnlyId>>(0)
            .map_err(VectorStoreError::datastore)?;

        let rows: Vec<(f64, String)> = rows
            .into_iter()
            .map(|row| (row.distance, record_key_to_string(&row.id.key)))
            .collect();

        Ok(rows)
    }
}

#[cfg(test)]
mod tests;
