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
    embeddings::{Embedding, EmbeddingModel},
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

impl<C, Model> InsertDocuments for SurrealVectorStore<C, Model>
where
    C: Connection + Send + Sync,
    Model: EmbeddingModel + Send + Sync,
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
                .create::<Option<CreateRecord>>(self.documents_table.clone())
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
    pub fn contains(key: String, val: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} CONTAINS {}", val.to_sql()))
    }

    /// Test if the value at `key` does *not* contain `val`
    pub fn does_not_contain(key: String, val: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} CONTAINSNOT {}", val.to_sql()))
    }

    /// Test if the value at `key` contains every element of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn all(key: String, vals: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} CONTAINSALL {}", vals.to_sql()))
    }

    /// Test if the value at `key` contains any elements of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn any(key: String, vals: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} CONTAINSANY {}", vals.to_sql()))
    }

    /// Test if the value at `key` is a member of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn member(key: String, vals: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} IN {}", vals.to_sql()))
    }

    /// Test if the value at `key` is *not* a member of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn not_member(key: String, vals: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} NOTIN {}", vals.to_sql()))
    }

    // Geospatial filters
    /// Test if the value at `key` is inside `geometry`
    pub fn inside(key: String, geometry: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} INSIDE {}", geometry.to_sql()))
    }

    /// Test if the value at `key` is outside `geometry`
    pub fn outside(key: String, geometry: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} OUTSIDE {}", geometry.to_sql()))
    }

    /// Test if the value at `key` intersects `geometry`
    pub fn intersects(key: String, geometry: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} INTERSECTS {}", geometry.to_sql()))
    }

    // String ops
    /// SurrealDB text search
    pub fn matches<'a, S: AsRef<&'a str>>(key: String, query: S) -> Self {
        Self(format!("{key} @@ {}", query.as_ref()))
    }

    /// Check if the value at `key` matches regex `pattern`
    /// `pattern` should be a valid surrealDB regex
    pub fn regex<'a, S: AsRef<&'a str>>(key: String, pattern: S) -> Self {
        Self(format!("{key} = /{}/", pattern.as_ref()))
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
                    .map(SurrealSearchFilter::inner)
                    .unwrap_or("true".into()),
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
mod tests {
    use super::{Mem, SurrealSearchFilter, SurrealVectorStore};
    use rig_core::{
        client::Nothing,
        embeddings::{Embedding, EmbeddingError, EmbeddingModel},
        vector_store::{VectorStoreIndexDyn, request::Filter},
    };
    use serde_json::json;
    use surrealdb::Surreal;

    #[derive(Clone)]
    struct MockEmbeddingModel;

    impl EmbeddingModel for MockEmbeddingModel {
        const MAX_DOCUMENTS: usize = 4;

        type Client = Nothing;

        fn make(_: &Self::Client, _: impl Into<String>, _: Option<usize>) -> Self {
            Self
        }

        fn ndims(&self) -> usize {
            3
        }

        async fn embed_texts(
            &self,
            texts: impl IntoIterator<Item = String> + Send,
        ) -> Result<Vec<Embedding>, EmbeddingError> {
            Ok(texts
                .into_iter()
                .map(|text| Embedding {
                    document: text,
                    vec: vec![0.0, 0.0, 0.0],
                })
                .collect())
        }
    }

    #[allow(clippy::panic)]
    #[test]
    fn filter_from_json_preserves_nested_values() {
        let filter = match SurrealSearchFilter::try_from(Filter::Eq(
            "metadata".to_string(),
            json!({
                "name": "rig",
                "flags": { "native": true },
                "tags": ["surreal", "json"]
            }),
        )) {
            Ok(filter) => filter,
            Err(err) => panic!("unexpected surreal filter conversion failure: {err}"),
        };

        let sql = filter.to_string();

        assert!(sql.starts_with("metadata = {"));
        assert!(sql.contains("name: 'rig'"));
        assert!(sql.contains("flags: { native: true }"));
        assert!(sql.contains("tags: ['surreal', 'json']"));
    }

    #[allow(clippy::panic)]
    #[tokio::test]
    async fn surreal_vector_store_supports_type_erased_queries() {
        fn assert_dyn<T: VectorStoreIndexDyn + Send + Sync + 'static>(_: T) {}

        let surreal = match Surreal::new::<Mem>(()).await {
            Ok(surreal) => surreal,
            Err(err) => panic!("failed to create in-memory surreal client: {err}"),
        };
        let vector_store = SurrealVectorStore::with_defaults(MockEmbeddingModel, surreal);

        assert_dyn(vector_store);
    }
}
