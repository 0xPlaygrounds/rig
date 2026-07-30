//! SurrealDB vector store integration for Rig.
//!
//! This crate provides [`SurrealVectorStore`], a Rig vector store backed by
//! SurrealDB. It supports local in-memory and remote WebSocket connections
//! through the re-exported SurrealDB engine types.
//!
//! Queries arrive pre-embedded via [`VectorSearchRequest`]; the store never
//! embeds text itself.
//!
//! The root `rig` facade re-exports this crate as `rig::surrealdb` when the
//! `surrealdb` feature is enabled.

use std::fmt::Display;

use rig_core::{
    OneOrMany,
    embeddings::Embedding,
    vector_store::{
        SearchHit, StoreRecord, VectorStoreError,
        request::{Filter as CoreFilter, FilterError, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use surrealdb::{
    Connection, Surreal,
    types::{RecordId, RecordIdKey, SurrealValue, ToSql, Value},
};

pub use surrealdb::engine::local::Mem;
pub use surrealdb::engine::remote::ws::{Ws, Wss};

pub struct SurrealVectorStore<C>
where
    C: Connection,
{
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
    fn into_hit(self) -> Result<SearchHit, VectorStoreError> {
        let payload: serde_json::Value =
            serde_json::from_str(&self.document).map_err(VectorStoreError::JsonError)?;

        Ok(SearchHit {
            id: record_key_to_string(&self.id.key),
            score: self.distance,
            payload,
        })
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealSearchFilter(String);

impl SurrealSearchFilter {
    fn inner(self) -> String {
        self.0
    }
}

impl TryFrom<CoreFilter<serde_json::Value>> for SurrealSearchFilter {
    type Error = FilterError;

    fn try_from(value: CoreFilter<serde_json::Value>) -> Result<Self, Self::Error> {
        match value {
            CoreFilter::Eq(key, value) => Ok(Self::eq(key, Value::from_t(value))),
            CoreFilter::Gt(key, value) => Ok(Self::gt(key, Value::from_t(value))),
            CoreFilter::Lt(key, value) => Ok(Self::lt(key, Value::from_t(value))),
            CoreFilter::And(lhs, rhs) => Ok(Self::try_from(*lhs)?.and(Self::try_from(*rhs)?)),
            CoreFilter::Or(lhs, rhs) => Ok(Self::try_from(*lhs)?.or(Self::try_from(*rhs)?)),
        }
    }
}

impl std::fmt::Display for SurrealSearchFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl SurrealSearchFilter {
    #[allow(clippy::should_implement_trait)]
    pub fn eq(key: impl AsRef<str>, value: Value) -> Self {
        Self(format!("{} = {}", key.as_ref(), value.to_sql()))
    }

    pub fn gt(key: impl AsRef<str>, value: Value) -> Self {
        Self(format!("{} > {}", key.as_ref(), value.to_sql()))
    }

    pub fn lt(key: impl AsRef<str>, value: Value) -> Self {
        Self(format!("{} < {}", key.as_ref(), value.to_sql()))
    }

    pub fn and(self, rhs: Self) -> Self {
        Self(format!("({self}) AND ({rhs})"))
    }

    pub fn or(self, rhs: Self) -> Self {
        Self(format!("({self}) OR ({rhs})"))
    }
}

impl SurrealSearchFilter {
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(format!("NOT ({self})"))
    }

    /// Test if the value at `key` contains `val`
    pub fn contains(key: String, val: Value) -> Self {
        Self(format!("{key} CONTAINS {}", val.to_sql()))
    }

    /// Test if the value at `key` does *not* contain `val`
    pub fn does_not_contain(key: String, val: Value) -> Self {
        Self(format!("{key} CONTAINSNOT {}", val.to_sql()))
    }

    /// Test if the value at `key` contains every element of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn all(key: String, vals: Value) -> Self {
        Self(format!("{key} CONTAINSALL {}", vals.to_sql()))
    }

    /// Test if the value at `key` contains any elements of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn any(key: String, vals: Value) -> Self {
        Self(format!("{key} CONTAINSANY {}", vals.to_sql()))
    }

    /// Test if the value at `key` is a member of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn member(key: String, vals: Value) -> Self {
        Self(format!("{key} IN {}", vals.to_sql()))
    }

    /// Test if the value at `key` is *not* a member of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn not_member(key: String, vals: Value) -> Self {
        Self(format!("{key} NOTIN {}", vals.to_sql()))
    }

    // Geospatial filters
    /// Test if the value at `key` is inside `geometry`
    pub fn inside(key: String, geometry: Value) -> Self {
        Self(format!("{key} INSIDE {}", geometry.to_sql()))
    }

    /// Test if the value at `key` is outside `geometry`
    pub fn outside(key: String, geometry: Value) -> Self {
        Self(format!("{key} OUTSIDE {}", geometry.to_sql()))
    }

    /// Test if the value at `key` intersects `geometry`
    pub fn intersects(key: String, geometry: Value) -> Self {
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

impl<C> SurrealVectorStore<C>
where
    C: Connection,
{
    pub fn new(
        surreal: Surreal<C>,
        documents_table: Option<String>,
        distance_function: SurrealDistanceFunction,
    ) -> Self {
        Self {
            surreal,
            documents_table: documents_table.unwrap_or(String::from("documents")),
            distance_function,
        }
    }

    pub fn inner_client(&self) -> &Surreal<C> {
        &self.surreal
    }

    pub fn with_defaults(surreal: Surreal<C>) -> Self {
        Self::new(surreal, None, SurrealDistanceFunction::Cosine)
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

    /// Extracts the query vector from a pre-embedded search request.
    ///
    /// Searches use the first query embedding; SurrealDB distance functions
    /// compare against a single query vector per statement.
    fn query_vector(req: &VectorSearchRequest<SurrealSearchFilter>) -> Vec<f64> {
        req.query().first().vec.clone()
    }

    /// Insert precomputed records into the store.
    ///
    /// The backing table stores one row per embedding. The record's `id` names
    /// the row created for the record's first embedding; rows for any
    /// additional embeddings get SurrealDB-generated ids.
    pub async fn insert(&self, records: Vec<StoreRecord>) -> Result<(), VectorStoreError> {
        for record in records {
            let json_document_as_string =
                serde_json::to_string(&record.payload).map_err(VectorStoreError::JsonError)?;

            for (i, embedding) in record.embeddings.into_iter().enumerate() {
                let embedded_text = embedding.document;
                let embedding: Vec<f64> = embedding.vec;

                let create_record = CreateRecord {
                    document: json_document_as_string.clone(),
                    embedded_text,
                    embedding,
                };

                if i == 0 {
                    self.surreal
                        .create::<Option<CreateRecord>>((
                            self.documents_table.clone(),
                            record.id.clone(),
                        ))
                        .content(create_record)
                        .await
                        .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
                } else {
                    self.surreal
                        .create::<Option<CreateRecord>>(self.documents_table.clone())
                        .content(create_record)
                        .await
                        .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
                }
            }
        }

        Ok(())
    }

    /// Serializes each document and inserts it. Sugar over [`Self::insert`].
    pub async fn insert_as<T: Serialize>(
        &self,
        docs: Vec<(String, T, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let records = docs
            .into_iter()
            .map(|(id, doc, embeddings)| StoreRecord::new(id, &doc, embeddings))
            .collect::<Result<Vec<_>, _>>()?;
        self.insert(records).await
    }

    /// Get the top n documents based on the distance to the pre-embedded query.
    /// The result is a list of [`SearchHit`]s carrying each document's JSON payload.
    pub async fn top_n(
        &self,
        req: VectorSearchRequest<SurrealSearchFilter>,
    ) -> Result<Vec<SearchHit>, VectorStoreError> {
        let embedded_query = Self::query_vector(&req);

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

        rows.into_iter().map(SearchResult::into_hit).collect()
    }

    /// Same as [`Self::top_n`] but returns the document ids only.
    pub async fn top_n_ids(
        &self,
        req: VectorSearchRequest<SurrealSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let embedded_query: Vec<f32> = Self::query_vector(&req).iter().map(|&x| x as f32).collect();

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

        let rows: Vec<SearchResultOnlyId> = response
            .take::<Vec<SearchResultOnlyId>>(0)
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows: Vec<(f64, String)> = rows
            .into_iter()
            .map(|row| (row.distance, record_key_to_string(&row.id.key)))
            .collect();

        Ok(rows)
    }

    /// Same as [`Self::top_n`] but deserializes each payload into `T`.
    /// The result is a list of `(score, id, document)` tuples.
    pub async fn top_n_as<T: DeserializeOwned>(
        &self,
        req: VectorSearchRequest<SurrealSearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.top_n(req)
            .await?
            .into_iter()
            .map(|hit| {
                let doc = serde_json::from_value(hit.payload)?;
                Ok((hit.score, hit.id, doc))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{Mem, SurrealSearchFilter, SurrealVectorStore};
    use rig_core::{
        OneOrMany,
        embeddings::Embedding,
        vector_store::{StoreRecord, request::Filter as CoreFilter, request::VectorSearchRequest},
    };
    use serde_json::json;
    use surrealdb::Surreal;

    #[allow(clippy::panic)]
    #[test]
    fn filter_from_json_preserves_nested_values() {
        let filter = match SurrealSearchFilter::try_from(CoreFilter::Eq(
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

    #[allow(clippy::panic, clippy::unwrap_used)]
    #[tokio::test]
    async fn insert_and_search_roundtrip_with_pre_embedded_query() {
        let surreal = match Surreal::new::<Mem>(()).await {
            Ok(surreal) => surreal,
            Err(err) => panic!("failed to create in-memory surreal client: {err}"),
        };
        surreal.use_ns("test").use_db("test").await.unwrap();

        let vector_store = SurrealVectorStore::with_defaults(surreal);

        let embedding = |text: &str, vec: Vec<f64>| Embedding {
            document: text.to_string(),
            vec,
        };

        vector_store
            .insert(vec![
                StoreRecord::new(
                    "doc1",
                    &json!({ "text": "glarb-garb" }),
                    OneOrMany::one(embedding("glarb-garb", vec![0.0, 0.1, 0.6])),
                )
                .unwrap(),
                StoreRecord::new(
                    "doc2",
                    &json!({ "text": "marble-marble" }),
                    OneOrMany::one(embedding("marble-marble", vec![0.7, -0.3, 0.0])),
                )
                .unwrap(),
            ])
            .await
            .unwrap();

        let hits = vector_store
            .top_n(VectorSearchRequest::new(
                OneOrMany::one(embedding("query", vec![0.0, 0.1, 0.6])),
                1,
            ))
            .await
            .unwrap();

        assert_eq!(hits.len(), 1);
        let hit = hits.into_iter().next().unwrap();
        assert_eq!(hit.id, "doc1");
        assert_eq!(hit.payload, json!({ "text": "glarb-garb" }));
        assert!(hit.score > 0.99);

        let ids = vector_store
            .top_n_ids(VectorSearchRequest::new(
                OneOrMany::one(embedding("query", vec![0.7, -0.3, 0.0])),
                1,
            ))
            .await
            .unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids.first().unwrap().1, "doc2");
    }
}
