#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! LanceDB vector store integration for Rig.
//!
//! This crate provides [`LanceDbVectorIndex`], a Rig vector store index backed
//! by LanceDB tables. It supports exact and approximate vector search through
//! [`SearchType`] and accepts LanceDB SQL filter expressions through
//! [`LanceDBFilter`].
//!
//! Queries arrive pre-embedded via
//! [`VectorSearchRequest`](rig_core::vector_store::VectorSearchRequest):
//! embed the query text with your embedding model, then pass the resulting
//! [`Embedding`](rig_core::embeddings::Embedding) to the request builder.
//!
//! The root `rig` facade re-exports this crate as `rig::lancedb` when the
//! `lancedb` feature is enabled.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
    types::{Float32Type, Float64Type},
};
use lancedb::{
    DistanceType,
    arrow::arrow_schema::DataType,
    query::{QueryBase, VectorQuery},
};
use rig_core::vector_store::{
    SearchHit, StoreRecord, VectorStoreError,
    request::{FilterError, SearchFilter, VectorSearchRequest},
};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use utils::{FilterTableColumns, QueryToJson};

mod utils;

fn lancedb_to_rig_error(e: lancedb::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

fn serde_to_rig_error(e: serde_json::Error) -> VectorStoreError {
    VectorStoreError::JsonError(e)
}

/// Type on which vector searches can be performed for a lanceDb table.
///
/// Queries arrive pre-embedded via [`VectorSearchRequest`]; the store never
/// embeds text itself.
///
/// # Example
/// ```ignore
/// use rig_lancedb::{LanceDbVectorIndex, SearchParams};
///
/// let table: lancedb::Table = db.create_table(""); // <-- Replace with your lancedb table here.
/// let vector_store_index = LanceDbVectorIndex::new(table, "id", SearchParams::default()).await?;
/// ```
pub struct LanceDbVectorIndex {
    /// LanceDB table containing embeddings.
    table: lancedb::Table,
    /// Column name in `table` that contains the id of a record.
    id_field: String,
    /// Vector search params that are used during vector search operations.
    search_params: SearchParams,
}

impl LanceDbVectorIndex {
    /// Create an instance of `LanceDbVectorIndex` with an existing table.
    /// Define the id field name of the table.
    /// Define search parameters that will be used to perform vector searches on the table.
    pub async fn new(
        table: lancedb::Table,
        id_field: &str,
        search_params: SearchParams,
    ) -> Result<Self, lancedb::Error> {
        Ok(Self {
            table,
            id_field: id_field.to_string(),
            search_params,
        })
    }

    /// Apply the search_params to the vector query.
    /// This is a helper function used by the `top_n` and `top_n_ids` methods.
    fn build_query(&self, mut query: VectorQuery) -> VectorQuery {
        let SearchParams {
            distance_type,
            search_type,
            nprobes,
            refine_factor,
            post_filter,
            column,
        } = self.search_params.clone();

        if let Some(distance_type) = distance_type {
            query = query.distance_type(distance_type);
        }

        if let Some(SearchType::Flat) = search_type {
            query = query.bypass_vector_index();
        }

        if let Some(SearchType::Approximate) = search_type {
            if let Some(nprobes) = nprobes {
                query = query.nprobes(nprobes);
            }
            if let Some(refine_factor) = refine_factor {
                query = query.refine_factor(refine_factor);
            }
        }

        if let Some(true) = post_filter {
            query = query.postfilter();
        }

        if let Some(column) = column {
            query = query.column(column.as_str())
        }

        query
    }
}

/// See [LanceDB vector search](https://lancedb.github.io/lancedb/search/) for more information.
#[derive(Debug, Clone)]
pub enum SearchType {
    // Flat search, also called ENN or kNN.
    Flat,
    /// Approximal Nearest Neighbor search, also called ANN.
    Approximate,
}

/// An eDSL for filtering expressions, is rendered as a `WHERE` clause
#[derive(Debug, Clone)]
pub struct LanceDBFilter(Result<String, FilterError>);

impl serde::Serialize for LanceDBFilter {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match &self.0 {
            Ok(s) => serializer.serialize_str(s),
            Err(e) => serializer.collect_str(e),
        }
    }
}

impl<'de> serde::Deserialize<'de> for LanceDBFilter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        // We can't deserialize to Error, so just create an Ok variant
        Ok(LanceDBFilter(Ok(s)))
    }
}

fn zip_result(
    l: Result<String, FilterError>,
    r: Result<String, FilterError>,
) -> Result<(String, String), FilterError> {
    l.and_then(|l| r.map(|r| (l, r)))
}

impl SearchFilter for LanceDBFilter {
    type Value = serde_json::Value;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(escape_value(value).map(|s| format!("{} = {s}", key.as_ref())))
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(escape_value(value).map(|s| format!("{} > {s}", key.as_ref())))
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(escape_value(value).map(|s| format!("{} < {s}", key.as_ref())))
    }

    fn and(self, rhs: Self) -> Self {
        Self(zip_result(self.0, rhs.0).map(|(l, r)| format!("({l}) AND ({r})")))
    }

    fn or(self, rhs: Self) -> Self {
        Self(zip_result(self.0, rhs.0).map(|(l, r)| format!("({l}) OR ({r})")))
    }
}

fn escape_value(value: serde_json::Value) -> Result<String, FilterError> {
    use serde_json::Value::*;

    match value {
        Null => Ok("NULL".into()),
        Bool(b) => Ok(b.to_string()),
        Number(n) => Ok(n.to_string()),
        String(s) => Ok(format!("'{}'", s.replace("'", "''"))),
        Array(xs) => Ok(format!(
            "({})",
            xs.into_iter()
                .map(escape_value)
                .collect::<Result<Vec<_>, _>>()?
                .join(", ")
        )),
        Object(_) => Err(FilterError::TypeError(
            "objects not supported in SQLite backend".into(),
        )),
    }
}

impl LanceDBFilter {
    pub fn into_inner(self) -> Result<String, FilterError> {
        self.0
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(self.0.map(|s| format!("NOT ({s})")))
    }

    /// IN operator
    pub fn in_values(key: String, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        Self(
            values
                .into_iter()
                .map(escape_value)
                .collect::<Result<Vec<_>, FilterError>>()
                .map(|xs| xs.join(","))
                .map(|xs| format!("{key} IN ({xs})")),
        )
    }

    /// LIKE operator (string pattern matching)
    pub fn like<S>(key: String, pattern: S) -> Self
    where
        S: AsRef<str>,
    {
        Self(
            escape_value(serde_json::Value::String(pattern.as_ref().into()))
                .map(|pat| format!("{key} LIKE {pat}")),
        )
    }

    /// ILIKE operator (case-insensitive pattern matching)
    pub fn ilike<S>(key: String, pattern: S) -> Self
    where
        S: AsRef<str>,
    {
        Self(
            escape_value(serde_json::Value::String(pattern.as_ref().into()))
                .map(|pat| format!("{key} ILIKE {pat}")),
        )
    }

    /// IS NULL check
    pub fn is_null(key: String) -> Self {
        Self(Ok(format!("{key} IS NULL")))
    }

    /// IS NOT NULL check
    pub fn is_not_null(key: String) -> Self {
        Self(Ok(format!("{key} IS NOT NULL")))
    }

    /// Array has any (for LIST columns with scalar index)
    pub fn array_has_any(key: String, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        Self(
            values
                .into_iter()
                .map(escape_value)
                .collect::<Result<Vec<_>, FilterError>>()
                .map(|xs| xs.join(","))
                .map(|xs| format!("array_has_any({key}, ARRAY[{xs}])")),
        )
    }

    /// Array has all (for LIST columns with scalar index)
    pub fn array_has_all(key: String, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        Self(
            values
                .into_iter()
                .map(escape_value)
                .collect::<Result<Vec<_>, FilterError>>()
                .map(|xs| xs.join(","))
                .map(|xs| format!("array_has_all({key}, ARRAY[{xs}])")),
        )
    }

    /// Array length comparison
    pub fn array_length(key: String, length: i32) -> Self {
        Self(Ok(format!("array_length({key}) = {length}")))
    }

    /// BETWEEN operator
    pub fn between<T>(key: String, Range { start, end }: Range<T>) -> Self
    where
        T: PartialOrd + std::fmt::Display + Into<serde_json::Number>,
    {
        Self(Ok(format!("{key} BETWEEN {start} AND {end}")))
    }
}

/// Parameters used to perform a vector search on a LanceDb table.
/// # Example
/// ```
/// let search_params = rig_lancedb::SearchParams::default().distance_type(lancedb::DistanceType::Cosine);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SearchParams {
    distance_type: Option<DistanceType>,
    search_type: Option<SearchType>,
    nprobes: Option<usize>,
    refine_factor: Option<u32>,
    post_filter: Option<bool>,
    column: Option<String>,
}

impl SearchParams {
    /// Sets the distance type of the search params.
    /// Always set the distance_type to match the value used to train the index.
    /// The default is DistanceType::L2.
    pub fn distance_type(mut self, distance_type: DistanceType) -> Self {
        self.distance_type = Some(distance_type);
        self
    }

    /// Sets the search type of the search params.
    /// By default, ANN will be used if there is an index on the table and kNN will be used if there is NO index on the table.
    /// To use the mentioned defaults, do not set the search type.
    pub fn search_type(mut self, search_type: SearchType) -> Self {
        self.search_type = Some(search_type);
        self
    }

    /// Sets the nprobes of the search params.
    /// Only set this value only when the search type is ANN.
    /// See [LanceDb ANN Search](https://lancedb.github.io/lancedb/ann_indexes/#querying-an-ann-index) for more information.
    pub fn nprobes(mut self, nprobes: usize) -> Self {
        self.nprobes = Some(nprobes);
        self
    }

    /// Sets the refine factor of the search params.
    /// Only set this value only when search type is ANN.
    /// See [LanceDb ANN Search](https://lancedb.github.io/lancedb/ann_indexes/#querying-an-ann-index) for more information.
    pub fn refine_factor(mut self, refine_factor: u32) -> Self {
        self.refine_factor = Some(refine_factor);
        self
    }

    /// Sets the post filter of the search params.
    /// If set to true, filtering will happen after the vector search instead of before.
    /// See [LanceDb pre/post filtering](https://lancedb.github.io/lancedb/sql/#pre-and-post-filtering) for more information.
    pub fn post_filter(mut self, post_filter: bool) -> Self {
        self.post_filter = Some(post_filter);
        self
    }

    /// Sets the column of the search params.
    /// Only set this value if there is more than one column that contains lists of floats.
    /// If there is only one column of list of floats, this column will be chosen for the vector search automatically.
    pub fn column(mut self, column: &str) -> Self {
        self.column = Some(column.to_string());
        self
    }
}

impl LanceDbVectorIndex {
    /// Returns the top N most similar rows for a pre-embedded query.
    ///
    /// The [`SearchHit::payload`] is the full table row (minus embedding
    /// columns), serialized to JSON.
    ///
    /// LanceDB searches a single vector: the *first* embedding of the request's
    /// query is used. Scores are raw distances (LanceDB's `_distance` column):
    /// lower is better.
    ///
    /// # Example
    /// ```ignore
    /// use rig_core::{OneOrMany, vector_store::VectorSearchRequest};
    ///
    /// let query_embedding = model.embed_text("What does zindle mean?").await?;
    /// let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 1);
    /// let hits = vector_store_index.top_n(req).await?;
    /// ```
    pub async fn top_n(
        &self,
        req: VectorSearchRequest<LanceDBFilter>,
    ) -> Result<Vec<SearchHit>, VectorStoreError> {
        let prompt_embedding = req.query().first();

        let mut query = self
            .table
            .vector_search(prompt_embedding.vec.clone())
            .map_err(lancedb_to_rig_error)?
            .limit(req.samples() as usize)
            .distance_range(None, req.threshold().map(|x| x as f32))
            .select(lancedb::query::Select::Columns(
                self.table
                    .schema()
                    .await
                    .map_err(lancedb_to_rig_error)?
                    .filter_embeddings(),
            ));

        if let Some(filter) = req.filter() {
            query = query.only_if(filter.clone().into_inner()?)
        }

        self.build_query(query)
            .execute_query()
            .await?
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                Ok(SearchHit {
                    score: match value.get("_distance") {
                        Some(Value::Number(distance)) => distance.as_f64().unwrap_or_default(),
                        _ => 0.0,
                    },
                    id: match value.get(self.id_field.clone()) {
                        Some(Value::String(id)) => id.to_string(),
                        _ => format!("unknown{i}"),
                    },
                    payload: value,
                })
            })
            .collect()
    }

    /// Returns the top N most similar row IDs as `(distance, id)` tuples.
    ///
    /// LanceDB searches a single vector: the *first* embedding of the request's
    /// query is used.
    pub async fn top_n_ids(
        &self,
        req: VectorSearchRequest<LanceDBFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let prompt_embedding = req.query().first();

        let mut query = self
            .table
            .query()
            .select(lancedb::query::Select::Columns(vec![self.id_field.clone()]))
            .nearest_to(prompt_embedding.vec.clone())
            .map_err(lancedb_to_rig_error)?
            .distance_range(None, req.threshold().map(|x| x as f32))
            .limit(req.samples() as usize);

        if let Some(filter) = req.filter() {
            query = query.only_if(filter.clone().into_inner()?)
        }

        self.build_query(query)
            .execute_query()
            .await?
            .into_iter()
            .map(|value| {
                Ok((
                    match value.get("_distance") {
                        Some(Value::Number(distance)) => distance.as_f64().unwrap_or_default(),
                        _ => 0.0,
                    },
                    match value.get(self.id_field.clone()) {
                        Some(Value::String(id)) => id.to_string(),
                        _ => "".to_string(),
                    },
                ))
            })
            .collect()
    }

    /// Returns the top N most similar rows deserialized into `T` as
    /// `(distance, id, document)` tuples. Sugar over [`Self::top_n`].
    pub async fn top_n_as<T: DeserializeOwned>(
        &self,
        req: VectorSearchRequest<LanceDBFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.top_n(req)
            .await?
            .into_iter()
            .map(|hit| {
                let doc = serde_json::from_value(hit.payload).map_err(serde_to_rig_error)?;
                Ok((hit.score, hit.id, doc))
            })
            .collect()
    }

    /// Insert precomputed records into the LanceDB table.
    ///
    /// Each [`StoreRecord`] is mapped onto the table's existing Arrow schema:
    /// - the configured id column receives [`StoreRecord::id`],
    /// - the vector column (the [`SearchParams::column`] if set, otherwise the
    ///   single fixed-size-list float column of the table) receives the record's
    ///   *first* embedding,
    /// - every other column is filled from the field of the same name in the
    ///   record's JSON payload.
    ///
    /// Supported non-vector column types: `Utf8`, `Boolean`, `Int32`, `Int64`,
    /// `Float32` and `Float64`.
    pub async fn insert(&self, records: Vec<StoreRecord>) -> Result<(), VectorStoreError> {
        if records.is_empty() {
            return Ok(());
        }

        let schema = self.table.schema().await.map_err(lancedb_to_rig_error)?;
        let vector_column = self.vector_column(&schema)?;

        let mut columns: Vec<(&str, ArrayRef)> = Vec::with_capacity(schema.fields().len());
        for field in schema.fields() {
            let name = field.name().as_str();
            let array: ArrayRef = if name == self.id_field {
                Arc::new(StringArray::from_iter_values(
                    records.iter().map(|record| record.id.clone()),
                ))
            } else if name == vector_column {
                build_vector_array(&records, field.data_type())?
            } else {
                build_payload_array(&records, name, field.data_type())?
            };
            columns.push((name, array));
        }

        let batch = RecordBatch::try_from_iter(
            columns
                .into_iter()
                .map(|(name, array)| (name.to_string(), array)),
        )
        .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        self.table
            .add(vec![batch])
            .execute()
            .await
            .map_err(lancedb_to_rig_error)?;

        Ok(())
    }

    /// Serializes each document and inserts it. Sugar over [`Self::insert`].
    pub async fn insert_as<T: Serialize>(
        &self,
        docs: Vec<(
            String,
            T,
            rig_core::OneOrMany<rig_core::embeddings::Embedding>,
        )>,
    ) -> Result<(), VectorStoreError> {
        let records = docs
            .into_iter()
            .map(|(id, doc, embeddings)| StoreRecord::new(id, &doc, embeddings))
            .collect::<Result<Vec<_>, _>>()?;
        self.insert(records).await
    }

    /// Resolves the table's vector column: the configured search column if set,
    /// otherwise the single fixed-size-list float column of the schema.
    fn vector_column(
        &self,
        schema: &lancedb::arrow::arrow_schema::Schema,
    ) -> Result<String, VectorStoreError> {
        if let Some(column) = &self.search_params.column {
            return Ok(column.clone());
        }

        let mut vector_columns = schema.fields().iter().filter_map(|field| {
            if let DataType::FixedSizeList(inner, _) = field.data_type()
                && matches!(inner.data_type(), DataType::Float32 | DataType::Float64)
            {
                Some(field.name().clone())
            } else {
                None
            }
        });

        match (vector_columns.next(), vector_columns.next()) {
            (Some(column), None) => Ok(column),
            (Some(_), Some(_)) => Err(VectorStoreError::DatastoreError(
                "table has multiple vector columns; set SearchParams::column to disambiguate"
                    .into(),
            )),
            (None, _) => Err(VectorStoreError::DatastoreError(
                "table has no fixed-size-list float column to store embeddings in".into(),
            )),
        }
    }
}

/// Builds the Arrow vector column from the first embedding of each record.
fn build_vector_array(
    records: &[StoreRecord],
    data_type: &DataType,
) -> Result<ArrayRef, VectorStoreError> {
    let DataType::FixedSizeList(inner, dims) = data_type else {
        return Err(VectorStoreError::DatastoreError(
            format!("vector column has unsupported type {data_type}").into(),
        ));
    };

    for record in records {
        let len = record.embeddings.first().vec.len();
        if len != *dims as usize {
            return Err(VectorStoreError::DatastoreError(
                format!(
                    "embedding for record `{}` has {len} dimensions but the table expects {dims}",
                    record.id
                )
                .into(),
            ));
        }
    }

    match inner.data_type() {
        DataType::Float64 => Ok(Arc::new(FixedSizeListArray::from_iter_primitive::<
            Float64Type,
            _,
            _,
        >(
            records.iter().map(|record| {
                Some(
                    record
                        .embeddings
                        .first()
                        .vec
                        .iter()
                        .map(|x| Some(*x))
                        .collect::<Vec<_>>(),
                )
            }),
            *dims,
        ))),
        DataType::Float32 => Ok(Arc::new(FixedSizeListArray::from_iter_primitive::<
            Float32Type,
            _,
            _,
        >(
            records.iter().map(|record| {
                Some(
                    record
                        .embeddings
                        .first()
                        .vec
                        .iter()
                        .map(|x| Some(*x as f32))
                        .collect::<Vec<_>>(),
                )
            }),
            *dims,
        ))),
        other => Err(VectorStoreError::DatastoreError(
            format!("vector column has unsupported element type {other}").into(),
        )),
    }
}

/// Builds an Arrow column for `field` from the same-named field of each
/// record's JSON payload.
fn build_payload_array(
    records: &[StoreRecord],
    field: &str,
    data_type: &DataType,
) -> Result<ArrayRef, VectorStoreError> {
    let values = records
        .iter()
        .map(|record| record.payload.get(field).cloned().unwrap_or(Value::Null));

    fn type_error(field: &str, value: &Value, data_type: &DataType) -> VectorStoreError {
        VectorStoreError::DatastoreError(
            format!("payload field `{field}` value {value} cannot be stored as {data_type}").into(),
        )
    }

    match data_type {
        DataType::Utf8 => {
            let strings = values
                .map(|value| match value {
                    Value::String(s) => Ok(Some(s)),
                    Value::Null => Ok(None),
                    other => Err(type_error(field, &other, data_type)),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Arc::new(StringArray::from(strings)))
        }
        DataType::Boolean => {
            let bools = values
                .map(|value| match value {
                    Value::Bool(b) => Ok(Some(b)),
                    Value::Null => Ok(None),
                    other => Err(type_error(field, &other, data_type)),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Arc::new(BooleanArray::from(bools)))
        }
        DataType::Int32 => {
            let ints = values
                .map(|value| match &value {
                    Value::Number(n) => n
                        .as_i64()
                        .and_then(|x| i32::try_from(x).ok())
                        .map(Some)
                        .ok_or_else(|| type_error(field, &value, data_type)),
                    Value::Null => Ok(None),
                    other => Err(type_error(field, other, data_type)),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Arc::new(Int32Array::from(ints)))
        }
        DataType::Int64 => {
            let ints = values
                .map(|value| match &value {
                    Value::Number(n) => n
                        .as_i64()
                        .map(Some)
                        .ok_or_else(|| type_error(field, &value, data_type)),
                    Value::Null => Ok(None),
                    other => Err(type_error(field, other, data_type)),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Arc::new(Int64Array::from(ints)))
        }
        DataType::Float32 => {
            let floats = values
                .map(|value| match &value {
                    Value::Number(n) => n
                        .as_f64()
                        .map(|x| Some(x as f32))
                        .ok_or_else(|| type_error(field, &value, data_type)),
                    Value::Null => Ok(None),
                    other => Err(type_error(field, other, data_type)),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Arc::new(Float32Array::from(floats)))
        }
        DataType::Float64 => {
            let floats = values
                .map(|value| match &value {
                    Value::Number(n) => n
                        .as_f64()
                        .map(Some)
                        .ok_or_else(|| type_error(field, &value, data_type)),
                    Value::Null => Ok(None),
                    other => Err(type_error(field, other, data_type)),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Arc::new(Float64Array::from(floats)))
        }
        other => Err(VectorStoreError::DatastoreError(
            format!("column `{field}` has unsupported type {other}").into(),
        )),
    }
}
