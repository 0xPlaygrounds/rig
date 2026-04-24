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

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, FixedSizeListArray, RecordBatch, RecordBatchIterator,
    types::{Float32Type, Float64Type},
};
use arrow_json::ReaderBuilder;
use lancedb::{
    DistanceType,
    arrow::arrow_schema::{DataType, Field, Fields, Schema},
    query::{QueryBase, VectorQuery},
};
use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, embedding::EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{FilterError, SearchFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utils::{FilterTableColumns, QueryToJson};

mod utils;

fn lancedb_to_rig_error(e: lancedb::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

fn serde_to_rig_error(e: serde_json::Error) -> VectorStoreError {
    VectorStoreError::JsonError(e)
}

fn arrow_to_rig_error(e: lancedb::arrow::arrow_schema::ArrowError) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

/// Type on which vector searches can be performed for a lanceDb table.
/// # Example
/// ```ignore
/// use rig_lancedb::{LanceDbVectorIndex, SearchParams};
/// use rig::client::ProviderClient;
/// use rig::providers::openai::{Client, TEXT_EMBEDDING_ADA_002, EmbeddingModel};
///
/// let openai_client = Client::from_env()?;
///
/// let table: lancedb::Table = db.create_table(""); // <-- Replace with your lancedb table here.
/// let model: EmbeddingModel = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002); // <-- Replace with your embedding model here.
/// let vector_store_index = LanceDbVectorIndex::new(table, model, "id", SearchParams::default()).await?;
/// ```
pub struct LanceDbVectorIndex<M: EmbeddingModel> {
    /// Defines which model is used to generate embeddings for the vector store.
    model: M,
    /// LanceDB table containing embeddings.
    table: lancedb::Table,
    /// Column name in `table` that contains the id of a record.
    id_field: String,
    /// Vector search params that are used during vector search operations.
    search_params: SearchParams,
}

impl<M> LanceDbVectorIndex<M>
where
    M: EmbeddingModel,
{
    /// Create an instance of `LanceDbVectorIndex` with an existing table and model.
    /// Define the id field name of the table.
    /// Define search parameters that will be used to perform vector searches on the table.
    pub async fn new(
        table: lancedb::Table,
        model: M,
        id_field: &str,
        search_params: SearchParams,
    ) -> Result<Self, lancedb::Error> {
        Ok(Self {
            table,
            model,
            id_field: id_field.to_string(),
            search_params,
        })
    }

    /// Apply the search_params to the vector query.
    /// This is a helper function used by the methods `top_n` and `top_n_ids` of the `VectorStoreIndex` trait.
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

    /// Resolve the embedding column name from `search_params.column` or by
    /// auto-detecting the single `FixedSizeList<Float32|Float64>` column in the
    /// table schema (mirroring LanceDB's default vector-column inference).
    fn resolve_embedding_column(
        &self,
        schema: &Schema,
    ) -> Result<String, VectorStoreError> {
        if let Some(col) = &self.search_params.column {
            return Ok(col.clone());
        }

        let candidates: Vec<&str> = schema
            .fields()
            .iter()
            .filter(|f| {
                matches!(
                    f.data_type(),
                    DataType::FixedSizeList(inner, _)
                        if matches!(
                            inner.data_type(),
                            DataType::Float32 | DataType::Float64
                        )
                )
            })
            .map(|f| f.name().as_str())
            .collect();

        match candidates.as_slice() {
            [only] => Ok((*only).to_string()),
            [] => Err(VectorStoreError::DatastoreError(
                "no FixedSizeList<Float32|Float64> column found in table schema; \
                 set SearchParams::column to specify the embedding column"
                    .into(),
            )),
            _ => Err(VectorStoreError::DatastoreError(
                format!(
                    "multiple FixedSizeList columns found ({candidates:?}); \
                     set SearchParams::column to disambiguate"
                )
                .into(),
            )),
        }
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

impl<M> VectorStoreIndex for LanceDbVectorIndex<M>
where
    M: EmbeddingModel + Sync + Send,
{
    type Filter = LanceDBFilter;

    /// Implement the `top_n` method of the `VectorStoreIndex` trait for `LanceDbVectorIndex`.
    /// # Example
    /// ```ignore
    /// use rig_lancedb::{LanceDbVectorIndex, SearchParams};
    /// use rig::client::ProviderClient;
    /// use rig::providers::openai::{EmbeddingModel, Client, TEXT_EMBEDDING_ADA_002};
    ///
    /// let openai_client = Client::from_env()?;
    ///
    /// let table: lancedb::Table = db.create_table("fake_definitions"); // <-- Replace with your lancedb table here.
    /// let model: EmbeddingModel = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002); // <-- Replace with your embedding model here.
    /// let vector_store_index = LanceDbVectorIndex::new(table, model, "id", SearchParams::default()).await?;
    ///
    /// // Query the index
    /// let result = vector_store_index
    ///     .top_n::<String>("My boss says I zindle too much, what does that mean?", 1)
    ///     .await?;
    /// ```
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<LanceDBFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_text(req.query()).await?;

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
                Ok((
                    match value.get("_distance") {
                        Some(Value::Number(distance)) => distance.as_f64().unwrap_or_default(),
                        _ => 0.0,
                    },
                    match value.get(self.id_field.clone()) {
                        Some(Value::String(id)) => id.to_string(),
                        _ => format!("unknown{i}"),
                    },
                    serde_json::from_value(value).map_err(serde_to_rig_error)?,
                ))
            })
            .collect()
    }

    /// Implement the `top_n_ids` method of the `VectorStoreIndex` trait for `LanceDbVectorIndex`.
    /// # Example
    /// ```ignore
    /// use rig_lancedb::{LanceDbVectorIndex, SearchParams};
    /// use rig::client::ProviderClient;
    /// use rig::providers::openai::{Client, TEXT_EMBEDDING_ADA_002, EmbeddingModel};
    ///
    /// let openai_client = Client::from_env()?;
    ///
    /// let table: lancedb::Table = db.create_table(""); // <-- Replace with your lancedb table here.
    /// let model: EmbeddingModel = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002); // <-- Replace with your embedding model here.
    /// let vector_store_index = LanceDbVectorIndex::new(table, model, "id", SearchParams::default()).await?;
    ///
    /// // Query the index
    /// let result = vector_store_index
    ///     .top_n_ids("My boss says I zindle too much, what does that mean?", 1)
    ///     .await?;
    /// ```
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<LanceDBFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_text(req.query()).await?;

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
                    match value.get("distance") {
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
}

/// Implement the `InsertDocuments` trait for `LanceDbVectorIndex` so callers can
/// push `(Doc, OneOrMany<Embedding>)` pairs directly into the backing table
/// without hand-building Arrow `RecordBatch`es.
///
/// Behaviour:
/// - Each embedding produces one row: the document fields are flattened onto
///   that row (or stored under a `document` field if the serialized doc is not
///   a JSON object), and the embedding vector is written to the table's
///   `FixedSizeList<Float32|Float64>` column.
/// - The embedding column is taken from `SearchParams::column` if set; otherwise
///   it is auto-detected as the sole `FixedSizeList<Float32|Float64>` column in
///   the table schema (same default LanceDB uses for search).
/// - Doc JSON fields that are absent from the table schema are silently dropped
///   by the Arrow JSON decoder. Rows should be consistent with the schema the
///   table was created with.
impl<M> InsertDocuments for LanceDbVectorIndex<M>
where
    M: EmbeddingModel + Sync + Send,
{
    async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        if documents.is_empty() {
            return Ok(());
        }

        let table_schema: Arc<Schema> =
            self.table.schema().await.map_err(lancedb_to_rig_error)?;
        let embedding_column = self.resolve_embedding_column(&table_schema)?;

        // Extract embedding field + its vector datatype + dims.
        let embedding_field = table_schema
            .field_with_name(&embedding_column)
            .map_err(arrow_to_rig_error)?
            .clone();
        let (embedding_inner_dtype, embedding_dims) = match embedding_field.data_type() {
            DataType::FixedSizeList(inner, dims) => (inner.data_type().clone(), *dims),
            _ => {
                return Err(VectorStoreError::DatastoreError(
                    format!(
                        "embedding column `{embedding_column}` is not a FixedSizeList"
                    )
                    .into(),
                ));
            }
        };

        // Build a schema for the non-embedding columns. We decode the JSON rows
        // against this schema, then splice the embedding column back in.
        let non_embedding_fields: Vec<Arc<Field>> = table_schema
            .fields()
            .iter()
            .filter(|f| f.name() != &embedding_column)
            .cloned()
            .collect();
        let non_embedding_schema = Arc::new(Schema::new(Fields::from(
            non_embedding_fields.clone(),
        )));

        // Flatten (Doc, OneOrMany<Embedding>) into parallel vectors of
        // JSON doc-rows and embedding vectors (one row per embedding).
        let mut json_rows: Vec<Value> = Vec::new();
        let mut embedding_vecs: Vec<Vec<f64>> = Vec::new();

        for (doc, embeddings) in documents {
            let doc_value = serde_json::to_value(&doc).map_err(serde_to_rig_error)?;
            for embedding in embeddings.into_iter() {
                let row = match &doc_value {
                    Value::Object(_) => doc_value.clone(),
                    other => {
                        let mut map = serde_json::Map::new();
                        map.insert("document".to_string(), other.clone());
                        Value::Object(map)
                    }
                };
                json_rows.push(row);
                if embedding.vec.len() != embedding_dims as usize {
                    return Err(VectorStoreError::DatastoreError(
                        format!(
                            "embedding dim mismatch: got {} expected {} \
                             for column `{embedding_column}`",
                            embedding.vec.len(),
                            embedding_dims
                        )
                        .into(),
                    ));
                }
                embedding_vecs.push(embedding.vec);
            }
        }

        if json_rows.is_empty() {
            return Ok(());
        }

        // Decode JSON rows into Arrow columns matching non-embedding schema.
        let mut decoder = ReaderBuilder::new(non_embedding_schema.clone())
            .build_decoder()
            .map_err(arrow_to_rig_error)?;
        decoder.serialize(&json_rows).map_err(arrow_to_rig_error)?;
        let partial_batch = decoder
            .flush()
            .map_err(arrow_to_rig_error)?
            .ok_or_else(|| {
                VectorStoreError::DatastoreError(
                    "arrow-json decoder produced no batch".into(),
                )
            })?;

        // Build the embedding column as FixedSizeList<Float32|Float64>.
        let embedding_array: ArrayRef = match embedding_inner_dtype {
            DataType::Float64 => Arc::new(FixedSizeListArray::from_iter_primitive::<
                Float64Type,
                _,
                _,
            >(
                embedding_vecs
                    .iter()
                    .map(|v| Some(v.iter().copied().map(Some).collect::<Vec<_>>()))
                    .collect::<Vec<_>>(),
                embedding_dims,
            )),
            DataType::Float32 => Arc::new(FixedSizeListArray::from_iter_primitive::<
                Float32Type,
                _,
                _,
            >(
                embedding_vecs
                    .iter()
                    .map(|v| {
                        Some(
                            v.iter()
                                .map(|x| Some(*x as f32))
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>(),
                embedding_dims,
            )),
            other => {
                return Err(VectorStoreError::DatastoreError(
                    format!(
                        "unsupported embedding inner dtype `{other:?}`; \
                         expected Float32 or Float64"
                    )
                    .into(),
                ));
            }
        };

        // Stitch columns back together in the order the table schema expects.
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(table_schema.fields().len());
        for field in table_schema.fields() {
            if field.name() == &embedding_column {
                columns.push(embedding_array.clone());
            } else {
                let idx = non_embedding_schema
                    .index_of(field.name())
                    .map_err(arrow_to_rig_error)?;
                columns.push(partial_batch.column(idx).clone());
            }
        }

        let batch = RecordBatch::try_new(table_schema.clone(), columns)
            .map_err(arrow_to_rig_error)?;

        self.table
            .add(RecordBatchIterator::new(vec![Ok(batch)], table_schema))
            .execute()
            .await
            .map_err(lancedb_to_rig_error)?;

        Ok(())
    }
}
