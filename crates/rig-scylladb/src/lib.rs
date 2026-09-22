//! ScyllaDB vector store for Rig.
//!
//! [`ScyllaDbVectorStore`] stores documents and embeddings in a ScyllaDB table
//! and scores candidates with cosine similarity in the client process. The `rig`
//! facade re-exports this crate as `rig::scylladb` under the `scylladb` feature.

use rig_core::{
    Embed,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{
            DynamicSearchFilter, Filter, FilterError, SearchFilter, SqlCondition,
            VectorSearchRequest,
        },
    },
    wasm_compat::WasmCompatSend,
};
use scylla::{
    client::{Compression, session::Session, session_builder::SessionBuilder},
    statement::prepared::PreparedStatement,
    value::CqlValue,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, RwLock},
};
use uuid::Uuid;

/// Vector store backed by a ScyllaDB table.
///
/// Queries are embedded with the same model `M` that populated the table, so
/// results are meaningless under another model. Every search reads the matching
/// rows and ranks them client-side.
pub struct ScyllaDbVectorStore<M> {
    model: M,
    pub session: Arc<Session>,
    keyspace: String,
    table: String,
    /// Vector width enforced on insertion.
    dimensions: usize,
    insert_stmt: PreparedStatement,
    search_stmt: PreparedStatement,
    get_by_id_stmt: PreparedStatement,
    /// Statements for filtered scans, keyed by filter condition text.
    cache: Arc<RwLock<HashMap<u64, PreparedStatement>>>,
}

/// Converts a JSON value into a bindable CQL value. Nulls become
/// [`CqlValue::Empty`] and unrepresentable numbers are rejected.
fn cql_value_from_json(value: serde_json::Value) -> Result<CqlValue, FilterError> {
    use scylla::value::CqlVarint;
    use serde_json::Value;

    match value {
        Value::Bool(b) => Ok(CqlValue::Boolean(b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(CqlValue::BigInt(i))
            } else if let Some(u) = n.as_u64() {
                // The leading zero byte keeps the big-endian varint positive.
                let mut bytes = vec![0u8];
                bytes.extend_from_slice(&u.to_be_bytes());
                Ok(CqlValue::Varint(CqlVarint::from_signed_bytes_be(bytes)))
            } else if let Some(f) = n.as_f64() {
                Ok(CqlValue::Double(f))
            } else {
                Err(FilterError::Expected {
                    expected: "Valid number".into(),
                    got: "Invalid number".into(),
                })
            }
        }
        Value::String(s) => Ok(CqlValue::Text(s)),
        Value::Array(arr) => Ok(CqlValue::List(
            arr.into_iter()
                .map(cql_value_from_json)
                .collect::<Result<_, _>>()?,
        )),
        Value::Object(map) => {
            let pairs = map
                .into_iter()
                .map(|(k, v)| Ok((CqlValue::Text(k), cql_value_from_json(v)?)))
                .collect::<Result<Vec<_>, FilterError>>()?;
            Ok(CqlValue::Map(pairs))
        }
        Value::Null => Ok(CqlValue::Empty),
    }
}

/// Bind placeholder token CQL expects.
const PLACEHOLDER: &str = "?";

/// CQL `WHERE` fragment with its bind values. Keys are spliced into the
/// statement verbatim; only values are bound.
#[derive(Clone, Debug)]
pub struct ScyllaSearchFilter(SqlCondition<CqlValue>);

/// Hashes only the condition text, since bound values do not change the
/// prepared statement.
impl std::hash::Hash for ScyllaSearchFilter {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.condition().hash(state);
    }
}

impl SearchFilter for ScyllaSearchFilter {
    type Value = CqlValue;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(SqlCondition::binary(key, "=", PLACEHOLDER, value))
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(SqlCondition::binary(key, ">", PLACEHOLDER, value))
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(SqlCondition::binary(key, "<", PLACEHOLDER, value))
    }

    fn and(self, rhs: Self) -> Self {
        Self(self.0.and(rhs.0))
    }

    fn or(self, rhs: Self) -> Self {
        Self(self.0.or(rhs.0))
    }
}

impl ScyllaSearchFilter {
    fn condition(&self) -> &str {
        self.0.condition()
    }

    fn params(&self) -> &[CqlValue] {
        self.0.params()
    }

    pub fn not(self) -> Self {
        Self(self.0.not())
    }

    pub fn gte(key: impl Into<String>, value: <Self as SearchFilter>::Value) -> Self {
        let key = key.into();
        Self(SqlCondition::binary(key, ">=", PLACEHOLDER, value))
    }

    pub fn lte(key: impl Into<String>, value: <Self as SearchFilter>::Value) -> Self {
        let key = key.into();
        Self(SqlCondition::binary(key, "<=", PLACEHOLDER, value))
    }

    pub fn ne(key: impl Into<String>, value: <Self as SearchFilter>::Value) -> Self {
        let key = key.into();
        Self(SqlCondition::binary(key, "!=", PLACEHOLDER, value))
    }

    pub fn member(key: impl Into<String>, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        let key = key.into();
        Self(SqlCondition::list(key, "IN", PLACEHOLDER, values))
    }
}

impl TryFrom<Filter<serde_json::Value>> for ScyllaSearchFilter {
    type Error = FilterError;

    fn try_from(value: Filter<serde_json::Value>) -> Result<Self, Self::Error> {
        value.try_interpret(cql_value_from_json)
    }
}

impl DynamicSearchFilter for ScyllaSearchFilter {
    fn from_dynamic_filter(filter: Filter<serde_json::Value>) -> Result<Self, FilterError> {
        Self::try_from(filter)
    }
}

impl<M: EmbeddingModel> ScyllaDbVectorStore<M> {
    /// Creates a store, creating the keyspace and table when absent and
    /// preparing the fixed statements.
    ///
    /// The keyspace is created with `SimpleStrategy` at replication factor one.
    /// `keyspace` and `table` are spliced into every statement verbatim.
    /// `dimensions` is the vector width insertion enforces.
    pub async fn new(
        model: M,
        session: Session,
        keyspace: &str,
        table: &str,
        dimensions: usize,
    ) -> Result<Self, VectorStoreError> {
        let session = Arc::new(session);

        let create_keyspace_cql = format!(
            "CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH REPLICATION = {{
                'class': 'SimpleStrategy',
                'replication_factor': 1
            }}"
        );
        session
            .query_unpaged(create_keyspace_cql, &[])
            .await
            .map_err(VectorStoreError::datastore)?;

        // Embeddings are stored as float lists because scoring happens client-side.
        let create_table_cql = format!(
            "CREATE TABLE IF NOT EXISTS {keyspace}.{table} (
                id UUID PRIMARY KEY,
                vector LIST<FLOAT>,
                metadata TEXT,
                created_at BIGINT
            )"
        );
        session
            .query_unpaged(create_table_cql, &[])
            .await
            .map_err(VectorStoreError::datastore)?;

        let insert_stmt = session
            .prepare(format!(
                "INSERT INTO {keyspace}.{table} (id, vector, metadata, created_at) VALUES (?, ?, ?, ?)"
            ))
            .await
            .map_err(VectorStoreError::datastore)?;

        let search_stmt = session
            .prepare(format!(
                "SELECT id, vector, metadata, created_at FROM {keyspace}.{table}"
            ))
            .await
            .map_err(VectorStoreError::datastore)?;

        let get_by_id_stmt = session
            .prepare(format!(
                "SELECT id, vector, metadata, created_at FROM {keyspace}.{table} WHERE id = ?"
            ))
            .await
            .map_err(VectorStoreError::datastore)?;

        Ok(Self {
            model,
            session,
            keyspace: keyspace.to_string(),
            table: table.to_string(),
            dimensions,
            insert_stmt,
            search_stmt,
            get_by_id_stmt,
            cache: Default::default(),
        })
    }

    pub fn session(&self) -> &Arc<Session> {
        &self.session
    }

    pub fn keyspace(&self) -> &str {
        &self.keyspace
    }

    pub fn table(&self) -> &str {
        &self.table
    }

    /// Looks up one stored document by row id, returning `None` when no row
    /// matches. Errors when `id` is not a UUID or the payload does not
    /// deserialize into `T`.
    pub async fn get_by_id<T: for<'a> Deserialize<'a> + Send>(
        &self,
        id: &str,
    ) -> Result<Option<T>, VectorStoreError> {
        let uuid = Uuid::parse_str(id).map_err(VectorStoreError::datastore)?;

        let result = self
            .session
            .execute_unpaged(&self.get_by_id_stmt, (uuid,))
            .await
            .map_err(VectorStoreError::datastore)?;

        let rows_result = result
            .into_rows_result()
            .map_err(VectorStoreError::datastore)?;

        if let Some(first_row) = rows_result
            .rows::<(Uuid, Vec<f32>, String, i64)>()
            .map_err(VectorStoreError::datastore)?
            .next()
        {
            let (_, _, metadata, _) = first_row.map_err(VectorStoreError::datastore)?;

            let payload: T = serde_json::from_str(&metadata)?;
            return Ok(Some(payload));
        }

        Ok(None)
    }

    /// Cosine similarity, returning zero when either vector has zero norm.
    /// Extra components of the longer vector are ignored.
    fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    async fn generate_query_vector(&self, query: &str) -> Result<Vec<f32>, VectorStoreError> {
        let embedding = self.model.embed_text(query).await?;
        Ok(embedding.vec.iter().map(|&x| x as f32).collect())
    }

    /// Returns the prepared scan for this request, preparing and caching a
    /// filtered statement the first time each filter condition is seen.
    async fn get_filter_statement_or_default(
        &self,
        req: &VectorSearchRequest<ScyllaSearchFilter>,
    ) -> Result<PreparedStatement, VectorStoreError> {
        if let Some(filter) = req.filter() {
            let mut hasher = DefaultHasher::new();
            filter.hash(&mut hasher);
            let filter_hash = hasher.finish();

            let statement = if let Some(cached) = self
                .cache
                .read()
                .ok()
                .and_then(|cache| cache.get(&filter_hash).cloned())
            {
                cached
            } else {
                let query = format!(
                    "SELECT id, vector, metadata, created_at FROM {}.{} WHERE {} ALLOW FILTERING",
                    self.keyspace,
                    self.table,
                    filter.condition()
                );

                let prepared = self
                    .session
                    .prepare(query)
                    .await
                    .map_err(VectorStoreError::datastore)?;

                let mut cache = self.cache.write().map_err(|e| {
                    VectorStoreError::DatastoreError(
                        format!("Error writing statement cache: {e}").into(),
                    )
                })?;
                cache.insert(filter_hash, prepared.clone());
                prepared
            };

            Ok(statement)
        } else {
            Ok(self.search_stmt.clone())
        }
    }

    /// Reads every matching row, scores it against the embedded query, and
    /// returns the thresholded rows sorted by descending similarity and
    /// truncated to the requested sample count.
    async fn search_candidates(
        &self,
        req: &VectorSearchRequest<ScyllaSearchFilter>,
    ) -> Result<Vec<(f64, String, String)>, VectorStoreError> {
        let query_vector = self.generate_query_vector(req.query()).await?;

        let statement = self.get_filter_statement_or_default(req).await?;
        let params = req
            .filter()
            .as_ref()
            .map(ScyllaSearchFilter::params)
            .unwrap_or_default();

        let results = self
            .session
            .execute_unpaged(&statement, params)
            .await
            .map_err(VectorStoreError::datastore)?;

        let rows_result = results
            .into_rows_result()
            .map_err(VectorStoreError::datastore)?;

        let mut candidates = Vec::new();

        for row_result in rows_result
            .rows::<(Uuid, Vec<f32>, String, i64)>()
            .map_err(VectorStoreError::datastore)?
        {
            let (id, vector, metadata, _) = row_result.map_err(VectorStoreError::datastore)?;

            let score = Self::cosine_similarity(&query_vector, &vector) as f64;

            if req.threshold().is_some_and(|threshold| score < threshold) {
                continue;
            }

            candidates.push((score, id.to_string(), metadata));
        }

        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        candidates.truncate(req.samples() as usize);

        Ok(candidates)
    }
}

impl<M: EmbeddingModel> InsertDocuments for ScyllaDbVectorStore<M> {
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        for (document, embeddings) in documents {
            let metadata = serde_json::to_string(&document)?;
            let now = chrono::Utc::now().timestamp();

            for embedding in embeddings.into_iter() {
                let vector: Vec<f32> = embedding.vec.into_iter().map(|x| x as f32).collect();

                if vector.len() != self.dimensions {
                    return Err(VectorStoreError::DatastoreError(
                        format!(
                            "Vector dimension mismatch: expected {}, got {}",
                            self.dimensions,
                            vector.len()
                        )
                        .into(),
                    ));
                }

                let id = Uuid::new_v4();

                self.session
                    .execute_unpaged(&self.insert_stmt, (id, vector, &metadata, now))
                    .await
                    .map_err(VectorStoreError::datastore)?;
            }
        }

        Ok(())
    }
}

impl<M: EmbeddingModel> VectorStoreIndex for ScyllaDbVectorStore<M> {
    type Filter = ScyllaSearchFilter;

    /// Returns matches as `(cosine similarity, row id, document)`. Scoring reads
    /// every row the filter admits, so cost grows with the scanned table.
    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<ScyllaSearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.search_candidates(&req)
            .await?
            .into_iter()
            .map(|(score, id, metadata)| Ok((score, id, serde_json::from_str(&metadata)?)))
            .collect()
    }

    /// Like `top_n` but returns `(cosine similarity, row id)` without
    /// deserializing documents.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<ScyllaSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(self
            .search_candidates(&req)
            .await?
            .into_iter()
            .map(|(score, id, _)| (score, id))
            .collect())
    }
}

/// Opens an LZ4-compressed session to a single known node.
pub async fn create_session(uri: &str) -> Result<Session, VectorStoreError> {
    SessionBuilder::new()
        .known_node(uri)
        .compression(Some(Compression::Lz4))
        .build()
        .await
        .map_err(VectorStoreError::datastore)
}

#[cfg(test)]
mod tests;
