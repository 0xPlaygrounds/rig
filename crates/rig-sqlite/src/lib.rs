//! SQLite vector store integration for Rig.
//!
//! This crate provides [`SqliteVectorStore`] and [`SqliteVectorIndex`] for
//! storing embedded documents in SQLite with the `sqlite-vec` extension. Define
//! document table schemas by implementing [`SqliteVectorStoreTable`].
//!
//! The root `rig` facade re-exports this crate as `rig::sqlite` when the
//! `sqlite` feature is enabled.

use rig_core::embeddings::{Embedding, EmbeddingModel};
use rig_core::vector_store::request::{FilterError, SearchFilter, VectorSearchRequest};
use rig_core::vector_store::{InsertDocuments, VectorStoreError, VectorStoreIndex};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use rig_core::{Embed, OneOrMany};
use rusqlite::OptionalExtension;
use rusqlite::types::{Type, Value, ValueRef};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Display};
use std::marker::PhantomData;
use std::ops::RangeInclusive;
use tokio_rusqlite::Connection;
use tracing::{debug, info};
use zerocopy::IntoBytes;

#[derive(Debug)]
pub enum SqliteError {
    DatabaseError(Box<dyn std::error::Error + Send + Sync>),
    SerializationError(Box<dyn std::error::Error + Send + Sync>),
    InvalidColumnType(String),
}

/// Value that can be stored in a SQLite vector store document column.
pub trait ColumnValue: Send + Sync {
    /// Converts this value to a typed SQLite value.
    fn to_sql_value(&self) -> Value;

    /// Returns the SQLite type name for this value.
    fn column_type(&self) -> &'static str;
}

#[derive(Clone, Debug)]
pub struct Column {
    name: &'static str,
    col_type: &'static str,
    indexed: bool,
}

impl Column {
    pub fn new(name: &'static str, col_type: &'static str) -> Self {
        Self {
            name,
            col_type,
            indexed: false,
        }
    }

    /// Marks this column as filterable.
    ///
    /// Filterable columns are indexed on the document table and stored as
    /// sqlite-vec metadata columns so supported filters can be applied during
    /// KNN candidate search. Filters that cannot be applied by sqlite-vec
    /// metadata constraints are rejected instead of being applied after the
    /// candidate limit.
    pub fn indexed(mut self) -> Self {
        self.indexed = true;
        self
    }
}

/// Example of a document type that can be used with SqliteVectorStore
/// ```rust
/// use rig_core::Embed;
/// use serde::{Deserialize, Serialize};
/// use rig_sqlite::{Column, ColumnValue, SqliteVectorStoreTable};
///
/// #[derive(Embed, Clone, Debug, Deserialize, Serialize)]
/// struct Document {
///     id: String,
///     #[embed]
///     content: String,
/// }
///
/// impl SqliteVectorStoreTable for Document {
///     fn name() -> &'static str {
///         "documents"
///     }
///
///     fn schema() -> Vec<Column> {
///         vec![
///             Column::new("id", "TEXT PRIMARY KEY"),
///             Column::new("content", "TEXT"),
///         ]
///     }
///
///     fn id(&self) -> String {
///         self.id.clone()
///     }
///
///     fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
///         vec![
///             ("id", Box::new(self.id.clone())),
///             ("content", Box::new(self.content.clone())),
///         ]
///     }
/// }
/// ```
pub trait SqliteVectorStoreTable: Send + Sync + Clone {
    fn name() -> &'static str;
    fn schema() -> Vec<Column>;
    fn id(&self) -> String;
    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)>;
}

/// Distance metric used by SQLite vector searches.
///
/// The metric is applied consistently to sqlite-vec candidate search,
/// thresholding, ordering, and returned scores. Returned scores are
/// higher-is-better: [`SqliteDistanceMetric::Cosine`] returns cosine similarity
/// (`1 - cosine_distance`), while [`SqliteDistanceMetric::L2`] and
/// [`SqliteDistanceMetric::L1`] return the negative sqlite-vec distance.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum SqliteDistanceMetric {
    /// Cosine similarity, returned as `1 - cosine_distance`.
    #[default]
    Cosine,
    /// Negative sqlite-vec L2 distance.
    L2,
    /// Negative sqlite-vec L1 distance.
    L1,
}

impl SqliteDistanceMetric {
    fn vec0_name(self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::L2 => "l2",
            Self::L1 => "l1",
        }
    }

    fn score_expression(self, query_param: &str, embedding_expr: &str) -> String {
        match self {
            Self::Cosine => {
                format!("(1 - vec_distance_cosine({query_param}, {embedding_expr}))")
            }
            Self::L2 => format!("(-vec_distance_l2({query_param}, {embedding_expr}))"),
            Self::L1 => format!("(-vec_distance_l1({query_param}, {embedding_expr}))"),
        }
    }
}

#[derive(Debug)]
struct SqliteDistanceMetricMismatch {
    table_name: String,
    requested: SqliteDistanceMetric,
    configured: SqliteDistanceMetric,
}

impl Display for SqliteDistanceMetricMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SQLite vector table `{}` uses {:?}, but {:?} was requested",
            self.table_name, self.configured, self.requested
        )
    }
}

impl std::error::Error for SqliteDistanceMetricMismatch {}

#[derive(Debug)]
struct SqliteVectorTableMissingSchema {
    table_name: String,
}

impl Display for SqliteVectorTableMissingSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SQLite vector table `{}` was created but is missing from sqlite_schema",
            self.table_name
        )
    }
}

impl std::error::Error for SqliteVectorTableMissingSchema {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SqliteMetadataType {
    Text,
    Integer,
    Float,
    Boolean,
}

impl SqliteMetadataType {
    fn from_column_type(column_type: &str) -> Option<Self> {
        let first_type_token = column_type
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_ascii_uppercase();

        match first_type_token.as_str() {
            "TEXT" => Some(Self::Text),
            "INTEGER" | "INT" | "INT64" | "INTEGER64" => Some(Self::Integer),
            "FLOAT" | "REAL" | "DOUBLE" | "FLOAT64" | "F64" => Some(Self::Float),
            "BOOLEAN" | "BOOL" => Some(Self::Boolean),
            _ => match SqliteColumnAffinity::from_column_type(column_type) {
                SqliteColumnAffinity::Text => Some(Self::Text),
                SqliteColumnAffinity::Integer => Some(Self::Integer),
                SqliteColumnAffinity::Float => Some(Self::Float),
                SqliteColumnAffinity::Boolean => Some(Self::Boolean),
                SqliteColumnAffinity::Numeric | SqliteColumnAffinity::Blob => None,
            },
        }
    }

    fn vec0_name(self) -> &'static str {
        match self {
            Self::Text => "TEXT",
            Self::Integer => "INTEGER",
            Self::Float => "FLOAT",
            Self::Boolean => "BOOLEAN",
        }
    }

    fn supports_native_comparison(self, op: SqliteComparisonOp) -> bool {
        !matches!(
            (self, op),
            (
                Self::Boolean,
                SqliteComparisonOp::Gt
                    | SqliteComparisonOp::Lt
                    | SqliteComparisonOp::Gte
                    | SqliteComparisonOp::Lte
            )
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SqliteColumnAffinity {
    Text,
    Integer,
    Float,
    Boolean,
    Numeric,
    Blob,
}

impl SqliteColumnAffinity {
    fn from_column_type(column_type: &str) -> Self {
        let column_type = column_type.to_ascii_uppercase();

        if column_type.contains("INT") {
            Self::Integer
        } else if column_type.contains("CHAR")
            || column_type.contains("CLOB")
            || column_type.contains("TEXT")
        {
            Self::Text
        } else if column_type.contains("BLOB") || column_type.trim().is_empty() {
            Self::Blob
        } else if column_type.contains("REAL")
            || column_type.contains("FLOA")
            || column_type.contains("DOUB")
        {
            Self::Float
        } else if column_type.contains("BOOL") {
            Self::Boolean
        } else {
            Self::Numeric
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SqliteMetadataColumn {
    name: &'static str,
    metadata_type: SqliteMetadataType,
}

#[derive(Debug)]
struct SqliteUnsupportedMetadataColumn {
    column_name: &'static str,
    column_type: &'static str,
}

impl Display for SqliteUnsupportedMetadataColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SQLite metadata column `{}` has unsupported type `{}`",
            self.column_name, self.column_type
        )
    }
}

impl std::error::Error for SqliteUnsupportedMetadataColumn {}

#[derive(Debug)]
struct SqliteMetadataSchemaMismatch {
    table_name: String,
    column_name: &'static str,
    column_type: SqliteMetadataType,
}

impl Display for SqliteMetadataSchemaMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SQLite vector table `{}` is missing metadata column `{} {}`",
            self.table_name,
            self.column_name,
            self.column_type.vec0_name()
        )
    }
}

impl std::error::Error for SqliteMetadataSchemaMismatch {}

#[derive(Debug)]
struct SqliteMetadataValueError {
    column_name: &'static str,
    column_type: SqliteMetadataType,
    value_type: Type,
}

impl Display for SqliteMetadataValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "could not convert SQLite value type `{:?}` for metadata column `{} {}`",
            self.value_type,
            self.column_name,
            self.column_type.vec0_name()
        )
    }
}

impl std::error::Error for SqliteMetadataValueError {}

#[derive(Debug)]
struct SqliteMissingIdColumn {
    table_name: String,
}

impl Display for SqliteMissingIdColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SQLite vector store table `{}` is missing an `id` column",
            self.table_name
        )
    }
}

impl std::error::Error for SqliteMissingIdColumn {}

fn sqlite_metadata_columns(
    schema: &[Column],
) -> Result<Vec<SqliteMetadataColumn>, VectorStoreError> {
    schema
        .iter()
        .filter(|column| column.indexed)
        .map(|column| {
            let metadata_type =
                SqliteMetadataType::from_column_type(column.col_type).ok_or_else(|| {
                    VectorStoreError::DatastoreError(Box::new(SqliteUnsupportedMetadataColumn {
                        column_name: column.name,
                        column_type: column.col_type,
                    }))
                })?;

            Ok(SqliteMetadataColumn {
                name: column.name,
                metadata_type,
            })
        })
        .collect()
}

fn sqlite_metadata_value(
    values: &[(&'static str, Box<dyn ColumnValue>)],
    column: &SqliteMetadataColumn,
) -> rusqlite::Result<Value> {
    let value = values
        .iter()
        .find(|(name, _)| *name == column.name)
        .ok_or_else(|| rusqlite::Error::InvalidParameterName(column.name.to_string()))?
        .1
        .to_sql_value();

    match (column.metadata_type, value) {
        (SqliteMetadataType::Text, Value::Text(value)) => Ok(Value::Text(value)),
        (SqliteMetadataType::Integer, Value::Integer(value)) => Ok(Value::Integer(value)),
        (SqliteMetadataType::Float, Value::Real(value)) => Ok(Value::Real(value)),
        (SqliteMetadataType::Float, Value::Integer(value)) => Ok(Value::Real(value as f64)),
        (SqliteMetadataType::Boolean, Value::Integer(value @ (0 | 1))) => Ok(Value::Integer(value)),
        (_, value) => Err(rusqlite::Error::ToSqlConversionFailure(Box::new(
            SqliteMetadataValueError {
                column_name: column.name,
                column_type: column.metadata_type,
                value_type: value.data_type(),
            },
        ))),
    }
}

#[derive(Clone)]
pub struct SqliteVectorStore<E, T>
where
    E: EmbeddingModel + 'static,
    T: SqliteVectorStoreTable + 'static,
{
    conn: Connection,
    distance_metric: SqliteDistanceMetric,
    metadata_columns: Vec<SqliteMetadataColumn>,
    _phantom: PhantomData<(E, T)>,
}

impl<E, T> SqliteVectorStore<E, T>
where
    E: EmbeddingModel + 'static,
    T: SqliteVectorStoreTable + 'static,
{
    async fn candidate_limit(&self, samples: u64) -> Result<u64, VectorStoreError>
    where
        Self: 'static,
    {
        if samples == 0 {
            return Ok(0);
        }

        let embedding_map_table_name = format!("{}_embedding_map", T::name());
        let (embedding_count, document_count) = self
            .conn
            .call(move |conn| {
                Ok(conn.query_row(
                    &format!(
                        "SELECT COUNT(*), COUNT(DISTINCT document_rowid) FROM {embedding_map_table_name}"
                    ),
                    [],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                )?)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let embedding_count = u64::try_from(embedding_count).unwrap_or(0);
        let document_count = u64::try_from(document_count).unwrap_or(0);

        if embedding_count > document_count {
            Ok(embedding_count)
        } else {
            Ok(samples)
        }
    }
}

impl<E, T> SqliteVectorStore<E, T>
where
    E: EmbeddingModel + Clone + 'static,
    T: SqliteVectorStoreTable + 'static,
{
    /// Creates a SQLite vector store using cosine similarity.
    pub async fn new(conn: Connection, embedding_model: &E) -> Result<Self, VectorStoreError> {
        Self::with_distance_metric(conn, embedding_model, SqliteDistanceMetric::default()).await
    }

    /// Creates a SQLite vector store with the requested distance metric.
    ///
    /// The metric is written into the sqlite-vec virtual table definition so
    /// candidate search uses the same metric as thresholding, ordering, and the
    /// returned score values.
    pub async fn with_distance_metric(
        conn: Connection,
        embedding_model: &E,
        distance_metric: SqliteDistanceMetric,
    ) -> Result<Self, VectorStoreError> {
        let dims = embedding_model.ndims();
        let table_name = T::name();
        let embeddings_table_name = format!("{table_name}_embeddings");
        let embeddings_table_name_for_sql = embeddings_table_name.clone();
        let embedding_map_table_name_for_sql = format!("{table_name}_embedding_map");
        let schema = T::schema();
        let metadata_columns = sqlite_metadata_columns(&schema)?;
        let metadata_columns_for_schema_check = metadata_columns.clone();
        let distance_metric_name = distance_metric.vec0_name();
        let mut embeddings_columns =
            format!("embedding float[{dims}] distance_metric={distance_metric_name}");
        for column in &metadata_columns {
            embeddings_columns.push_str(&format!(
                ", {} {}",
                column.name,
                column.metadata_type.vec0_name()
            ));
        }

        // Build the table schema
        let mut create_table = format!("CREATE TABLE IF NOT EXISTS {table_name} (");

        // Add columns
        let mut first = true;
        for column in &schema {
            if !first {
                create_table.push(',');
            }
            create_table.push_str(&format!("\n    {} {}", column.name, column.col_type));
            first = false;
        }

        create_table.push_str("\n)");

        // Build index creation statements
        let mut create_indexes = vec![format!(
            "CREATE INDEX IF NOT EXISTS idx_{}_id ON {}(id)",
            table_name, table_name
        )];

        // Add indexes for marked columns
        for column in schema {
            if column.indexed {
                create_indexes.push(format!(
                    "CREATE INDEX IF NOT EXISTS idx_{}_{} ON {}({})",
                    table_name, column.name, table_name, column.name
                ));
            }
        }

        let embeddings_table_sql = conn
            .call(move |conn| {
                conn.execute_batch("BEGIN")?;

                // Create document table
                conn.execute_batch(&create_table)?;

                // Create indexes
                for index_stmt in create_indexes {
                    conn.execute_batch(&index_stmt)?;
                }

                // Create embeddings table
                conn.execute_batch(&format!(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS {embeddings_table_name_for_sql} USING vec0({embeddings_columns})"
                ))?;
                conn.execute_batch(&format!(
                    "CREATE TABLE IF NOT EXISTS {embedding_map_table_name_for_sql} (
                        embedding_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_rowid INTEGER NOT NULL
                    )"
                ))?;
                conn.execute_batch(&format!(
                    "CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding_map_document_rowid ON {embedding_map_table_name_for_sql}(document_rowid)"
                ))?;

                conn.execute_batch("COMMIT")?;

                let schema_sql = conn
                    .query_row(
                        "SELECT sql FROM sqlite_schema WHERE name = ?1",
                        [&embeddings_table_name_for_sql],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()?;

                Ok(schema_sql)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let schema_sql = embeddings_table_sql.ok_or_else(|| {
            VectorStoreError::DatastoreError(Box::new(SqliteVectorTableMissingSchema {
                table_name: embeddings_table_name.clone(),
            }))
        })?;

        let configured = sqlite_distance_metric_from_schema(&schema_sql);
        if configured != distance_metric {
            return Err(VectorStoreError::DatastoreError(Box::new(
                SqliteDistanceMetricMismatch {
                    table_name: embeddings_table_name,
                    requested: distance_metric,
                    configured,
                },
            )));
        }
        for column in metadata_columns_for_schema_check {
            if !sqlite_schema_contains_metadata_column(&schema_sql, &column) {
                return Err(VectorStoreError::DatastoreError(Box::new(
                    SqliteMetadataSchemaMismatch {
                        table_name: embeddings_table_name.clone(),
                        column_name: column.name,
                        column_type: column.metadata_type,
                    },
                )));
            }
        }

        Ok(Self {
            conn,
            distance_metric,
            metadata_columns,
            _phantom: PhantomData,
        })
    }

    pub fn index(self, model: E) -> SqliteVectorIndex<E, T> {
        SqliteVectorIndex::new(model, self)
    }

    pub fn add_rows_with_txn(
        &self,
        txn: &rusqlite::Transaction<'_>,
        documents: Vec<(T, OneOrMany<Embedding>)>,
    ) -> Result<i64, tokio_rusqlite::Error> {
        info!("Adding {} documents to store", documents.len());
        let table_name = T::name();
        let embeddings_table_name = format!("{table_name}_embeddings");
        let embedding_map_table_name = format!("{table_name}_embedding_map");
        let mut last_id = 0;
        let embedding_columns = std::iter::once("rowid")
            .chain(std::iter::once("embedding"))
            .chain(self.metadata_columns.iter().map(|column| column.name))
            .collect::<Vec<_>>();
        let embedding_placeholders = (1..=embedding_columns.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>();
        let embeddings_sql = format!(
            "INSERT INTO {embeddings_table_name} ({}) VALUES ({})",
            embedding_columns.join(", "),
            embedding_placeholders.join(", ")
        );
        let existing_rowid_sql = format!("SELECT rowid FROM {table_name} WHERE id = ?1");
        let existing_embedding_rowids_sql = format!(
            "SELECT embedding_rowid FROM {embedding_map_table_name} WHERE document_rowid = ?1"
        );
        let insert_embedding_map_sql =
            format!("INSERT INTO {embedding_map_table_name}(document_rowid) VALUES (?1)");
        let delete_embedding_map_sql =
            format!("DELETE FROM {embedding_map_table_name} WHERE document_rowid = ?1");
        let delete_embeddings_sql = format!("DELETE FROM {embeddings_table_name} WHERE rowid = ?1");

        for (doc, embeddings) in &documents {
            debug!("Storing document with id {}", doc.id());

            let values = doc.column_values();
            let id_value = values
                .iter()
                .find(|(name, _)| *name == "id")
                .map(|(_, value)| value.to_sql_value())
                .unwrap_or_else(|| Value::Text(doc.id()));
            if let Some(existing_rowid) = txn
                .query_row(&existing_rowid_sql, rusqlite::params![id_value], |row| {
                    row.get::<_, i64>(0)
                })
                .optional()?
            {
                let existing_embedding_rowids = txn
                    .prepare(&existing_embedding_rowids_sql)?
                    .query_map([existing_rowid], |row| row.get::<_, i64>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                for embedding_rowid in existing_embedding_rowids {
                    txn.execute(&delete_embeddings_sql, [embedding_rowid])?;
                }
                txn.execute(&delete_embedding_map_sql, [existing_rowid])?;
                txn.execute(&delete_embeddings_sql, [existing_rowid])?;
            }

            let columns = values.iter().map(|(col, _)| *col).collect::<Vec<_>>();

            let placeholders = (1..=values.len())
                .map(|i| format!("?{i}"))
                .collect::<Vec<_>>();

            let insert_sql = format!(
                "INSERT OR REPLACE INTO {} ({}) VALUES ({})",
                table_name,
                columns.join(", "),
                placeholders.join(", ")
            );

            txn.execute(
                &insert_sql,
                rusqlite::params_from_iter(values.iter().map(|(_, val)| val.to_sql_value())),
            )?;
            last_id = txn.last_insert_rowid();

            let metadata_values = self
                .metadata_columns
                .iter()
                .map(|column| sqlite_metadata_value(&values, column))
                .collect::<rusqlite::Result<Vec<_>>>()?;

            let mut stmt = txn.prepare(&embeddings_sql)?;
            for (i, embedding) in embeddings.iter().enumerate() {
                let vec = serialize_embedding(embedding);
                debug!(
                    "Storing embedding {} of {} (size: {} bytes)",
                    i + 1,
                    embeddings.len(),
                    vec.len() * 4
                );
                txn.execute(&insert_embedding_map_sql, [last_id])?;
                let embedding_rowid = txn.last_insert_rowid();
                let mut params = Vec::with_capacity(2 + metadata_values.len());
                params.push(Value::Integer(embedding_rowid));
                params.push(Value::Blob(vec.as_bytes().to_vec()));
                params.extend(metadata_values.iter().cloned());
                stmt.execute(rusqlite::params_from_iter(params))?;
            }
        }

        Ok(last_id)
    }

    pub async fn add_rows(
        &self,
        documents: Vec<(T, OneOrMany<Embedding>)>,
    ) -> Result<i64, VectorStoreError>
    where
        T: 'static,
        Self: 'static,
    {
        let cloned = self.clone();

        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;
                let result = cloned.add_rows_with_txn(&tx, documents)?;
                tx.commit()?;

                Ok(result)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
    }
}

impl<E, T> InsertDocuments for SqliteVectorStore<E, T>
where
    E: EmbeddingModel + Clone + WasmCompatSend + WasmCompatSync + 'static,
    T: SqliteVectorStoreTable
        + for<'de> Deserialize<'de>
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        if documents.is_empty() {
            return Ok(());
        }

        let rows = documents
            .into_iter()
            .map(|(document, embeddings)| {
                let document = serde_json::to_value(document)?;
                let row = serde_json::from_value::<T>(document)?;

                Ok((row, embeddings))
            })
            .collect::<Result<Vec<_>, VectorStoreError>>()?;

        self.add_rows(rows).await?;

        Ok(())
    }
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct SqliteSearchFilter {
    expr: SqliteSearchFilterExpr,
}

impl Default for SqliteSearchFilter {
    fn default() -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Raw {
                condition: "1 = 1".to_string(),
                params: Vec::new(),
            },
        }
    }
}

#[derive(Clone, Deserialize, Serialize, Debug)]
enum SqliteSearchFilterExpr {
    Comparison {
        key: String,
        op: SqliteComparisonOp,
        value: serde_json::Value,
    },
    And(Box<SqliteSearchFilterExpr>, Box<SqliteSearchFilterExpr>),
    Or(Box<SqliteSearchFilterExpr>, Box<SqliteSearchFilterExpr>),
    Not(Box<SqliteSearchFilterExpr>),
    Between {
        key: String,
        lo: serde_json::Value,
        hi: serde_json::Value,
    },
    NullCheck {
        key: String,
        negated: bool,
    },
    Pattern {
        key: String,
        op: SqlitePatternOp,
        pattern: String,
    },
    Raw {
        condition: String,
        params: Vec<serde_json::Value>,
    },
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize, Debug)]
enum SqliteComparisonOp {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
}

impl SqliteComparisonOp {
    fn as_sql(self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::Ne => "!=",
            Self::Gt => ">",
            Self::Gte => ">=",
            Self::Lt => "<",
            Self::Lte => "<=",
        }
    }

    fn negate(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Gt => Self::Lte,
            Self::Gte => Self::Lt,
            Self::Lt => Self::Gte,
            Self::Lte => Self::Gt,
        }
    }
}

#[derive(Clone, Copy, Deserialize, Serialize, Debug)]
enum SqlitePatternOp {
    Glob,
    Like,
}

impl SqlitePatternOp {
    fn as_sql(self) -> &'static str {
        match self {
            Self::Glob => "glob",
            Self::Like => "like",
        }
    }
}

#[derive(Clone, Copy)]
enum SqliteFilterTarget {
    VectorMetadata,
}

impl SqliteFilterTarget {
    fn alias(self) -> &'static str {
        match self {
            Self::VectorMetadata => "e",
        }
    }
}

#[derive(Default)]
struct SqliteRenderedFilters {
    native: Vec<SqliteRenderedFilter>,
    post: Vec<SqliteRenderedFilter>,
}

impl SqliteRenderedFilters {
    fn extend(&mut self, rhs: Self) {
        self.native.extend(rhs.native);
        self.post.extend(rhs.post);
    }
}

struct SqliteRenderedFilter {
    condition: String,
    params: Vec<Value>,
}

impl SearchFilter for SqliteSearchFilter {
    type Value = serde_json::Value;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Comparison {
                key: key.as_ref().to_string(),
                op: SqliteComparisonOp::Eq,
                value,
            },
        }
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Comparison {
                key: key.as_ref().to_string(),
                op: SqliteComparisonOp::Gt,
                value,
            },
        }
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Comparison {
                key: key.as_ref().to_string(),
                op: SqliteComparisonOp::Lt,
                value,
            },
        }
    }

    fn and(self, rhs: Self) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::And(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }

    fn or(self, rhs: Self) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Or(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl SqliteSearchFilter {
    #[allow(clippy::should_implement_trait)]
    /// Negates a simple indexed metadata comparison.
    ///
    /// SQLite vector search lowers `NOT` over comparisons to sqlite-vec-native
    /// operators such as `!=`, `<=`, and `>=`. Broader negations that require
    /// `OR` or post-filtering are rejected during search.
    pub fn not(self) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Not(Box::new(self.expr)),
        }
    }

    /// Tests whether an indexed non-boolean metadata value is contained in the range.
    pub fn between<N>(key: String, range: RangeInclusive<N>) -> Self
    where
        N: Into<serde_json::Value>,
    {
        let (lo, hi) = range.into_inner();

        Self {
            expr: SqliteSearchFilterExpr::Between {
                key,
                lo: lo.into(),
                hi: hi.into(),
            },
        }
    }

    // Null checks
    pub fn is_null(key: String) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::NullCheck {
                key,
                negated: false,
            },
        }
    }

    pub fn is_not_null(key: String) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::NullCheck { key, negated: true },
        }
    }

    /// Tests whether the value at `key` satisfies the glob pattern.
    ///
    /// sqlite-vec cannot enforce `GLOB` during candidate search, so SQLite
    /// vector searches currently reject this filter.
    pub fn glob(key: String, pattern: impl Into<String>) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Pattern {
                key,
                op: SqlitePatternOp::Glob,
                pattern: pattern.into(),
            },
        }
    }

    /// Tests whether the value at `key` satisfies the `LIKE` pattern.
    ///
    /// sqlite-vec cannot enforce `LIKE` during candidate search, so SQLite
    /// vector searches currently reject this filter.
    pub fn like(key: String, pattern: impl Into<String>) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Pattern {
                key,
                op: SqlitePatternOp::Like,
                pattern: pattern.into(),
            },
        }
    }
}

impl SqliteSearchFilter {
    fn raw(condition: impl Into<String>, params: Vec<serde_json::Value>) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Raw {
                condition: condition.into(),
                params,
            },
        }
    }

    fn render_split(
        &self,
        metadata_columns: &[SqliteMetadataColumn],
    ) -> Result<SqliteRenderedFilters, FilterError> {
        self.expr.render_split(metadata_columns)
    }
}

impl SqliteSearchFilterExpr {
    fn render_native_comparison(
        key: &str,
        op: SqliteComparisonOp,
        value: serde_json::Value,
        metadata_columns: &[SqliteMetadataColumn],
    ) -> Result<SqliteRenderedFilters, FilterError> {
        let metadata_column = sqlite_native_metadata_column(key, metadata_columns)?;

        if !metadata_column.metadata_type.supports_native_comparison(op) {
            return Err(sqlite_unsupported_filter(format!(
                "`{key}` is a BOOLEAN metadata column, and sqlite-vec only supports `=` and `!=` filters for booleans"
            )));
        }

        let expr = Self::Comparison {
            key: key.to_string(),
            op,
            value,
        };
        Ok(SqliteRenderedFilters {
            native: vec![expr.render(SqliteFilterTarget::VectorMetadata)?],
            post: Vec::new(),
        })
    }

    fn render_split(
        &self,
        metadata_columns: &[SqliteMetadataColumn],
    ) -> Result<SqliteRenderedFilters, FilterError> {
        match self {
            Self::Comparison { key, op, value } => {
                Self::render_native_comparison(key, *op, value.clone(), metadata_columns)
            }
            Self::And(lhs, rhs) => {
                let mut rendered = lhs.render_split(metadata_columns)?;
                rendered.extend(rhs.render_split(metadata_columns)?);
                Ok(rendered)
            }
            Self::Between { key, lo, hi } => {
                let metadata_column = sqlite_native_metadata_column(key, metadata_columns)?;
                if metadata_column.metadata_type == SqliteMetadataType::Boolean {
                    return Err(sqlite_unsupported_filter(format!(
                        "`{key}` is a BOOLEAN metadata column, and sqlite-vec does not support range filters for booleans"
                    )));
                }

                Ok(SqliteRenderedFilters {
                    native: vec![SqliteRenderedFilter {
                        condition: format!("e.{key} >= ? AND e.{key} <= ?"),
                        params: vec![
                            sqlite_filter_param(lo.clone())?,
                            sqlite_filter_param(hi.clone())?,
                        ],
                    }],
                    post: Vec::new(),
                })
            }
            Self::Raw { condition, params } if condition == "1 = 1" && params.is_empty() => {
                Ok(SqliteRenderedFilters {
                    native: Vec::new(),
                    post: Vec::new(),
                })
            }
            Self::Or(_, _) => Err(sqlite_unsupported_filter(
                "`OR` filters cannot be applied during sqlite-vec candidate search",
            )),
            Self::Not(expr) => expr.render_negated_split(metadata_columns),
            Self::NullCheck { .. } => Err(sqlite_unsupported_filter(
                "`IS NULL` filters are not supported by sqlite-vec metadata constraints",
            )),
            Self::Pattern { op, .. } => Err(sqlite_unsupported_filter(format!(
                "`{}` filters are not supported by sqlite-vec metadata constraints",
                op.as_sql().to_ascii_uppercase()
            ))),
            Self::Raw { .. } => Err(sqlite_unsupported_filter(
                "raw filters cannot be validated as sqlite-vec metadata constraints",
            )),
        }
    }

    fn render_negated_split(
        &self,
        metadata_columns: &[SqliteMetadataColumn],
    ) -> Result<SqliteRenderedFilters, FilterError> {
        match self {
            Self::Comparison { key, op, value } => {
                Self::render_native_comparison(key, op.negate(), value.clone(), metadata_columns)
            }
            Self::Not(expr) => expr.render_split(metadata_columns),
            _ => Err(sqlite_unsupported_filter(
                "`NOT` filters are only supported for simple indexed metadata comparisons",
            )),
        }
    }

    fn render(&self, target: SqliteFilterTarget) -> Result<SqliteRenderedFilter, FilterError> {
        match self {
            Self::Comparison { key, op, value } => Ok(SqliteRenderedFilter {
                condition: format!("{} {} ?", sqlite_qualify_key(key, target), op.as_sql()),
                params: vec![sqlite_filter_param(value.clone())?],
            }),
            Self::And(lhs, rhs) => {
                let lhs = lhs.render(target)?;
                let rhs = rhs.render(target)?;
                Ok(SqliteRenderedFilter {
                    condition: format!("({}) AND ({})", lhs.condition, rhs.condition),
                    params: lhs.params.into_iter().chain(rhs.params).collect(),
                })
            }
            Self::Or(lhs, rhs) => {
                let lhs = lhs.render(target)?;
                let rhs = rhs.render(target)?;
                Ok(SqliteRenderedFilter {
                    condition: format!("({}) OR ({})", lhs.condition, rhs.condition),
                    params: lhs.params.into_iter().chain(rhs.params).collect(),
                })
            }
            Self::Not(expr) => {
                let expr = expr.render(target)?;
                Ok(SqliteRenderedFilter {
                    condition: format!("NOT ({})", expr.condition),
                    params: expr.params,
                })
            }
            Self::Between { key, lo, hi } => Ok(SqliteRenderedFilter {
                condition: format!("{} between ? and ?", sqlite_qualify_key(key, target)),
                params: vec![
                    sqlite_filter_param(lo.clone())?,
                    sqlite_filter_param(hi.clone())?,
                ],
            }),
            Self::NullCheck { key, negated } => {
                let operator = if *negated { "is not null" } else { "is null" };
                Ok(SqliteRenderedFilter {
                    condition: format!("{} {operator}", sqlite_qualify_key(key, target)),
                    params: Vec::new(),
                })
            }
            Self::Pattern { key, op, pattern } => Ok(SqliteRenderedFilter {
                condition: format!("{} {} ?", sqlite_qualify_key(key, target), op.as_sql()),
                params: vec![Value::Text(pattern.clone())],
            }),
            Self::Raw { condition, params } => Ok(SqliteRenderedFilter {
                condition: condition.clone(),
                params: params
                    .iter()
                    .cloned()
                    .map(sqlite_filter_param)
                    .collect::<Result<Vec<_>, _>>()?,
            }),
        }
    }
}

fn sqlite_native_metadata_column<'a>(
    key: &str,
    metadata_columns: &'a [SqliteMetadataColumn],
) -> Result<&'a SqliteMetadataColumn, FilterError> {
    if sqlite_key_is_qualified(key) {
        return Err(sqlite_unsupported_filter(format!(
            "`{key}` is not a plain sqlite-vec metadata column name"
        )));
    }

    metadata_columns
        .iter()
        .find(|column| column.name == key)
        .ok_or_else(|| {
            sqlite_unsupported_filter(format!(
                "`{key}` is not an indexed sqlite-vec metadata column"
            ))
        })
}

fn sqlite_unsupported_filter(reason: impl Into<String>) -> FilterError {
    FilterError::TypeError(format!(
        "SQLite filters must be enforceable during sqlite-vec candidate search; {}",
        reason.into()
    ))
}

fn sqlite_filter_param(value: serde_json::Value) -> Result<Value, FilterError> {
    use serde_json::Value::*;

    match value {
        Null => Ok(Value::Null),
        Bool(b) => Ok(Value::Integer(b as i64)),
        String(s) => Ok(Value::Text(s)),
        Number(n) => Ok(if let Some(value) = n.as_i64() {
            Value::Integer(value)
        } else if let Some(value) = n.as_u64() {
            let value = i64::try_from(value).map_err(|_| {
                FilterError::TypeError(format!(
                    "SQLite integer filter value `{n}` exceeds i64::MAX"
                ))
            })?;
            Value::Integer(value)
        } else if let Some(float) = n.as_f64() {
            Value::Real(float)
        } else {
            Value::Text(n.to_string())
        }),
        Array(arr) => {
            let blob =
                serde_json::to_vec(&arr).map_err(|e| FilterError::Serialization(e.to_string()))?;

            Ok(Value::Blob(blob))
        }
        Object(obj) => {
            let blob =
                serde_json::to_vec(&obj).map_err(|e| FilterError::Serialization(e.to_string()))?;

            Ok(Value::Blob(blob))
        }
    }
}

fn sqlite_key_is_qualified(key: &str) -> bool {
    key.contains('.') || key.contains('(') || key.contains(' ') || key.contains('?')
}

fn sqlite_qualify_key(key: &str, target: SqliteFilterTarget) -> String {
    if sqlite_key_is_qualified(key) {
        key.to_string()
    } else {
        format!("{}.{}", target.alias(), key)
    }
}

/// SQLite vector store implementation for Rig.
///
/// This crate provides a SQLite-based vector store implementation that can be used with Rig.
/// It uses the `sqlite-vec` extension to enable vector similarity search capabilities.
///
/// # Example
/// ```no_run
/// use rig_core::{
///     client::EmbeddingsClient,
///     embeddings::EmbeddingsBuilder,
///     providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
///     vector_store::{InsertDocuments, VectorStoreIndex},
///     Embed,
/// };
/// use rig_sqlite::{
///     Column, ColumnValue, SqliteDistanceMetric, SqliteVectorStore, SqliteVectorStoreTable,
/// };
/// use rig_core::vector_store::request::VectorSearchRequest;
/// use serde::{Deserialize, Serialize};
/// use tokio_rusqlite::Connection;
///
/// # async fn example() -> anyhow::Result<()> {
/// #[derive(Embed, Clone, Debug, Deserialize, Serialize)]
/// struct Document {
///     id: String,
///     #[embed]
///     content: String,
/// }
///
/// impl SqliteVectorStoreTable for Document {
///     fn name() -> &'static str {
///         "documents"
///     }
///
///     fn schema() -> Vec<Column> {
///         vec![
///             Column::new("id", "TEXT PRIMARY KEY"),
///             Column::new("content", "TEXT"),
///         ]
///     }
///
///     fn id(&self) -> String {
///         self.id.clone()
///     }
///
///     fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
///         vec![
///             ("id", Box::new(self.id.clone())),
///             ("content", Box::new(self.content.clone())),
///         ]
///     }
/// }
///
/// let conn = Connection::open("vector_store.db").await?;
/// let openai_client = Client::new("YOUR_API_KEY")?;
/// let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
///
/// // Initialize vector store
/// let vector_store: SqliteVectorStore<_, Document> = SqliteVectorStore::with_distance_metric(
///     conn,
///     &model,
///     SqliteDistanceMetric::Cosine,
/// )
/// .await?;
///
/// // Create documents
/// let documents = vec![
///     Document {
///         id: "doc1".to_string(),
///         content: "Example document 1".to_string(),
///     },
///     Document {
///         id: "doc2".to_string(),
///         content: "Example document 2".to_string(),
///     },
/// ];
///
/// // Generate embeddings
/// let embeddings = EmbeddingsBuilder::new(model.clone())
///     .documents(documents)?
///     .build()
///     .await?;
///
/// // Add to vector store
/// vector_store.insert_documents(embeddings).await?;
///
/// // Create index and search
/// let index = vector_store.index(model);
/// let req = VectorSearchRequest::builder()
///     .query("Example query")
///     .samples(2)
///     .build();
/// let results = index.top_n::<Document>(req).await?;
/// # let _ = results;
/// # Ok(())
/// # }
/// # let _ = example();
/// ```
pub struct SqliteVectorIndex<E, T>
where
    E: EmbeddingModel + 'static,
    T: SqliteVectorStoreTable + 'static,
{
    store: SqliteVectorStore<E, T>,
    embedding_model: E,
}

impl<E, T> SqliteVectorIndex<E, T>
where
    E: EmbeddingModel + 'static,
    T: SqliteVectorStoreTable,
{
    pub fn new(embedding_model: E, store: SqliteVectorStore<E, T>) -> Self {
        Self {
            store,
            embedding_model,
        }
    }
}

fn sqlite_distance_metric_from_schema(schema_sql: &str) -> SqliteDistanceMetric {
    let normalized = sqlite_normalized_schema(schema_sql);

    if normalized.contains("distance_metric=cosine") {
        SqliteDistanceMetric::Cosine
    } else if normalized.contains("distance_metric=l1") {
        SqliteDistanceMetric::L1
    } else {
        SqliteDistanceMetric::L2
    }
}

fn sqlite_normalized_schema(schema_sql: &str) -> String {
    schema_sql
        .chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn sqlite_schema_contains_metadata_column(schema_sql: &str, column: &SqliteMetadataColumn) -> bool {
    let normalized = sqlite_normalized_schema(schema_sql);
    let column_sql = format!(
        ",{}{}",
        column.name.to_ascii_lowercase(),
        column.metadata_type.vec0_name().to_ascii_lowercase()
    );

    normalized.contains(&column_sql)
}

fn build_where_clause(
    req: &VectorSearchRequest<SqliteSearchFilter>,
    query_vec: Vec<f32>,
    distance_metric: SqliteDistanceMetric,
    metadata_columns: &[SqliteMetadataColumn],
    candidate_limit: u64,
) -> Result<(String, Vec<Value>), FilterError> {
    let score_expression = distance_metric.score_expression("?1", "e.embedding");
    let threshold_filter = req.threshold().map(|threshold| {
        SqliteSearchFilter::raw(format!("{score_expression} >= ?"), vec![threshold.into()])
    });

    let mut filters = SqliteRenderedFilters::default();
    if let Some(threshold_filter) = threshold_filter {
        filters.native.push(
            threshold_filter
                .expr
                .render(SqliteFilterTarget::VectorMetadata)?,
        );
    }
    if let Some(filter) = req.filter() {
        filters.extend(filter.render_split(metadata_columns)?);
    }

    let mut conditions = vec!["e.embedding MATCH ?".to_string(), "k = ?".to_string()];
    conditions.extend(
        filters
            .native
            .iter()
            .chain(filters.post.iter())
            .map(|filter| format!("({})", filter.condition)),
    );

    let where_clause = format!("WHERE {}", conditions.join(" AND "));

    let query_vec = query_vec.into_iter().flat_map(f32::to_le_bytes).collect();
    let query_vec = Value::Blob(query_vec);
    let candidate_limit = sqlite_limit_param(candidate_limit, "candidate limit")?;

    let mut params = vec![query_vec.clone(), query_vec, candidate_limit];
    params.extend(filters.native.into_iter().flat_map(|filter| filter.params));
    params.extend(filters.post.into_iter().flat_map(|filter| filter.params));

    Ok((where_clause, params))
}

fn sqlite_limit_param(value: u64, name: &str) -> Result<Value, FilterError> {
    i64::try_from(value)
        .map(Value::Integer)
        .map_err(|_| FilterError::TypeError(format!("SQLite {name} `{value}` exceeds i64::MAX")))
}

#[derive(Debug)]
struct SqliteColumnValueError {
    column_name: &'static str,
    column_type: &'static str,
    message: String,
}

impl Display for SqliteColumnValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "could not convert SQLite column `{}` with declared type `{}`: {}",
            self.column_name, self.column_type, self.message
        )
    }
}

impl std::error::Error for SqliteColumnValueError {}

fn sqlite_column_value_error(
    index: usize,
    value_type: Type,
    column: &Column,
    message: impl Into<String>,
) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        index,
        value_type,
        Box::new(SqliteColumnValueError {
            column_name: column.name,
            column_type: column.col_type,
            message: message.into(),
        }),
    )
}

fn sqlite_number_value(
    index: usize,
    value_type: Type,
    column: &Column,
    value: f64,
) -> rusqlite::Result<serde_json::Value> {
    let number = serde_json::Number::from_f64(value).ok_or_else(|| {
        sqlite_column_value_error(index, value_type, column, "non-finite float value")
    })?;

    Ok(serde_json::Value::Number(number))
}

fn sqlite_text_value(
    index: usize,
    value_type: Type,
    column: &Column,
    value: &[u8],
) -> rusqlite::Result<serde_json::Value> {
    let value = std::str::from_utf8(value).map_err(|e| {
        sqlite_column_value_error(
            index,
            value_type,
            column,
            format!("invalid UTF-8 text: {e}"),
        )
    })?;

    Ok(serde_json::Value::String(value.to_string()))
}

fn sqlite_column_value_to_json(
    index: usize,
    column: &Column,
    value: ValueRef<'_>,
) -> rusqlite::Result<serde_json::Value> {
    let value_type = value.data_type();
    let column_affinity = SqliteColumnAffinity::from_column_type(column.col_type);

    match (column_affinity, value) {
        (_, ValueRef::Null) => Ok(serde_json::Value::Null),
        (SqliteColumnAffinity::Boolean, ValueRef::Integer(0)) => Ok(serde_json::Value::Bool(false)),
        (SqliteColumnAffinity::Boolean, ValueRef::Integer(1)) => Ok(serde_json::Value::Bool(true)),
        (SqliteColumnAffinity::Boolean, _) => Err(sqlite_column_value_error(
            index,
            value_type,
            column,
            "stored SQLite boolean value must be 0 or 1",
        )),
        (_, ValueRef::Text(value)) => sqlite_text_value(index, value_type, column, value),
        (_, ValueRef::Integer(value)) => Ok(serde_json::Value::Number(value.into())),
        (_, ValueRef::Real(value)) => sqlite_number_value(index, value_type, column, value),
        (_, ValueRef::Blob(value)) => Ok(serde_json::to_value(value)
            .map_err(|e| sqlite_column_value_error(index, value_type, column, e.to_string()))?),
    }
}

fn sqlite_id_value_to_string(index: usize, value: ValueRef<'_>) -> rusqlite::Result<String> {
    match value {
        ValueRef::Integer(value) => Ok(value.to_string()),
        ValueRef::Real(value) => Ok(value.to_string()),
        ValueRef::Text(value) => std::str::from_utf8(value)
            .map(ToString::to_string)
            .map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    index,
                    Type::Text,
                    Box::new(SqliteColumnValueError {
                        column_name: "id",
                        column_type: "TEXT",
                        message: format!("invalid UTF-8 text: {e}"),
                    }),
                )
            }),
        value => Err(rusqlite::Error::FromSqlConversionFailure(
            index,
            value.data_type(),
            Box::new(SqliteColumnValueError {
                column_name: "id",
                column_type: "TEXT or INTEGER",
                message: "id cannot be NULL or BLOB".to_string(),
            }),
        )),
    }
}

impl<E: EmbeddingModel + std::marker::Sync, T: SqliteVectorStoreTable> VectorStoreIndex
    for SqliteVectorIndex<E, T>
{
    type Filter = SqliteSearchFilter;

    async fn top_n<D>(
        &self,
        req: VectorSearchRequest<SqliteSearchFilter>,
    ) -> Result<Vec<(f64, String, D)>, VectorStoreError>
    where
        D: for<'de> Deserialize<'de>,
    {
        tracing::debug!("Finding top {} matches for query", req.samples() as usize);
        if req.samples() == 0 {
            return Ok(Vec::new());
        }

        let embedding = self.embedding_model.embed_text(req.query()).await?;
        let query_vec: Vec<f32> = serialize_embedding(&embedding);
        let table_name = T::name();
        let embedding_map_table_name = format!("{table_name}_embedding_map");

        let columns = T::schema();
        let id_column_index = columns
            .iter()
            .position(|column| column.name == "id")
            .ok_or_else(|| {
                VectorStoreError::DatastoreError(Box::new(SqliteMissingIdColumn {
                    table_name: table_name.to_string(),
                }))
            })?;

        // Build SELECT statement with all columns
        let select_cols = columns
            .iter()
            .map(|column| format!("d.{} AS {}", column.name, column.name))
            .collect::<Vec<_>>()
            .join(", ");
        let outer_select_cols = columns
            .iter()
            .map(|column| column.name)
            .collect::<Vec<_>>()
            .join(", ");

        let distance_metric = self.store.distance_metric;
        let score_expression = distance_metric.score_expression("?1", "e.embedding");
        let candidate_limit = self.store.candidate_limit(req.samples()).await?;
        let (where_clause, params) = build_where_clause(
            &req,
            query_vec,
            distance_metric,
            &self.store.metadata_columns,
            candidate_limit,
        )?;
        let mut params = params;
        params.push(sqlite_limit_param(req.samples(), "result limit")?);

        let rows = self
            .store
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(&format!(
                    "WITH scored AS (
                        SELECT {select_cols},
                            {score_expression} as __rig_score,
                            ROW_NUMBER() OVER (
                                PARTITION BY d.rowid
                                ORDER BY {score_expression} DESC, e.rowid ASC
                            ) AS __rig_rank
                        FROM {table_name}_embeddings e
                        JOIN {embedding_map_table_name} m ON e.rowid = m.embedding_rowid
                        JOIN {table_name} d ON m.document_rowid = d.rowid
                        {where_clause}
                    )
                    SELECT {outer_select_cols}, __rig_score
                    FROM scored
                    WHERE __rig_rank = 1
                    ORDER BY __rig_score DESC
                    LIMIT ?"
                ))?;

                let rows = stmt
                    .query_map(rusqlite::params_from_iter(params), |row| {
                        // Create a map of column names to values
                        let mut map = serde_json::Map::new();
                        for (i, column) in columns.iter().enumerate() {
                            let value = sqlite_column_value_to_json(i, column, row.get_ref(i)?)?;
                            map.insert(column.name.to_string(), value);
                        }
                        let score: f64 = row.get(columns.len())?;
                        let id = sqlite_id_value_to_string(
                            id_column_index,
                            row.get_ref(id_column_index)?,
                        )?;

                        Ok((id, serde_json::Value::Object(map), score))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        debug!("Found {} potential matches", rows.len());
        let mut top_n = Vec::new();
        for (id, doc_value, score) in rows {
            match serde_json::from_value::<D>(doc_value) {
                Ok(doc) => {
                    top_n.push((score, id, doc));
                }
                Err(e) => {
                    debug!("Failed to deserialize document {}: {}", id, e);
                    continue;
                }
            }
        }

        debug!("Returning {} matches", top_n.len());
        Ok(top_n)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<SqliteSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        tracing::debug!(
            "Finding top {} document IDs for query",
            req.samples() as usize
        );
        if req.samples() == 0 {
            return Ok(Vec::new());
        }

        let embedding = self.embedding_model.embed_text(req.query()).await?;
        let query_vec = serialize_embedding(&embedding);
        let table_name = T::name();
        let embedding_map_table_name = format!("{table_name}_embedding_map");

        let distance_metric = self.store.distance_metric;
        let score_expression = distance_metric.score_expression("?1", "e.embedding");
        let candidate_limit = self.store.candidate_limit(req.samples()).await?;
        let (where_clause, params) = build_where_clause(
            &req,
            query_vec,
            distance_metric,
            &self.store.metadata_columns,
            candidate_limit,
        )?;
        let mut params = params;
        params.push(sqlite_limit_param(req.samples(), "result limit")?);

        let results = self
            .store
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(&format!(
                    "WITH scored AS (
                        SELECT d.id AS id,
                            {score_expression} as __rig_score,
                            ROW_NUMBER() OVER (
                                PARTITION BY d.rowid
                                ORDER BY {score_expression} DESC, e.rowid ASC
                            ) AS __rig_rank
                        FROM {table_name}_embeddings e
                        JOIN {embedding_map_table_name} m ON e.rowid = m.embedding_rowid
                        JOIN {table_name} d ON m.document_rowid = d.rowid
                        {where_clause}
                     )
                     SELECT id, __rig_score
                     FROM scored
                     WHERE __rig_rank = 1
                     ORDER BY __rig_score DESC
                     LIMIT ?"
                ))?;

                let results = stmt
                    .query_map(rusqlite::params_from_iter(params), |row| {
                        Ok((
                            row.get::<_, f64>(1)?,
                            sqlite_id_value_to_string(0, row.get_ref(0)?)?,
                        ))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(results)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        debug!("Found {} matching document IDs", results.len());
        Ok(results)
    }
}

fn serialize_embedding(embedding: &Embedding) -> Vec<f32> {
    embedding.vec.iter().map(|x| *x as f32).collect()
}

impl ColumnValue for String {
    fn to_sql_value(&self) -> Value {
        Value::Text(self.clone())
    }

    fn column_type(&self) -> &'static str {
        "TEXT"
    }
}

impl ColumnValue for i64 {
    fn to_sql_value(&self) -> Value {
        Value::Integer(*self)
    }

    fn column_type(&self) -> &'static str {
        "INTEGER"
    }
}

impl ColumnValue for i32 {
    fn to_sql_value(&self) -> Value {
        Value::Integer(i64::from(*self))
    }

    fn column_type(&self) -> &'static str {
        "INTEGER"
    }
}

impl ColumnValue for f64 {
    fn to_sql_value(&self) -> Value {
        Value::Real(*self)
    }

    fn column_type(&self) -> &'static str {
        "FLOAT"
    }
}

impl ColumnValue for f32 {
    fn to_sql_value(&self) -> Value {
        Value::Real(f64::from(*self))
    }

    fn column_type(&self) -> &'static str {
        "FLOAT"
    }
}

impl ColumnValue for bool {
    fn to_sql_value(&self) -> Value {
        Value::Integer(if *self { 1 } else { 0 })
    }

    fn column_type(&self) -> &'static str {
        "BOOLEAN"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::embeddings::EmbeddingError;
    use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};
    use sqlite_vec::sqlite3_vec_init;
    use std::cmp::Ordering;
    use std::os::raw::c_char;
    use std::sync::Once;
    use tokio_rusqlite::Connection;

    fn test_metadata_columns() -> Vec<SqliteMetadataColumn> {
        vec![SqliteMetadataColumn {
            name: "category",
            metadata_type: SqliteMetadataType::Text,
        }]
    }

    fn typed_metadata_columns() -> Vec<SqliteMetadataColumn> {
        vec![
            SqliteMetadataColumn {
                name: "priority",
                metadata_type: SqliteMetadataType::Integer,
            },
            SqliteMetadataColumn {
                name: "rating",
                metadata_type: SqliteMetadataType::Float,
            },
            SqliteMetadataColumn {
                name: "published",
                metadata_type: SqliteMetadataType::Boolean,
            },
        ]
    }

    fn filter_error<T: std::fmt::Debug>(
        result: Result<T, FilterError>,
        context: &str,
    ) -> anyhow::Result<FilterError> {
        match result {
            Ok(value) => anyhow::bail!("{context} should have failed, got {value:?}"),
            Err(err) => Ok(err),
        }
    }

    fn ensure_vector_store_filter_error<T: std::fmt::Debug>(
        result: Result<T, VectorStoreError>,
        context: &str,
    ) -> anyhow::Result<()> {
        match result {
            Err(VectorStoreError::FilterError(_)) => Ok(()),
            Err(err) => anyhow::bail!("{context} returned unexpected error: {err}"),
            Ok(value) => anyhow::bail!("{context} should have failed, got {value:?}"),
        }
    }

    #[test]
    fn threshold_filter_uses_computed_similarity_expression() -> anyhow::Result<()> {
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .threshold(0.95)
            .build();

        let (where_clause, params) =
            build_where_clause(&req, vec![1.0, 0.0], SqliteDistanceMetric::Cosine, &[], 5)?;

        anyhow::ensure!(
            where_clause.contains("e.embedding MATCH ?"),
            "missing vector match constraint: {where_clause}"
        );
        anyhow::ensure!(
            where_clause.contains("k = ?"),
            "missing vector k constraint: {where_clause}"
        );
        anyhow::ensure!(
            where_clause.contains("(1 - vec_distance_cosine(?1, e.embedding)) >= ?"),
            "threshold should use computed similarity expression: {where_clause}"
        );
        anyhow::ensure!(params.len() == 4, "unexpected params: {params:?}");
        anyhow::ensure!(
            params.get(3) == Some(&Value::Real(0.95)),
            "unexpected threshold param: {params:?}"
        );

        Ok(())
    }

    #[test]
    fn l2_threshold_filter_uses_l2_score_expression() -> anyhow::Result<()> {
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .threshold(-1.5)
            .build();

        let (where_clause, params) =
            build_where_clause(&req, vec![1.0, 0.0], SqliteDistanceMetric::L2, &[], 5)?;

        anyhow::ensure!(
            where_clause.contains("(-vec_distance_l2(?1, e.embedding)) >= ?"),
            "threshold should use L2 score expression: {where_clause}"
        );
        anyhow::ensure!(params.len() == 4, "unexpected params: {params:?}");
        anyhow::ensure!(
            params.get(3) == Some(&Value::Real(-1.5)),
            "unexpected threshold param: {params:?}"
        );

        Ok(())
    }

    #[test]
    fn no_threshold_does_not_add_similarity_predicate() -> anyhow::Result<()> {
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .build();

        let (where_clause, params) =
            build_where_clause(&req, vec![1.0, 0.0], SqliteDistanceMetric::Cosine, &[], 5)?;

        anyhow::ensure!(
            where_clause == "WHERE e.embedding MATCH ? AND k = ?",
            "unexpected where clause: {where_clause}"
        );
        anyhow::ensure!(params.len() == 3, "unexpected params: {params:?}");

        Ok(())
    }

    #[test]
    fn or_filter_is_rejected_because_it_cannot_constrain_vec0_candidates() -> anyhow::Result<()> {
        let filter = SqliteSearchFilter::eq("category", serde_json::json!("docs")).or(
            SqliteSearchFilter::eq("category", serde_json::json!("archive")),
        );

        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .filter(filter)
            .build();

        let err = filter_error(
            build_where_clause(
                &req,
                vec![1.0, 0.0],
                SqliteDistanceMetric::Cosine,
                &test_metadata_columns(),
                5,
            ),
            "OR filters",
        )?;

        anyhow::ensure!(
            err.to_string().contains("OR"),
            "unexpected error for OR filter: {err}"
        );

        Ok(())
    }

    #[test]
    fn indexed_filter_uses_vec0_metadata_constraint() -> anyhow::Result<()> {
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .filter(SqliteSearchFilter::eq(
                "category",
                serde_json::json!("docs"),
            ))
            .build();

        let (where_clause, params) = build_where_clause(
            &req,
            vec![1.0, 0.0],
            SqliteDistanceMetric::Cosine,
            &test_metadata_columns(),
            5,
        )?;

        anyhow::ensure!(
            where_clause == "WHERE e.embedding MATCH ? AND k = ? AND (e.category = ?)",
            "unexpected where clause: {where_clause}"
        );
        anyhow::ensure!(params.len() == 4, "unexpected params: {params:?}");
        anyhow::ensure!(
            params.get(3) == Some(&Value::Text("docs".to_string())),
            "unexpected filter param: {params:?}"
        );

        Ok(())
    }

    #[test]
    fn negated_eq_filter_uses_vec0_metadata_inequality() -> anyhow::Result<()> {
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .filter(SqliteSearchFilter::eq("category", serde_json::json!("docs")).not())
            .build();

        let (where_clause, params) = build_where_clause(
            &req,
            vec![1.0, 0.0],
            SqliteDistanceMetric::Cosine,
            &test_metadata_columns(),
            5,
        )?;

        anyhow::ensure!(
            where_clause == "WHERE e.embedding MATCH ? AND k = ? AND (e.category != ?)",
            "unexpected where clause: {where_clause}"
        );
        anyhow::ensure!(params.len() == 4, "unexpected params: {params:?}");
        anyhow::ensure!(
            params.get(3) == Some(&Value::Text("docs".to_string())),
            "unexpected filter param: {params:?}"
        );

        Ok(())
    }

    #[test]
    fn negated_range_comparison_uses_vec0_metadata_boundary() -> anyhow::Result<()> {
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .filter(SqliteSearchFilter::gt("priority", serde_json::json!(10)).not())
            .build();

        let (where_clause, params) = build_where_clause(
            &req,
            vec![1.0, 0.0],
            SqliteDistanceMetric::Cosine,
            &typed_metadata_columns(),
            5,
        )?;

        anyhow::ensure!(
            where_clause == "WHERE e.embedding MATCH ? AND k = ? AND (e.priority <= ?)",
            "unexpected where clause: {where_clause}"
        );
        anyhow::ensure!(params.len() == 4, "unexpected params: {params:?}");
        anyhow::ensure!(
            params.get(3) == Some(&Value::Integer(10)),
            "unexpected filter param: {params:?}"
        );

        Ok(())
    }

    #[test]
    fn negated_boolean_eq_filter_uses_vec0_metadata_inequality() -> anyhow::Result<()> {
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .filter(SqliteSearchFilter::eq("published", serde_json::json!(true)).not())
            .build();

        let (where_clause, params) = build_where_clause(
            &req,
            vec![1.0, 0.0],
            SqliteDistanceMetric::Cosine,
            &typed_metadata_columns(),
            5,
        )?;

        anyhow::ensure!(
            where_clause == "WHERE e.embedding MATCH ? AND k = ? AND (e.published != ?)",
            "unexpected where clause: {where_clause}"
        );
        anyhow::ensure!(
            params.get(3) == Some(&Value::Integer(1)),
            "unexpected boolean filter param: {params:?}"
        );

        Ok(())
    }

    #[test]
    fn unsupported_negated_filter_shapes_are_rejected() -> anyhow::Result<()> {
        let filter = SqliteSearchFilter::between("priority".to_string(), 1_i64..=10_i64).not();
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .filter(filter)
            .build();

        let err = filter_error(
            build_where_clause(
                &req,
                vec![1.0, 0.0],
                SqliteDistanceMetric::Cosine,
                &typed_metadata_columns(),
                5,
            ),
            "negated between filters",
        )?;

        anyhow::ensure!(
            err.to_string()
                .contains("simple indexed metadata comparisons"),
            "unexpected error for negated between filter: {err}"
        );

        Ok(())
    }

    #[test]
    fn boolean_range_filter_is_rejected() -> anyhow::Result<()> {
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .filter(SqliteSearchFilter::gt(
                "published",
                serde_json::json!(false),
            ))
            .build();

        let err = filter_error(
            build_where_clause(
                &req,
                vec![1.0, 0.0],
                SqliteDistanceMetric::Cosine,
                &typed_metadata_columns(),
                5,
            ),
            "boolean range filters",
        )?;

        anyhow::ensure!(
            err.to_string().contains("BOOLEAN"),
            "unexpected error for boolean range filter: {err}"
        );

        Ok(())
    }

    #[test]
    fn indexed_between_filter_uses_vec0_metadata_constraints() -> anyhow::Result<()> {
        let filter = SqliteSearchFilter::between("priority".to_string(), 1_i64..=10_i64);
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .filter(filter)
            .build();

        let (where_clause, params) = build_where_clause(
            &req,
            vec![1.0, 0.0],
            SqliteDistanceMetric::Cosine,
            &typed_metadata_columns(),
            5,
        )?;

        anyhow::ensure!(
            where_clause
                == "WHERE e.embedding MATCH ? AND k = ? AND (e.priority >= ? AND e.priority <= ?)",
            "unexpected where clause: {where_clause}"
        );
        anyhow::ensure!(params.len() == 5, "unexpected params: {params:?}");
        anyhow::ensure!(
            params.get(3) == Some(&Value::Integer(1)) && params.get(4) == Some(&Value::Integer(10)),
            "between bounds should be bound as parameters: {params:?}"
        );

        Ok(())
    }

    #[test]
    fn pattern_filters_are_rejected() -> anyhow::Result<()> {
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .filter(SqliteSearchFilter::like(
                "category".to_string(),
                "%O'Reilly%",
            ))
            .build();

        let err = filter_error(
            build_where_clause(
                &req,
                vec![1.0, 0.0],
                SqliteDistanceMetric::Cosine,
                &test_metadata_columns(),
                5,
            ),
            "LIKE filters",
        )?;

        anyhow::ensure!(
            err.to_string().contains("LIKE"),
            "unexpected error for LIKE filter: {err}"
        );

        Ok(())
    }

    #[test]
    fn nonindexed_filters_are_rejected() -> anyhow::Result<()> {
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .filter(SqliteSearchFilter::eq("title", serde_json::json!("docs")))
            .build();

        let err = filter_error(
            build_where_clause(
                &req,
                vec![1.0, 0.0],
                SqliteDistanceMetric::Cosine,
                &test_metadata_columns(),
                5,
            ),
            "filters on non-indexed columns",
        )?;

        anyhow::ensure!(
            err.to_string().contains("not an indexed"),
            "unexpected error for non-indexed filter: {err}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn live_search_orders_by_similarity_and_applies_threshold() -> anyhow::Result<()> {
        let index = live_test_index(
            "live_search_orders_by_similarity_and_applies_threshold",
            vec![
                row("exact", "docs", "exact match", vec![1.0, 0.0]),
                row("close", "docs", "close match", vec![0.8, 0.6]),
                row("opposite", "docs", "opposite match", vec![-1.0, 0.0]),
            ],
        )
        .await?;

        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(3)
            .threshold(0.75)
            .build();

        let results = index.top_n::<TestDocument>(req.clone()).await?;
        let ids = results
            .iter()
            .map(|(_, id, _)| id.as_str())
            .collect::<Vec<_>>();
        let exact_score = results.first().map(|(score, _, _)| *score);
        let close_score = results.get(1).map(|(score, _, _)| *score);

        anyhow::ensure!(
            ids.as_slice() == ["exact", "close"],
            "unexpected ids: {ids:?}"
        );
        anyhow::ensure!(
            exact_score
                .zip(close_score)
                .is_some_and(|(exact, close)| exact > close),
            "expected exact score to be greater than close score: {results:?}"
        );
        anyhow::ensure!(
            results.iter().all(|(score, _, _)| *score > 0.75),
            "threshold should remove low-scoring rows: {results:?}"
        );

        let id_results = index.top_n_ids(req).await?;
        let result_ids = id_results
            .iter()
            .map(|(_, id)| id.as_str())
            .collect::<Vec<_>>();

        anyhow::ensure!(
            result_ids.as_slice() == ["exact", "close"],
            "unexpected top_n_ids ids: {id_results:?}"
        );
        anyhow::ensure!(
            id_results.iter().all(|(score, _)| *score > 0.75),
            "top_n_ids threshold should remove low-scoring rows: {id_results:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn live_reinsert_same_document_id_removes_stale_vec0_candidates() -> anyhow::Result<()> {
        register_sqlite_vec_extension();

        let conn = Connection::open(
            "file:live_reinsert_same_document_id_removes_stale_vec0_candidates?mode=memory",
        )
        .await?;
        let model = TestEmbeddingModel;
        let vector_store: SqliteVectorStore<_, TestDocument> =
            SqliteVectorStore::new(conn, &model).await?;

        vector_store
            .add_rows(vec![row(
                "replace",
                "docs",
                "original near vector",
                vec![1.0, 0.0],
            )])
            .await?;
        vector_store
            .add_rows(vec![
                row("replace", "docs", "replacement far vector", vec![-1.0, 0.0]),
                row("fresh", "docs", "fresh near vector", vec![0.9, 0.1]),
            ])
            .await?;

        let index = vector_store.index(model);
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(1)
            .build();

        let results = index.top_n::<TestDocument>(req.clone()).await?;
        let ids = results
            .iter()
            .map(|(_, id, _)| id.as_str())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            ids.as_slice() == ["fresh"],
            "stale replaced vectors should not consume sqlite-vec candidates: {results:?}"
        );

        let id_results = index.top_n_ids(req).await?;
        let result_ids = id_results
            .iter()
            .map(|(_, id)| id.as_str())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            result_ids.as_slice() == ["fresh"],
            "top_n_ids should not return or be starved by stale replaced vectors: {id_results:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn live_multiple_embeddings_per_document_use_best_embedding() -> anyhow::Result<()> {
        let multi_document = TestDocument {
            id: "multi".to_string(),
            category: "docs".to_string(),
            title: "multi-vector document".to_string(),
        };
        let index = live_test_index(
            "live_multiple_embeddings_per_document_use_best_embedding",
            vec![
                (
                    multi_document.clone(),
                    OneOrMany::many(vec![
                        Embedding {
                            document: "far chunk".to_string(),
                            vec: vec![-1.0, 0.0],
                        },
                        Embedding {
                            document: "exact chunk".to_string(),
                            vec: vec![1.0, 0.0],
                        },
                    ])?,
                ),
                row("single", "docs", "single close chunk", vec![0.8, 0.6]),
            ],
        )
        .await?;

        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(2)
            .build();
        let results = index.top_n::<TestDocument>(req.clone()).await?;
        let ids = results
            .iter()
            .map(|(_, id, _)| id.as_str())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            ids.as_slice() == ["multi", "single"],
            "top_n should return each document once using its best embedding: {results:?}"
        );

        let id_results = index.top_n_ids(req).await?;
        let result_ids = id_results
            .iter()
            .map(|(_, id)| id.as_str())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            result_ids.as_slice() == ["multi", "single"],
            "top_n_ids should return each document once using its best embedding: {id_results:?}"
        );

        let threshold_req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(2)
            .threshold(1.0)
            .build();
        let threshold_results = index.top_n::<TestDocument>(threshold_req.clone()).await?;
        let threshold_ids = threshold_results
            .iter()
            .map(|(_, id, _)| id.as_str())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            threshold_ids.as_slice() == ["multi"],
            "threshold should include scores equal to the minimum and filter lower scores: {threshold_results:?}"
        );

        let threshold_id_results = index.top_n_ids(threshold_req).await?;
        let threshold_result_ids = threshold_id_results
            .iter()
            .map(|(_, id)| id.as_str())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            threshold_result_ids.as_slice() == ["multi"],
            "top_n_ids threshold should include scores equal to the minimum: {threshold_id_results:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn live_common_sqlite_text_types_round_trip_in_top_n() -> anyhow::Result<()> {
        let index = live_common_type_test_index(
            "live_common_sqlite_text_types_round_trip_in_top_n",
            vec![common_type_row(
                "common",
                "varchar name",
                "clob notes",
                7,
                vec![1.0, 0.0],
            )],
        )
        .await?;

        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(1)
            .build();
        let results = index.top_n::<CommonTypeDocument>(req).await?;

        let Some((_, id, doc)) = results.first() else {
            anyhow::bail!("expected common type document result");
        };
        anyhow::ensure!(id == "common", "unexpected id: {id}");
        anyhow::ensure!(
            doc.name == "varchar name",
            "VARCHAR value should round-trip: {doc:?}"
        );
        anyhow::ensure!(
            doc.notes == "clob notes",
            "CLOB value should round-trip: {doc:?}"
        );
        anyhow::ensure!(doc.rank == 7, "NUMERIC value should round-trip: {doc:?}");

        Ok(())
    }

    #[tokio::test]
    async fn live_text_affinity_metadata_filters_during_candidate_search() -> anyhow::Result<()> {
        let index = live_common_type_test_index(
            "live_text_affinity_metadata_filters_during_candidate_search",
            vec![
                common_type_row("nearest", "misc", "nearest excluded", 1, vec![1.0, 0.0]),
                common_type_row("docs", "docs", "docs match", 2, vec![0.0, 1.0]),
            ],
        )
        .await?;

        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(1)
            .filter(SqliteSearchFilter::eq("name", serde_json::json!("docs")))
            .build();

        let results = index.top_n::<CommonTypeDocument>(req.clone()).await?;
        let ids = results
            .iter()
            .map(|(_, id, _)| id.as_str())
            .collect::<Vec<_>>();

        anyhow::ensure!(
            ids.as_slice() == ["docs"],
            "VARCHAR metadata filters should constrain sqlite-vec candidate search: {results:?}"
        );

        let id_results = index.top_n_ids(req).await?;
        let result_ids = id_results
            .iter()
            .map(|(_, id)| id.as_str())
            .collect::<Vec<_>>();

        anyhow::ensure!(
            result_ids.as_slice() == ["docs"],
            "top_n_ids should use VARCHAR metadata filters during candidate search: {id_results:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn live_l2_metric_is_consistent() -> anyhow::Result<()> {
        let index = live_test_index_with_metric(
            "live_l2_metric_is_consistent",
            vec![
                row("exact", "docs", "exact match", vec![1.0, 0.0]),
                row("l2-close", "docs", "l2 close match", vec![1.0, 1.0]),
                row(
                    "same-direction-far",
                    "docs",
                    "same direction far away",
                    vec![10.0, 0.0],
                ),
            ],
            SqliteDistanceMetric::L2,
        )
        .await?;

        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(2)
            .threshold(-2.0)
            .build();

        let results = index.top_n::<TestDocument>(req.clone()).await?;
        let ids = results
            .iter()
            .map(|(_, id, _)| id.as_str())
            .collect::<Vec<_>>();
        let exact_score = results
            .iter()
            .find(|(_, id, _)| id == "exact")
            .map(|(score, _, _)| *score);
        let close_score = results
            .iter()
            .find(|(_, id, _)| id == "l2-close")
            .map(|(score, _, _)| *score);

        anyhow::ensure!(
            ids.as_slice() == ["exact", "l2-close"],
            "L2 search should return the nearest L2 candidates: {results:?}"
        );
        anyhow::ensure!(
            exact_score
                .zip(close_score)
                .is_some_and(|(exact, close)| exact > close && close > -2.0),
            "expected L2 scores to be ordered and thresholded: {results:?}"
        );
        anyhow::ensure!(
            results.iter().all(|(score, _, _)| *score > -2.0),
            "threshold should be applied to L2 scores: {results:?}"
        );

        let id_results = index.top_n_ids(req).await?;
        let result_ids = id_results
            .iter()
            .map(|(_, id)| id.as_str())
            .collect::<Vec<_>>();

        anyhow::ensure!(
            result_ids.as_slice() == ["exact", "l2-close"],
            "top_n_ids should use the same L2 metric: {id_results:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn live_indexed_filter_is_applied_during_candidate_search() -> anyhow::Result<()> {
        let index = live_test_index(
            "live_indexed_filter_is_applied_during_candidate_search",
            vec![
                row(
                    "nearest",
                    "misc",
                    "nearest excluded category",
                    vec![1.0, 0.0],
                ),
                row("docs", "docs", "docs match", vec![0.0, 1.0]),
            ],
        )
        .await?;

        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(1)
            .filter(SqliteSearchFilter::eq(
                "category",
                serde_json::json!("docs"),
            ))
            .build();

        let results = index.top_n::<TestDocument>(req.clone()).await?;
        let ids = results
            .iter()
            .map(|(_, id, _)| id.as_str())
            .collect::<Vec<_>>();

        anyhow::ensure!(
            ids.as_slice() == ["docs"],
            "indexed filters should constrain sqlite-vec candidate search: {results:?}"
        );

        let id_results = index.top_n_ids(req).await?;
        let result_ids = id_results
            .iter()
            .map(|(_, id)| id.as_str())
            .collect::<Vec<_>>();

        anyhow::ensure!(
            result_ids.as_slice() == ["docs"],
            "top_n_ids should use indexed filters during candidate search: {id_results:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn live_negated_eq_filter_is_applied_during_candidate_search() -> anyhow::Result<()> {
        let index = live_test_index(
            "live_negated_eq_filter_is_applied_during_candidate_search",
            vec![
                row(
                    "nearest",
                    "misc",
                    "nearest excluded category",
                    vec![1.0, 0.0],
                ),
                row("docs", "docs", "docs match", vec![0.0, 1.0]),
            ],
        )
        .await?;

        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(1)
            .filter(SqliteSearchFilter::eq("category", serde_json::json!("misc")).not())
            .build();

        let results = index.top_n::<TestDocument>(req.clone()).await?;
        let ids = results
            .iter()
            .map(|(_, id, _)| id.as_str())
            .collect::<Vec<_>>();

        anyhow::ensure!(
            ids.as_slice() == ["docs"],
            "negated filters should constrain sqlite-vec candidate search: {results:?}"
        );

        let id_results = index.top_n_ids(req).await?;
        let result_ids = id_results
            .iter()
            .map(|(_, id)| id.as_str())
            .collect::<Vec<_>>();

        anyhow::ensure!(
            result_ids.as_slice() == ["docs"],
            "top_n_ids should use negated filters during candidate search: {id_results:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn live_top_n_reads_id_by_column_name_not_schema_position() -> anyhow::Result<()> {
        register_sqlite_vec_extension();

        let conn = Connection::open(
            "file:live_top_n_reads_id_by_column_name_not_schema_position?mode=memory",
        )
        .await?;
        let model = TestEmbeddingModel;
        let vector_store: SqliteVectorStore<_, ReorderedIdDocument> =
            SqliteVectorStore::new(conn, &model).await?;

        vector_store
            .add_rows(vec![
                reordered_id_row("winner", "winner title", "docs", vec![1.0, 0.0]),
                reordered_id_row("other", "other title", "docs", vec![0.0, 1.0]),
            ])
            .await?;

        let index = vector_store.index(model);
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(1)
            .build();

        let results = index.top_n::<ReorderedIdDocument>(req.clone()).await?;
        let Some((_, id, doc)) = results.first() else {
            anyhow::bail!("expected reordered-id result");
        };
        anyhow::ensure!(
            id == "winner",
            "top_n should return the id column, not the first schema column: {results:?}"
        );
        anyhow::ensure!(
            doc.id == "winner" && doc.title == "winner title",
            "document columns should still deserialize in schema order: {doc:?}"
        );

        let id_results = index.top_n_ids(req).await?;
        anyhow::ensure!(
            id_results.first().map(|(_, id)| id.as_str()) == Some("winner"),
            "top_n_ids should agree with top_n id handling: {id_results:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn live_typed_columns_round_trip_and_filter_during_candidate_search() -> anyhow::Result<()>
    {
        let index = live_typed_test_index(
            "live_typed_columns_round_trip_and_filter_during_candidate_search",
            vec![
                typed_row(
                    1,
                    "misc",
                    100,
                    0.99,
                    true,
                    "nearest excluded by typed metadata",
                    vec![1.0, 0.0],
                ),
                typed_row(2, "docs", 5, 0.95, true, "typed docs match", vec![0.0, 1.0]),
                typed_row(
                    3,
                    "docs",
                    5,
                    0.97,
                    false,
                    "unpublished docs match",
                    vec![0.0, 0.9],
                ),
            ],
        )
        .await?;

        let filter = SqliteSearchFilter::lt("priority", serde_json::json!(10))
            .and(SqliteSearchFilter::gt("rating", serde_json::json!(0.9)))
            .and(SqliteSearchFilter::eq("published", serde_json::json!(true)));
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(1)
            .filter(filter)
            .build();

        let results = index.top_n::<TypedTestDocument>(req.clone()).await?;
        anyhow::ensure!(
            results.len() == 1,
            "expected one typed document result: {results:?}"
        );

        let Some((_, id, doc)) = results.first() else {
            anyhow::bail!("expected one typed document result");
        };
        anyhow::ensure!(id == "2", "expected integer id to be returned as string");
        anyhow::ensure!(doc.id == 2, "typed integer id should round-trip: {doc:?}");
        anyhow::ensure!(
            doc.priority == 5,
            "typed integer field should round-trip: {doc:?}"
        );
        anyhow::ensure!(
            (doc.rating - 0.95).abs() < f64::EPSILON,
            "typed float field should round-trip: {doc:?}"
        );
        anyhow::ensure!(
            doc.published,
            "typed boolean field should round-trip: {doc:?}"
        );

        let id_results = index.top_n_ids(req).await?;
        let result_ids = id_results
            .iter()
            .map(|(_, id)| id.as_str())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            result_ids.as_slice() == ["2"],
            "top_n_ids should use the same typed metadata filters: {id_results:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn live_boolean_range_filter_is_rejected() -> anyhow::Result<()> {
        let index = live_typed_test_index(
            "live_boolean_range_filter_is_rejected",
            vec![
                typed_row(
                    1,
                    "misc",
                    1,
                    0.5,
                    false,
                    "nearest unpublished doc",
                    vec![1.0, 0.0],
                ),
                typed_row(2, "docs", 2, 0.7, true, "published doc", vec![0.0, 1.0]),
            ],
        )
        .await?;

        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(2)
            .filter(SqliteSearchFilter::gt(
                "published",
                serde_json::json!(false),
            ))
            .build();

        ensure_vector_store_filter_error(
            index.top_n::<TypedTestDocument>(req.clone()).await,
            "top_n boolean range filter",
        )?;
        ensure_vector_store_filter_error(
            index.top_n_ids(req).await,
            "top_n_ids boolean range filter",
        )?;

        Ok(())
    }

    #[tokio::test]
    async fn live_matches_exact_oracle_for_metrics_filters_and_thresholds() -> anyhow::Result<()> {
        let query = vec![1.0, 0.0];
        let rows = oracle_test_rows();
        let filter = SqliteSearchFilter::eq("category", serde_json::json!("docs"))
            .and(SqliteSearchFilter::lt("priority", serde_json::json!(10)))
            .and(SqliteSearchFilter::gt("rating", serde_json::json!(0.8)))
            .and(SqliteSearchFilter::eq("published", serde_json::json!(true)));

        for distance_metric in [
            SqliteDistanceMetric::Cosine,
            SqliteDistanceMetric::L2,
            SqliteDistanceMetric::L1,
        ] {
            let threshold = oracle_threshold(distance_metric);
            let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
                .query("needle")
                .samples(u64::try_from(rows.len())?)
                .threshold(threshold)
                .filter(filter.clone())
                .build();
            let expected = exact_oracle_results(
                &rows,
                &query,
                distance_metric,
                threshold,
                rows.len(),
                |row| {
                    row.category == "docs" && row.priority < 10 && row.rating > 0.8 && row.published
                },
            )?;
            let test_name =
                format!("live_matches_exact_oracle_for_{distance_metric:?}").to_ascii_lowercase();
            let index = live_typed_test_index_with_metric(
                &test_name,
                sqlite_oracle_rows(&rows),
                distance_metric,
            )
            .await?;

            let results = index.top_n::<TypedTestDocument>(req.clone()).await?;
            let scored_ids = results
                .iter()
                .map(|(score, id, doc)| {
                    anyhow::ensure!(
                        id == &doc.id.to_string(),
                        "top_n returned mismatched id and document: id={id}, doc={doc:?}"
                    );
                    Ok((*score, id.clone()))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            assert_scored_ids_match(&scored_ids, &expected, distance_metric, "top_n")?;

            let id_results = index.top_n_ids(req).await?;
            assert_scored_ids_match(&id_results, &expected, distance_metric, "top_n_ids")?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn live_or_filter_is_rejected() -> anyhow::Result<()> {
        let index = live_test_index(
            "live_or_filter_is_rejected",
            vec![
                row(
                    "nearest",
                    "misc",
                    "nearest excluded category",
                    vec![1.0, 0.0],
                ),
                row("archived", "archive", "far archive match", vec![-1.0, 0.0]),
                row("docs", "docs", "far docs match", vec![0.0, 1.0]),
            ],
        )
        .await?;

        let filter = SqliteSearchFilter::eq("category", serde_json::json!("docs")).or(
            SqliteSearchFilter::eq("category", serde_json::json!("archive")),
        );

        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(1)
            .filter(filter)
            .build();

        ensure_vector_store_filter_error(
            index.top_n::<TestDocument>(req.clone()).await,
            "top_n OR filter",
        )?;
        ensure_vector_store_filter_error(index.top_n_ids(req).await, "top_n_ids OR filter")?;

        Ok(())
    }

    type SqliteExtensionFn =
        unsafe extern "C" fn(*mut sqlite3, *mut *mut c_char, *const sqlite3_api_routines) -> i32;

    fn register_sqlite_vec_extension() {
        static REGISTER_SQLITE_VEC: Once = Once::new();

        REGISTER_SQLITE_VEC.call_once(|| unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute::<*const (), SqliteExtensionFn>(
                sqlite3_vec_init as *const (),
            )));
        });
    }

    async fn live_test_index(
        name: &str,
        rows: Vec<(TestDocument, OneOrMany<Embedding>)>,
    ) -> anyhow::Result<SqliteVectorIndex<TestEmbeddingModel, TestDocument>> {
        live_test_index_with_metric(name, rows, SqliteDistanceMetric::Cosine).await
    }

    async fn live_test_index_with_metric(
        name: &str,
        rows: Vec<(TestDocument, OneOrMany<Embedding>)>,
        distance_metric: SqliteDistanceMetric,
    ) -> anyhow::Result<SqliteVectorIndex<TestEmbeddingModel, TestDocument>> {
        register_sqlite_vec_extension();

        let conn = Connection::open(format!("file:{name}?mode=memory")).await?;
        let model = TestEmbeddingModel;
        let vector_store =
            SqliteVectorStore::with_distance_metric(conn, &model, distance_metric).await?;

        vector_store.add_rows(rows).await?;

        Ok(vector_store.index(model))
    }

    async fn live_typed_test_index(
        name: &str,
        rows: Vec<(TypedTestDocument, OneOrMany<Embedding>)>,
    ) -> anyhow::Result<SqliteVectorIndex<TestEmbeddingModel, TypedTestDocument>> {
        live_typed_test_index_with_metric(name, rows, SqliteDistanceMetric::Cosine).await
    }

    async fn live_typed_test_index_with_metric(
        name: &str,
        rows: Vec<(TypedTestDocument, OneOrMany<Embedding>)>,
        distance_metric: SqliteDistanceMetric,
    ) -> anyhow::Result<SqliteVectorIndex<TestEmbeddingModel, TypedTestDocument>> {
        register_sqlite_vec_extension();

        let conn = Connection::open(format!("file:{name}?mode=memory")).await?;
        let model = TestEmbeddingModel;
        let vector_store: SqliteVectorStore<_, TypedTestDocument> =
            SqliteVectorStore::with_distance_metric(conn, &model, distance_metric).await?;

        vector_store.add_rows(rows).await?;

        Ok(vector_store.index(model))
    }

    async fn live_common_type_test_index(
        name: &str,
        rows: Vec<(CommonTypeDocument, OneOrMany<Embedding>)>,
    ) -> anyhow::Result<SqliteVectorIndex<TestEmbeddingModel, CommonTypeDocument>> {
        register_sqlite_vec_extension();

        let conn = Connection::open(format!("file:{name}?mode=memory")).await?;
        let model = TestEmbeddingModel;
        let vector_store: SqliteVectorStore<_, CommonTypeDocument> =
            SqliteVectorStore::new(conn, &model).await?;

        vector_store.add_rows(rows).await?;

        Ok(vector_store.index(model))
    }

    fn row(
        id: impl Into<String>,
        category: impl Into<String>,
        title: impl Into<String>,
        vec: Vec<f64>,
    ) -> (TestDocument, OneOrMany<Embedding>) {
        let document = TestDocument {
            id: id.into(),
            category: category.into(),
            title: title.into(),
        };

        (
            document.clone(),
            OneOrMany::one(Embedding {
                document: document.title,
                vec,
            }),
        )
    }

    fn common_type_row(
        id: impl Into<String>,
        name: impl Into<String>,
        notes: impl Into<String>,
        rank: i64,
        vec: Vec<f64>,
    ) -> (CommonTypeDocument, OneOrMany<Embedding>) {
        let document = CommonTypeDocument {
            id: id.into(),
            name: name.into(),
            notes: notes.into(),
            rank,
        };

        (
            document.clone(),
            OneOrMany::one(Embedding {
                document: document.name.clone(),
                vec,
            }),
        )
    }

    fn reordered_id_row(
        id: impl Into<String>,
        title: impl Into<String>,
        category: impl Into<String>,
        vec: Vec<f64>,
    ) -> (ReorderedIdDocument, OneOrMany<Embedding>) {
        let document = ReorderedIdDocument {
            title: title.into(),
            id: id.into(),
            category: category.into(),
        };

        (
            document.clone(),
            OneOrMany::one(Embedding {
                document: document.title.clone(),
                vec,
            }),
        )
    }

    fn typed_row(
        id: i64,
        category: impl Into<String>,
        priority: i64,
        rating: f64,
        published: bool,
        title: impl Into<String>,
        vec: Vec<f64>,
    ) -> (TypedTestDocument, OneOrMany<Embedding>) {
        let document = TypedTestDocument {
            id,
            category: category.into(),
            priority,
            rating,
            published,
            title: title.into(),
        };

        (
            document.clone(),
            OneOrMany::one(Embedding {
                document: document.title,
                vec,
            }),
        )
    }

    #[derive(Clone, Debug)]
    struct OracleRow {
        document: TypedTestDocument,
        embedding: Vec<f64>,
    }

    #[derive(Debug)]
    struct ExpectedScoredId {
        id: String,
        score: f64,
    }

    fn oracle_test_rows() -> Vec<OracleRow> {
        vec![
            oracle_row(1, "docs", 1, 0.95, true, "exact match", vec![1.0, 0.0]),
            oracle_row(2, "docs", 2, 0.90, true, "close match", vec![0.8, 0.6]),
            oracle_row(3, "docs", 3, 0.81, true, "borderline match", vec![0.5, 0.5]),
            oracle_row(
                4,
                "docs",
                4,
                0.70,
                true,
                "filtered by rating",
                vec![0.95, 0.05],
            ),
            oracle_row(
                5,
                "docs",
                15,
                0.99,
                true,
                "filtered by priority",
                vec![1.0, 0.0],
            ),
            oracle_row(
                6,
                "docs",
                5,
                0.99,
                false,
                "filtered by published",
                vec![1.0, 0.0],
            ),
            oracle_row(
                7,
                "misc",
                1,
                0.99,
                true,
                "filtered by category",
                vec![1.0, 0.0],
            ),
            oracle_row(8, "docs", 5, 0.95, true, "far match", vec![0.0, 1.0]),
        ]
    }

    fn oracle_row(
        id: i64,
        category: impl Into<String>,
        priority: i64,
        rating: f64,
        published: bool,
        title: impl Into<String>,
        embedding: Vec<f64>,
    ) -> OracleRow {
        OracleRow {
            document: TypedTestDocument {
                id,
                category: category.into(),
                priority,
                rating,
                published,
                title: title.into(),
            },
            embedding,
        }
    }

    fn sqlite_oracle_rows(rows: &[OracleRow]) -> Vec<(TypedTestDocument, OneOrMany<Embedding>)> {
        rows.iter()
            .map(|row| {
                (
                    row.document.clone(),
                    OneOrMany::one(Embedding {
                        document: row.document.title.clone(),
                        vec: row.embedding.clone(),
                    }),
                )
            })
            .collect()
    }

    fn oracle_threshold(distance_metric: SqliteDistanceMetric) -> f64 {
        match distance_metric {
            SqliteDistanceMetric::Cosine => 0.75,
            SqliteDistanceMetric::L2 => -0.8,
            SqliteDistanceMetric::L1 => -0.9,
        }
    }

    fn exact_oracle_results(
        rows: &[OracleRow],
        query: &[f64],
        distance_metric: SqliteDistanceMetric,
        threshold: f64,
        samples: usize,
        filter: impl Fn(&TypedTestDocument) -> bool,
    ) -> anyhow::Result<Vec<ExpectedScoredId>> {
        let mut expected = Vec::new();
        for row in rows {
            if !filter(&row.document) {
                continue;
            }

            let score = oracle_score(distance_metric, query, &row.embedding)?;
            if score > threshold {
                expected.push(ExpectedScoredId {
                    id: row.document.id.to_string(),
                    score,
                });
            }
        }

        sort_expected_scores(&mut expected);
        expected.truncate(samples);
        Ok(expected)
    }

    fn sort_expected_scores(expected: &mut [ExpectedScoredId]) {
        expected.sort_by(|lhs, rhs| {
            rhs.score
                .partial_cmp(&lhs.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| lhs.id.cmp(&rhs.id))
        });
    }

    fn oracle_score(
        distance_metric: SqliteDistanceMetric,
        query: &[f64],
        embedding: &[f64],
    ) -> anyhow::Result<f64> {
        anyhow::ensure!(
            query.len() == embedding.len(),
            "query and embedding dimensions differ: query={}, embedding={}",
            query.len(),
            embedding.len()
        );

        let query = query.iter().map(|value| *value as f32).collect::<Vec<_>>();
        let embedding = embedding
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();

        let score = match distance_metric {
            SqliteDistanceMetric::Cosine => {
                let dot = query
                    .iter()
                    .zip(&embedding)
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum::<f32>();
                let query_norm = query.iter().map(|value| value * value).sum::<f32>().sqrt();
                let embedding_norm = embedding
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt();
                anyhow::ensure!(
                    query_norm > 0.0 && embedding_norm > 0.0,
                    "cosine oracle requires non-zero vectors"
                );
                dot / (query_norm * embedding_norm)
            }
            SqliteDistanceMetric::L2 => -query
                .iter()
                .zip(&embedding)
                .map(|(lhs, rhs)| {
                    let delta = lhs - rhs;
                    delta * delta
                })
                .sum::<f32>()
                .sqrt(),
            SqliteDistanceMetric::L1 => -query
                .iter()
                .zip(&embedding)
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .sum::<f32>(),
        };

        Ok(f64::from(score))
    }

    fn assert_scored_ids_match(
        actual: &[(f64, String)],
        expected: &[ExpectedScoredId],
        distance_metric: SqliteDistanceMetric,
        context: &str,
    ) -> anyhow::Result<()> {
        const SCORE_EPSILON: f64 = 1e-5;

        let actual_ids = actual.iter().map(|(_, id)| id.as_str()).collect::<Vec<_>>();
        let expected_ids = expected
            .iter()
            .map(|expected| expected.id.as_str())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            actual_ids == expected_ids,
            "{context} ids for {distance_metric:?} did not match exact oracle: actual={actual:?}, expected={expected:?}"
        );

        for ((actual_score, actual_id), expected) in actual.iter().zip(expected) {
            anyhow::ensure!(
                (actual_score - expected.score).abs() <= SCORE_EPSILON,
                "{context} score for {distance_metric:?} id `{actual_id}` did not match exact oracle: actual={actual_score}, expected={}",
                expected.score
            );
        }

        Ok(())
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    struct TestDocument {
        id: String,
        category: String,
        title: String,
    }

    impl SqliteVectorStoreTable for TestDocument {
        fn name() -> &'static str {
            "live_test_documents"
        }

        fn schema() -> Vec<Column> {
            vec![
                Column::new("id", "TEXT PRIMARY KEY"),
                Column::new("category", "TEXT").indexed(),
                Column::new("title", "TEXT"),
            ]
        }

        fn id(&self) -> String {
            self.id.clone()
        }

        fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
            vec![
                ("id", Box::new(self.id.clone())),
                ("category", Box::new(self.category.clone())),
                ("title", Box::new(self.title.clone())),
            ]
        }
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    struct ReorderedIdDocument {
        title: String,
        id: String,
        category: String,
    }

    impl SqliteVectorStoreTable for ReorderedIdDocument {
        fn name() -> &'static str {
            "live_reordered_id_test_documents"
        }

        fn schema() -> Vec<Column> {
            vec![
                Column::new("title", "TEXT"),
                Column::new("id", "TEXT PRIMARY KEY"),
                Column::new("category", "TEXT").indexed(),
            ]
        }

        fn id(&self) -> String {
            self.id.clone()
        }

        fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
            vec![
                ("title", Box::new(self.title.clone())),
                ("id", Box::new(self.id.clone())),
                ("category", Box::new(self.category.clone())),
            ]
        }
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    struct CommonTypeDocument {
        id: String,
        name: String,
        notes: String,
        rank: i64,
    }

    impl SqliteVectorStoreTable for CommonTypeDocument {
        fn name() -> &'static str {
            "live_common_type_test_documents"
        }

        fn schema() -> Vec<Column> {
            vec![
                Column::new("id", "TEXT PRIMARY KEY"),
                Column::new("name", "VARCHAR(255)").indexed(),
                Column::new("notes", "CLOB"),
                Column::new("rank", "NUMERIC"),
            ]
        }

        fn id(&self) -> String {
            self.id.clone()
        }

        fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
            vec![
                ("id", Box::new(self.id.clone())),
                ("name", Box::new(self.name.clone())),
                ("notes", Box::new(self.notes.clone())),
                ("rank", Box::new(self.rank)),
            ]
        }
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    struct TypedTestDocument {
        id: i64,
        category: String,
        priority: i64,
        rating: f64,
        published: bool,
        title: String,
    }

    impl SqliteVectorStoreTable for TypedTestDocument {
        fn name() -> &'static str {
            "live_typed_test_documents"
        }

        fn schema() -> Vec<Column> {
            vec![
                Column::new("id", "INTEGER PRIMARY KEY"),
                Column::new("category", "TEXT").indexed(),
                Column::new("priority", "INTEGER").indexed(),
                Column::new("rating", "FLOAT").indexed(),
                Column::new("published", "BOOLEAN").indexed(),
                Column::new("title", "TEXT"),
            ]
        }

        fn id(&self) -> String {
            self.id.to_string()
        }

        fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
            vec![
                ("id", Box::new(self.id)),
                ("category", Box::new(self.category.clone())),
                ("priority", Box::new(self.priority)),
                ("rating", Box::new(self.rating)),
                ("published", Box::new(self.published)),
                ("title", Box::new(self.title.clone())),
            ]
        }
    }

    #[derive(Clone)]
    struct TestEmbeddingModel;

    impl EmbeddingModel for TestEmbeddingModel {
        const MAX_DOCUMENTS: usize = 16;

        type Client = ();

        fn make(_: &Self::Client, _: impl Into<String>, _: Option<usize>) -> Self {
            Self
        }

        fn ndims(&self) -> usize {
            2
        }

        async fn embed_texts(
            &self,
            texts: impl IntoIterator<Item = String> + WasmCompatSend,
        ) -> Result<Vec<Embedding>, EmbeddingError> {
            Ok(texts
                .into_iter()
                .map(|text| Embedding {
                    document: text,
                    vec: vec![1.0, 0.0],
                })
                .collect())
        }
    }
}
