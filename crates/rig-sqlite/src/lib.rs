//! SQLite vector store integration for Rig.
//!
//! This crate provides [`SqliteVectorStore`] and [`SqliteVectorIndex`] for
//! storing embedded documents in SQLite with the `sqlite-vec` extension. Define
//! document table schemas by implementing [`SqliteVectorStoreTable`].
//!
//! The root `rig` facade re-exports this crate as `rig::sqlite` when the
//! `sqlite` feature is enabled.

use rig_core::Embed;
use rig_core::embeddings::{Embedding, EmbeddingModel};
use rig_core::vector_store::request::{FilterError, SearchFilter, VectorSearchRequest};
use rig_core::vector_store::{InsertDocuments, VectorStoreError, VectorStoreIndex};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use rusqlite::OptionalExtension;
use rusqlite::types::{Type, Value, ValueRef};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::RangeInclusive;
use tokio_rusqlite::Connection;
use tracing::{debug, info};

/// Maximum `k` accepted by a `sqlite-vec` `vec0` KNN query (`embedding MATCH ?
/// AND k = ?`). `sqlite-vec` enforces this as a hard `#define
/// SQLITE_VEC_VEC0_K_MAX 4096` and rejects larger values with
/// `"k value in knn query too large, ..."`. When more candidates than this are
/// required for an exact result, searches fall back to a brute-force scan that
/// ranks every row with the scalar `vec_distance_*` functions instead (same
/// exact result, no `k` cap).
const SQLITE_VEC_MAX_K: u64 = 4096;

/// Value that can be stored in a SQLite vector store document column.
///
/// Use [`serde_json::Value`] for columns declared as `JSON`.
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
    /// KNN candidate search. Filters on other document-table fields are applied
    /// after candidate search with an exhaustive candidate limit, which is
    /// correct but can be more expensive on large stores.
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

#[derive(Debug, thiserror::Error)]
enum SqliteInternalError {
    #[error(
        "SQLite vector table `{table_name}` uses {configured:?}, but {requested:?} was requested"
    )]
    DistanceMetricMismatch {
        table_name: String,
        requested: SqliteDistanceMetric,
        configured: SqliteDistanceMetric,
    },
    #[error("SQLite vector table `{0}` was created but is missing from sqlite_schema")]
    VectorTableMissingSchema(String),
    #[error("SQLite metadata column `{column_name}` has unsupported type `{column_type}`")]
    UnsupportedMetadataColumn {
        column_name: &'static str,
        column_type: &'static str,
    },
    #[error("SQLite vector table `{table_name}` is missing metadata column `{column_name} {}`", column_type.vec0_name())]
    MetadataSchemaMismatch {
        table_name: String,
        column_name: &'static str,
        column_type: SqliteMetadataType,
    },
    #[error("could not convert SQLite value type `{value_type:?}` for metadata column `{column_name} {}`", column_type.vec0_name())]
    MetadataValueError {
        column_name: &'static str,
        column_type: SqliteMetadataType,
        value_type: Type,
    },
    #[error("SQLite vector store table `{0}` is missing an `id` column")]
    MissingIdColumn(String),
    #[error(
        "could not convert SQLite column `{column_name}` with declared type `{column_type}`: {message}"
    )]
    ColumnValueError {
        column_name: &'static str,
        column_type: &'static str,
        message: String,
    },
}

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

fn sqlite_metadata_columns(
    schema: &[Column],
) -> Result<Vec<SqliteMetadataColumn>, VectorStoreError> {
    schema
        .iter()
        .filter(|column| column.indexed)
        .map(|column| {
            let metadata_type =
                SqliteMetadataType::from_column_type(column.col_type).ok_or_else(|| {
                    VectorStoreError::datastore(SqliteInternalError::UnsupportedMetadataColumn {
                        column_name: column.name,
                        column_type: column.col_type,
                    })
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
            SqliteInternalError::MetadataValueError {
                column_name: column.name,
                column_type: column.metadata_type,
                value_type: value.data_type(),
            },
        ))),
    }
}

/// A SQLite-backed vector store for documents of type `T`.
///
/// The store does not name an embedding model in its type; the model is only
/// consulted (for `ndims`) at construction, and the index built from it via
/// [`SqliteVectorStore::index`] holds the model behind an erased
/// the embedding model `M`.
#[derive(Clone)]
pub struct SqliteVectorStore<T> {
    conn: Connection,
    distance_metric: SqliteDistanceMetric,
    metadata_columns: Vec<SqliteMetadataColumn>,
    _phantom: PhantomData<T>,
}

impl<T> SqliteVectorStore<T>
where
    T: SqliteVectorStoreTable,
{
    async fn candidate_limit(
        &self,
        samples: u64,
        exhaustive: bool,
    ) -> Result<u64, VectorStoreError> {
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
            .map_err(VectorStoreError::datastore)?;

        let embedding_count = u64::try_from(embedding_count).unwrap_or(0);
        let document_count = u64::try_from(document_count).unwrap_or(0);

        if exhaustive {
            // Post-filters are applied after candidate search, so any candidate
            // can be discarded; only an exhaustive scan guarantees the requested
            // number of results survives filtering.
            Ok(embedding_count.max(samples))
        } else if embedding_count > document_count {
            // Some document owns multiple embeddings. After dedup-to-document
            // (keeping each document's best embedding), guaranteeing the exact
            // top-`samples` documents needs `samples + (extra embeddings)`
            // candidates: at most `embedding_count - document_count` higher-
            // ranked embeddings can collapse into already-seen documents. This
            // bound is tight (one fewer can drop the last document) and never
            // exceeds the total embedding count.
            Ok(samples
                .saturating_add(embedding_count - document_count)
                .min(embedding_count))
        } else {
            Ok(samples)
        }
    }
}

impl<T> SqliteVectorStore<T>
where
    T: SqliteVectorStoreTable + 'static,
{
    /// Creates a SQLite vector store using cosine similarity.
    pub async fn new(
        conn: Connection,
        embedding_model: &impl EmbeddingModel,
    ) -> Result<Self, VectorStoreError> {
        Self::with_distance_metric(conn, embedding_model, SqliteDistanceMetric::default()).await
    }

    /// Creates a SQLite vector store with the requested distance metric.
    ///
    /// The metric is written into the sqlite-vec virtual table definition so
    /// candidate search uses the same metric as thresholding, ordering, and the
    /// returned score values.
    pub async fn with_distance_metric(
        conn: Connection,
        embedding_model: &impl EmbeddingModel,
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
            .map_err(VectorStoreError::datastore)?;

        let schema_sql = embeddings_table_sql.ok_or_else(|| {
            VectorStoreError::datastore(SqliteInternalError::VectorTableMissingSchema(
                embeddings_table_name.clone(),
            ))
        })?;

        let configured = sqlite_distance_metric_from_schema(&schema_sql);
        if configured != distance_metric {
            return Err(VectorStoreError::datastore(
                SqliteInternalError::DistanceMetricMismatch {
                    table_name: embeddings_table_name,
                    requested: distance_metric,
                    configured,
                },
            ));
        }
        for column in metadata_columns_for_schema_check {
            if !sqlite_schema_contains_metadata_column(&schema_sql, &column) {
                return Err(VectorStoreError::datastore(
                    SqliteInternalError::MetadataSchemaMismatch {
                        table_name: embeddings_table_name.clone(),
                        column_name: column.name,
                        column_type: column.metadata_type,
                    },
                ));
            }
        }

        Ok(Self {
            conn,
            distance_metric,
            metadata_columns,
            _phantom: PhantomData,
        })
    }

    pub fn index<M: EmbeddingModel>(self, model: M) -> SqliteVectorIndex<T, M> {
        SqliteVectorIndex::new(model, self)
    }

    pub fn add_rows_with_txn(
        &self,
        txn: &rusqlite::Transaction<'_>,
        documents: &[(T, Vec<Embedding>)],
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

        for (doc, embeddings) in documents {
            debug!("Storing document with id {}", doc.id());

            let values = doc.column_values();
            let id_value = values
                .iter()
                .find(|(name, _)| *name == "id")
                .map_or_else(|| Value::Text(doc.id()), |(_, value)| value.to_sql_value());
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
                    vec.len()
                );
                txn.execute(&insert_embedding_map_sql, [last_id])?;
                let embedding_rowid = txn.last_insert_rowid();
                let mut params = Vec::with_capacity(2 + metadata_values.len());
                params.push(Value::Integer(embedding_rowid));
                params.push(Value::Blob(vec));
                params.extend(metadata_values.iter().cloned());
                stmt.execute(rusqlite::params_from_iter(params))?;
            }
        }

        Ok(last_id)
    }

    pub async fn add_rows(
        &self,
        documents: Vec<(T, Vec<Embedding>)>,
    ) -> Result<i64, VectorStoreError> {
        let cloned = self.clone();

        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;
                let result = cloned.add_rows_with_txn(&tx, &documents)?;
                tx.commit()?;

                Ok(result)
            })
            .await
            .map_err(VectorStoreError::datastore)
    }
}

impl<T> InsertDocuments for SqliteVectorStore<T>
where
    T: SqliteVectorStoreTable
        + serde::de::DeserializeOwned
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
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

/// Search filter for SQLite vector searches.
///
/// SQLite vector search applies simple indexed metadata comparisons and ranges
/// during sqlite-vec KNN candidate search when possible. Other supported
/// document-table expressions, including JSON expressions, `OR`, null checks,
/// `LIKE`, and `GLOB`, are applied after candidate search with an exhaustive
/// candidate limit so custom document columns can still be filtered correctly.
///
/// For hot scalar filters, prefer marking columns with [`Column::indexed`] so
/// they can be pushed into sqlite-vec metadata constraints instead of requiring
/// exhaustive candidate retrieval.
#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct SqliteSearchFilter {
    expr: SqliteSearchFilterExpr,
}

impl Default for SqliteSearchFilter {
    fn default() -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Noop,
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
    /// Matches every document; used by [`SqliteSearchFilter::default`].
    Noop,
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

#[derive(Debug, Default)]
struct SqliteRenderedFilters {
    native: Vec<SqliteRenderedFilter>,
    post: Vec<SqliteRenderedFilter>,
}

impl SqliteRenderedFilters {
    fn post_only(filter: SqliteRenderedFilter) -> Self {
        Self {
            native: Vec::new(),
            post: vec![filter],
        }
    }

    fn extend(&mut self, rhs: Self) {
        self.native.extend(rhs.native);
        self.post.extend(rhs.post);
    }

    fn has_post_filters(&self) -> bool {
        !self.post.is_empty()
    }
}

#[derive(Debug)]
struct SqliteRenderedFilter {
    condition: String,
    params: Vec<Value>,
}

impl SqliteRenderedFilter {
    fn combine(joiner: &str, lhs: Self, rhs: Self) -> Self {
        Self {
            condition: format!("({}) {joiner} ({})", lhs.condition, rhs.condition),
            params: lhs.params.into_iter().chain(rhs.params).collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SqliteDocumentValueMode {
    Sql,
    JsonText,
}

#[derive(Debug)]
struct SqliteQualifiedDocumentKey {
    expression: String,
    value_mode: SqliteDocumentValueMode,
    plain_column: Option<String>,
}

impl SqliteSearchFilter {
    fn cmp(key: impl AsRef<str>, op: SqliteComparisonOp, value: serde_json::Value) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Comparison {
                key: key.as_ref().to_string(),
                op,
                value,
            },
        }
    }

    fn pattern(key: String, op: SqlitePatternOp, pattern: impl Into<String>) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Pattern {
                key,
                op,
                pattern: pattern.into(),
            },
        }
    }

    fn null_check(key: String, negated: bool) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::NullCheck { key, negated },
        }
    }
}

impl SearchFilter for SqliteSearchFilter {
    type Value = serde_json::Value;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self::cmp(key, SqliteComparisonOp::Eq, value)
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self::cmp(key, SqliteComparisonOp::Gt, value)
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self::cmp(key, SqliteComparisonOp::Lt, value)
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
    /// Negates a filter.
    ///
    /// SQLite vector search lowers `NOT` over simple indexed metadata
    /// comparisons to native operators such as `!=`, `<=`, and `>=`. Broader
    /// negations are applied as document-table post-filters when their
    /// expressions can be lowered safely.
    pub fn not(self) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Not(Box::new(self.expr)),
        }
    }

    /// Tests whether a value is contained in the range.
    ///
    /// Non-boolean indexed metadata ranges are applied during sqlite-vec
    /// candidate search. Document-table ranges are applied after candidate
    /// search and may require exhaustive candidate retrieval.
    pub fn between<N>(key: impl Into<String>, range: RangeInclusive<N>) -> Self
    where
        N: Into<serde_json::Value>,
    {
        let key = key.into();
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
    pub fn is_null(key: impl Into<String>) -> Self {
        let key = key.into();
        Self::null_check(key, false)
    }

    pub fn is_not_null(key: impl Into<String>) -> Self {
        let key = key.into();
        Self::null_check(key, true)
    }

    /// Tests whether the value at `key` satisfies the glob pattern.
    ///
    /// sqlite-vec cannot enforce `GLOB` during candidate search, so this is
    /// applied as a document-table post-filter.
    pub fn glob(key: impl Into<String>, pattern: impl Into<String>) -> Self {
        let key = key.into();
        Self::pattern(key, SqlitePatternOp::Glob, pattern)
    }

    /// Tests whether the value at `key` satisfies the `LIKE` pattern.
    ///
    /// sqlite-vec cannot enforce `LIKE` during candidate search, so this is
    /// applied as a document-table post-filter.
    pub fn like(key: impl Into<String>, pattern: impl Into<String>) -> Self {
        let key = key.into();
        Self::pattern(key, SqlitePatternOp::Like, pattern)
    }
}

impl SqliteSearchFilter {
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
        let Some(metadata_column) = sqlite_native_metadata_column(key, metadata_columns) else {
            return Ok(SqliteRenderedFilters::post_only(
                Self::render_document_comparison(key, op, value, metadata_columns)?,
            ));
        };

        if !metadata_column.metadata_type.supports_native_comparison(op) {
            return Err(sqlite_unsupported_filter(format!(
                "`{key}` is a BOOLEAN metadata column, and sqlite-vec only supports `=` and `!=` filters for booleans"
            )));
        }

        Ok(SqliteRenderedFilters {
            native: vec![SqliteRenderedFilter {
                condition: format!("e.{key} {} ?", op.as_sql()),
                params: vec![sqlite_metadata_filter_param(metadata_column, value)?],
            }],
            post: Vec::new(),
        })
    }

    fn render_document_comparison(
        key: &str,
        op: SqliteComparisonOp,
        value: serde_json::Value,
        metadata_columns: &[SqliteMetadataColumn],
    ) -> Result<SqliteRenderedFilter, FilterError> {
        let key = sqlite_qualify_document_key(key)?;
        Ok(SqliteRenderedFilter {
            condition: format!("{} {} ?", key.expression, op.as_sql()),
            params: vec![sqlite_document_filter_param(&key, metadata_columns, value)?],
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
                let Some(metadata_column) = sqlite_native_metadata_column(key, metadata_columns)
                else {
                    return Ok(SqliteRenderedFilters::post_only(
                        self.render_document(metadata_columns)?,
                    ));
                };

                if metadata_column.metadata_type == SqliteMetadataType::Boolean {
                    return Err(sqlite_unsupported_filter(format!(
                        "`{key}` is a BOOLEAN metadata column, and sqlite-vec does not support range filters for booleans"
                    )));
                }

                Ok(SqliteRenderedFilters {
                    native: vec![SqliteRenderedFilter {
                        condition: format!("e.{key} >= ? AND e.{key} <= ?"),
                        params: vec![
                            sqlite_metadata_filter_param(metadata_column, lo.clone())?,
                            sqlite_metadata_filter_param(metadata_column, hi.clone())?,
                        ],
                    }],
                    post: Vec::new(),
                })
            }
            Self::Noop => Ok(SqliteRenderedFilters::default()),
            Self::Or(_, _) | Self::NullCheck { .. } | Self::Pattern { .. } => Ok(
                SqliteRenderedFilters::post_only(self.render_document(metadata_columns)?),
            ),
            Self::Not(expr) => expr.render_negated_split(metadata_columns),
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
            _ => {
                let rendered = self.render_document(metadata_columns)?;
                Ok(SqliteRenderedFilters::post_only(SqliteRenderedFilter {
                    condition: format!("NOT ({})", rendered.condition),
                    params: rendered.params,
                }))
            }
        }
    }

    fn render_document(
        &self,
        metadata_columns: &[SqliteMetadataColumn],
    ) -> Result<SqliteRenderedFilter, FilterError> {
        match self {
            Self::Comparison { key, op, value } => {
                Self::render_document_comparison(key, *op, value.clone(), metadata_columns)
            }
            Self::And(lhs, rhs) => Ok(SqliteRenderedFilter::combine(
                "AND",
                lhs.render_document(metadata_columns)?,
                rhs.render_document(metadata_columns)?,
            )),
            Self::Or(lhs, rhs) => Ok(SqliteRenderedFilter::combine(
                "OR",
                lhs.render_document(metadata_columns)?,
                rhs.render_document(metadata_columns)?,
            )),
            Self::Not(expr) => {
                let expr = expr.render_document(metadata_columns)?;
                Ok(SqliteRenderedFilter {
                    condition: format!("NOT ({})", expr.condition),
                    params: expr.params,
                })
            }
            Self::Between { key, lo, hi } => {
                let key = sqlite_qualify_document_key(key)?;
                Ok(SqliteRenderedFilter {
                    condition: format!("{} between ? and ?", key.expression),
                    params: vec![
                        sqlite_document_filter_param(&key, metadata_columns, lo.clone())?,
                        sqlite_document_filter_param(&key, metadata_columns, hi.clone())?,
                    ],
                })
            }
            Self::NullCheck { key, negated } => {
                let key = sqlite_qualify_document_key(key)?;
                let operator = if *negated { "is not null" } else { "is null" };
                Ok(SqliteRenderedFilter {
                    condition: format!("{} {operator}", key.expression),
                    params: Vec::new(),
                })
            }
            Self::Pattern { key, op, pattern } => {
                let key = sqlite_qualify_document_key(key)?;
                Ok(SqliteRenderedFilter {
                    condition: format!("{} {} ?", key.expression, op.as_sql()),
                    params: vec![Value::Text(pattern.clone())],
                })
            }
            // `Noop` matches every document, so it renders as a tautology when
            // composed under `Or`/`Not`/`And` on the document path.
            Self::Noop => Ok(SqliteRenderedFilter {
                condition: "1 = 1".to_owned(),
                params: Vec::new(),
            }),
        }
    }
}

fn sqlite_native_metadata_column<'a>(
    key: &str,
    metadata_columns: &'a [SqliteMetadataColumn],
) -> Option<&'a SqliteMetadataColumn> {
    if !sqlite_is_plain_identifier(key) {
        return None;
    }

    metadata_columns.iter().find(|column| column.name == key)
}

fn sqlite_is_plain_identifier(key: &str) -> bool {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return false;
    };

    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn sqlite_leading_identifier_len(key: &str) -> Option<usize> {
    let mut chars = key.char_indices();
    let (_, first) = chars.next()?;
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return None;
    }

    let mut end = first.len_utf8();
    for (index, c) in chars {
        if c == '_' || c.is_ascii_alphanumeric() {
            end = index + c.len_utf8();
        } else {
            break;
        }
    }

    Some(end)
}

fn sqlite_unsupported_filter(reason: impl Into<String>) -> FilterError {
    FilterError::TypeError(format!(
        "SQLite filter cannot be safely lowered; {}",
        reason.into()
    ))
}

fn sqlite_json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn sqlite_metadata_filter_type_error(
    column: &SqliteMetadataColumn,
    value: &serde_json::Value,
    expected: &str,
) -> FilterError {
    sqlite_unsupported_filter(format!(
        "`{}` is a {} metadata column and requires {expected}; got {}",
        column.name,
        column.metadata_type.vec0_name(),
        sqlite_json_type_name(value)
    ))
}

fn sqlite_metadata_filter_param(
    column: &SqliteMetadataColumn,
    value: serde_json::Value,
) -> Result<Value, FilterError> {
    let expected = match column.metadata_type {
        SqliteMetadataType::Text => "a string filter value",
        SqliteMetadataType::Integer => "an integer filter value",
        SqliteMetadataType::Float => "a finite number filter value",
        SqliteMetadataType::Boolean => "a boolean filter value",
    };

    match (column.metadata_type, value) {
        (SqliteMetadataType::Text, serde_json::Value::String(value)) => Ok(Value::Text(value)),
        (SqliteMetadataType::Integer, serde_json::Value::Number(number)) => {
            if let Some(value) = number.as_i64() {
                Ok(Value::Integer(value))
            } else if let Some(value) = number.as_u64() {
                i64::try_from(value).map(Value::Integer).map_err(|_| {
                    FilterError::TypeError(format!(
                        "SQLite integer filter value `{number}` exceeds i64::MAX"
                    ))
                })
            } else {
                Err(sqlite_metadata_filter_type_error(
                    column,
                    &serde_json::Value::Number(number),
                    expected,
                ))
            }
        }
        (SqliteMetadataType::Float, serde_json::Value::Number(number)) => {
            number.as_f64().map(Value::Real).ok_or_else(|| {
                sqlite_metadata_filter_type_error(
                    column,
                    &serde_json::Value::Number(number),
                    expected,
                )
            })
        }
        (SqliteMetadataType::Boolean, serde_json::Value::Bool(value)) => {
            Ok(Value::Integer(value as i64))
        }
        (_, value) => Err(sqlite_metadata_filter_type_error(column, &value, expected)),
    }
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

fn sqlite_qualify_document_key(key: &str) -> Result<SqliteQualifiedDocumentKey, FilterError> {
    if let Some(key_without_alias) = key.strip_prefix("d.") {
        if sqlite_is_plain_identifier(key_without_alias) {
            return Ok(SqliteQualifiedDocumentKey {
                expression: key.to_string(),
                value_mode: SqliteDocumentValueMode::Sql,
                plain_column: Some(key_without_alias.to_string()),
            });
        }

        if let Some(value_mode) = sqlite_json_operator_value_mode(key_without_alias) {
            return Ok(SqliteQualifiedDocumentKey {
                expression: key.to_string(),
                value_mode,
                plain_column: None,
            });
        }

        return Err(sqlite_unsupported_filter(format!(
            "`{key}` is not a supported SQLite document filter expression"
        )));
    }

    if sqlite_is_plain_identifier(key) {
        return Ok(SqliteQualifiedDocumentKey {
            expression: format!("d.{key}"),
            value_mode: SqliteDocumentValueMode::Sql,
            plain_column: Some(key.to_string()),
        });
    }

    if let Some(value_mode) = sqlite_json_operator_value_mode(key) {
        return Ok(SqliteQualifiedDocumentKey {
            expression: format!("d.{key}"),
            value_mode,
            plain_column: None,
        });
    }

    Err(sqlite_unsupported_filter(format!(
        "`{key}` is not a supported SQLite document filter expression"
    )))
}

fn sqlite_document_filter_param(
    key: &SqliteQualifiedDocumentKey,
    metadata_columns: &[SqliteMetadataColumn],
    value: serde_json::Value,
) -> Result<Value, FilterError> {
    match key.value_mode {
        SqliteDocumentValueMode::Sql => {
            if let Some(column_name) = key.plain_column.as_deref()
                && let Some(metadata_column) = metadata_columns
                    .iter()
                    .find(|column| column.name == column_name)
            {
                return sqlite_metadata_filter_param(metadata_column, value);
            }

            sqlite_filter_param(value)
        }
        SqliteDocumentValueMode::JsonText => serde_json::to_string(&value)
            .map(Value::Text)
            .map_err(|e| FilterError::Serialization(e.to_string())),
    }
}

fn sqlite_json_operator_value_mode(expr: &str) -> Option<SqliteDocumentValueMode> {
    let mut index = sqlite_leading_identifier_len(expr)?;

    if index == expr.len() {
        return None;
    }

    let mut value_mode = None;
    while index < expr.len() {
        let remaining = &expr[index..];
        let (operator_len, next_value_mode) = if remaining.starts_with("->>") {
            (3, SqliteDocumentValueMode::Sql)
        } else if remaining.starts_with("->") {
            (2, SqliteDocumentValueMode::JsonText)
        } else {
            return None;
        };
        value_mode = Some(next_value_mode);
        index += operator_len;

        let operand_len = sqlite_json_operator_operand_len(&expr[index..])?;
        index += operand_len;
    }

    value_mode
}

fn sqlite_json_operator_operand_len(operand: &str) -> Option<usize> {
    if operand.is_empty() {
        return None;
    }

    if let Some(operand) = operand.strip_prefix('\'') {
        let closing_quote = operand.find('\'')?;
        let literal = &operand[..closing_quote];
        if literal.chars().any(char::is_control) {
            return None;
        }

        return Some(closing_quote + 2);
    }

    let mut chars = operand.char_indices();
    let mut end = 0;
    if let Some((_, '-')) = chars.clone().next() {
        end = 1;
        chars.next();
    }

    let mut has_digit = false;
    for (index, c) in chars {
        if c.is_ascii_digit() {
            has_digit = true;
            end = index + c.len_utf8();
        } else {
            break;
        }
    }

    has_digit.then_some(end)
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
/// use rig_reqwest::prelude::*;
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
/// let vector_store: SqliteVectorStore<Document> = SqliteVectorStore::with_distance_metric(
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
///
/// The store is generic over its embedding model `M`, which is fixed for the
/// store's lifetime: an index populated under one model is only meaningful under
/// that same model.
pub struct SqliteVectorIndex<T, M> {
    store: SqliteVectorStore<T>,
    embedding_model: M,
}

impl<T, M: EmbeddingModel> SqliteVectorIndex<T, M>
where
    T: SqliteVectorStoreTable,
{
    pub fn new(embedding_model: M, store: SqliteVectorStore<T>) -> Self {
        Self {
            store,
            embedding_model,
        }
    }
}

impl<T, M: EmbeddingModel> SqliteVectorIndex<T, M>
where
    T: SqliteVectorStoreTable,
{
    /// Runs the shared candidate search for `top_n`/`top_n_ids`.
    ///
    /// `outer_select_cols` is the outer `SELECT` list (aliases `d` for the
    /// document table and `scored` for the ranked candidates); `map_row` maps
    /// each result row, which ends with `scored.__rig_score`.
    async fn search_rows<R, F>(
        &self,
        req: &VectorSearchRequest<SqliteSearchFilter>,
        outer_select_cols: String,
        map_row: F,
    ) -> Result<Vec<R>, VectorStoreError>
    where
        R: Send + 'static,
        F: Fn(&rusqlite::Row<'_>) -> rusqlite::Result<R> + Send + 'static,
    {
        let embedding = self.embedding_model.embed_text(req.query()).await?;
        let query_vec: Vec<u8> = serialize_embedding(&embedding);
        let table_name = T::name();
        let embedding_map_table_name = format!("{table_name}_embedding_map");

        let distance_metric = self.store.distance_metric;
        let score_expression = distance_metric.score_expression("?1", "e.embedding");
        let filters = render_search_filters(req, distance_metric, &self.store.metadata_columns)?;
        let candidate_limit = self
            .store
            .candidate_limit(req.samples(), filters.has_post_filters())
            .await?;
        let search_query = build_search_query(query_vec, filters, candidate_limit)?;
        let where_clause = search_query.vector_where_clause;
        let document_filter_clause = search_query.document_filter_clause;
        let mut params = search_query.params;
        params.push(sqlite_limit_param(req.samples(), "result limit")?);

        self.store
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(&format!(
                    "WITH scored AS (
                        SELECT m.document_rowid AS __rig_document_rowid,
                            {score_expression} AS __rig_score,
                            ROW_NUMBER() OVER (
                                PARTITION BY m.document_rowid
                                ORDER BY {score_expression} DESC, e.rowid ASC
                            ) AS __rig_rank
                        FROM {table_name}_embeddings e
                        JOIN {embedding_map_table_name} m ON e.rowid = m.embedding_rowid
                        {where_clause}
                    )
                    SELECT {outer_select_cols}, scored.__rig_score
                    FROM scored
                    JOIN {table_name} d ON scored.__rig_document_rowid = d.rowid
                    WHERE scored.__rig_rank = 1
                        {document_filter_clause}
                    ORDER BY scored.__rig_score DESC, d.id ASC
                    LIMIT ?"
                ))?;

                let rows = stmt
                    .query_map(rusqlite::params_from_iter(params), |row| map_row(row))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(VectorStoreError::datastore)
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

struct SqliteSearchQuery {
    vector_where_clause: String,
    document_filter_clause: String,
    params: Vec<Value>,
}

fn render_search_filters(
    req: &VectorSearchRequest<SqliteSearchFilter>,
    distance_metric: SqliteDistanceMetric,
    metadata_columns: &[SqliteMetadataColumn],
) -> Result<SqliteRenderedFilters, FilterError> {
    let score_expression = distance_metric.score_expression("?1", "e.embedding");

    let mut filters = SqliteRenderedFilters::default();
    if let Some(threshold) = req.threshold() {
        filters.native.push(SqliteRenderedFilter {
            condition: format!("{score_expression} >= ?"),
            params: vec![Value::Real(threshold)],
        });
    }
    if let Some(filter) = req.filter() {
        filters.extend(filter.render_split(metadata_columns)?);
    }

    Ok(filters)
}

fn build_search_query(
    query_vec: Vec<u8>,
    filters: SqliteRenderedFilters,
    candidate_limit: u64,
) -> Result<SqliteSearchQuery, FilterError> {
    // `sqlite-vec`'s `vec0` KNN query caps `k` at `SQLITE_VEC_MAX_K`. When more
    // candidates than that are required for an exact result, drop the
    // `MATCH`/`k` KNN constraints and rank every row with the scalar
    // `vec_distance_*` functions already used by the score expression. The
    // `vec0` KNN path is itself an exact brute-force scan, so this yields the
    // same results without the `k` cap (just without the SIMD/chunk fast path).
    let brute_force = candidate_limit > SQLITE_VEC_MAX_K;

    let mut conditions = Vec::new();
    if !brute_force {
        conditions.push("e.embedding MATCH ?".to_string());
        conditions.push("k = ?".to_string());
    }
    conditions.extend(
        filters
            .native
            .iter()
            .map(|filter| format!("({})", filter.condition)),
    );

    // `conditions` is only empty on the brute-force path with no native
    // filters; emitting a bare `WHERE` then would be a syntax error.
    let vector_where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };
    let document_filter_clause = if filters.post.is_empty() {
        String::new()
    } else {
        format!(
            "AND {}",
            filters
                .post
                .iter()
                .map(|filter| format!("({})", filter.condition))
                .collect::<Vec<_>>()
                .join(" AND ")
        )
    };

    let query_vec = Value::Blob(query_vec);

    // Parameter binding is positional. The score expression uses the explicit
    // `?1` (bound by the first element here); the `MATCH`/`k` conditions and the
    // filter conditions use anonymous `?`, numbered left-to-right after `?1`.
    // On the brute-force path the `MATCH` and `k` placeholders are gone, so the
    // second `query_vec` and the candidate limit must be dropped too, leaving a
    // single leading `query_vec` for `?1`. Removing the tokens without removing
    // these two values would silently misalign every downstream filter param.
    let mut params = if brute_force {
        vec![query_vec]
    } else {
        let candidate_limit = sqlite_limit_param(candidate_limit, "candidate limit")?;
        vec![query_vec.clone(), query_vec, candidate_limit]
    };
    params.extend(filters.native.into_iter().flat_map(|filter| filter.params));
    params.extend(filters.post.into_iter().flat_map(|filter| filter.params));

    Ok(SqliteSearchQuery {
        vector_where_clause,
        document_filter_clause,
        params,
    })
}

#[cfg(test)]
fn build_where_clause(
    req: &VectorSearchRequest<SqliteSearchFilter>,
    query_vec: Vec<u8>,
    distance_metric: SqliteDistanceMetric,
    metadata_columns: &[SqliteMetadataColumn],
    candidate_limit: u64,
) -> Result<(String, Vec<Value>), FilterError> {
    let filters = render_search_filters(req, distance_metric, metadata_columns)?;
    let query = build_search_query(query_vec, filters, candidate_limit)?;
    Ok((query.vector_where_clause, query.params))
}

fn sqlite_limit_param(value: u64, name: &str) -> Result<Value, FilterError> {
    i64::try_from(value)
        .map(Value::Integer)
        .map_err(|_| FilterError::TypeError(format!("SQLite {name} `{value}` exceeds i64::MAX")))
}

fn sqlite_column_value_error(
    index: usize,
    value_type: Type,
    column: &Column,
    message: impl Into<String>,
) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        index,
        value_type,
        Box::new(SqliteInternalError::ColumnValueError {
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

fn sqlite_utf8_value<'a>(
    index: usize,
    value_type: Type,
    column: &Column,
    value: &'a [u8],
    label: &str,
) -> rusqlite::Result<&'a str> {
    std::str::from_utf8(value).map_err(|e| {
        sqlite_column_value_error(
            index,
            value_type,
            column,
            format!("invalid UTF-8 {label}: {e}"),
        )
    })
}

fn sqlite_text_value(
    index: usize,
    value_type: Type,
    column: &Column,
    value: &[u8],
) -> rusqlite::Result<serde_json::Value> {
    let value = sqlite_utf8_value(index, value_type, column, value, "text")?;

    Ok(serde_json::Value::String(value.to_string()))
}

fn sqlite_column_declares_json(column_type: &str) -> bool {
    column_type
        .split_whitespace()
        .next()
        .is_some_and(|token| token.eq_ignore_ascii_case("JSON"))
}

fn sqlite_json_text_value(
    index: usize,
    value_type: Type,
    column: &Column,
    value: &[u8],
) -> rusqlite::Result<serde_json::Value> {
    let value = sqlite_utf8_value(index, value_type, column, value, "JSON text")?;

    serde_json::from_str(value).map_err(|e| {
        sqlite_column_value_error(index, value_type, column, format!("invalid JSON text: {e}"))
    })
}

fn sqlite_column_value_to_json(
    index: usize,
    column: &Column,
    value: ValueRef<'_>,
) -> rusqlite::Result<serde_json::Value> {
    let value_type = value.data_type();

    if sqlite_column_declares_json(column.col_type) {
        return match value {
            ValueRef::Null => Ok(serde_json::Value::Null),
            ValueRef::Text(value) => sqlite_json_text_value(index, value_type, column, value),
            ValueRef::Integer(value) => Ok(serde_json::Value::Number(value.into())),
            ValueRef::Real(value) => sqlite_number_value(index, value_type, column, value),
            ValueRef::Blob(value) => sqlite_json_text_value(index, value_type, column, value),
        };
    }

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
                    Box::new(SqliteInternalError::ColumnValueError {
                        column_name: "id",
                        column_type: "TEXT",
                        message: format!("invalid UTF-8 text: {e}"),
                    }),
                )
            }),
        value => Err(rusqlite::Error::FromSqlConversionFailure(
            index,
            value.data_type(),
            Box::new(SqliteInternalError::ColumnValueError {
                column_name: "id",
                column_type: "TEXT or INTEGER",
                message: "id cannot be NULL or BLOB".to_string(),
            }),
        )),
    }
}

impl<T: SqliteVectorStoreTable, M: EmbeddingModel> VectorStoreIndex for SqliteVectorIndex<T, M> {
    type Filter = SqliteSearchFilter;

    async fn top_n<D>(
        &self,
        req: VectorSearchRequest<SqliteSearchFilter>,
    ) -> Result<Vec<(f64, String, D)>, VectorStoreError>
    where
        D: serde::de::DeserializeOwned,
    {
        tracing::debug!("Finding top {} matches for query", req.samples() as usize);
        if req.samples() == 0 {
            return Ok(Vec::new());
        }

        let columns = T::schema();
        let id_column_index = columns
            .iter()
            .position(|column| column.name == "id")
            .ok_or_else(|| {
                VectorStoreError::datastore(SqliteInternalError::MissingIdColumn(
                    T::name().to_string(),
                ))
            })?;

        let outer_select_cols = columns
            .iter()
            .map(|column| format!("d.{} AS {}", column.name, column.name))
            .collect::<Vec<_>>()
            .join(", ");

        let rows = self
            .search_rows(&req, outer_select_cols, move |row| {
                // Create a map of column names to values
                let mut map = serde_json::Map::new();
                for (i, column) in columns.iter().enumerate() {
                    let value = sqlite_column_value_to_json(i, column, row.get_ref(i)?)?;
                    map.insert(column.name.to_string(), value);
                }
                let score: f64 = row.get(columns.len())?;
                let id = sqlite_id_value_to_string(id_column_index, row.get_ref(id_column_index)?)?;

                Ok((id, serde_json::Value::Object(map), score))
            })
            .await?;

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

        let results = self
            .search_rows(&req, "d.id".to_string(), |row| {
                Ok((
                    row.get::<_, f64>(1)?,
                    sqlite_id_value_to_string(0, row.get_ref(0)?)?,
                ))
            })
            .await?;

        debug!("Found {} matching document IDs", results.len());
        Ok(results)
    }
}

/// Serializes an embedding straight to the little-endian f32 blob SQLite
/// stores, so neither the insert nor the query path needs an intermediate
/// `Vec<f32>`.
fn serialize_embedding(embedding: &Embedding) -> Vec<u8> {
    embedding
        .vec
        .iter()
        .flat_map(|x| (*x as f32).to_le_bytes())
        .collect()
}

macro_rules! impl_column_value {
    ($($ty:ty => $col_type:literal, |$value:ident| $to_sql:expr;)*) => {$(
        impl ColumnValue for $ty {
            fn to_sql_value(&self) -> Value {
                let $value = self;
                $to_sql
            }

            fn column_type(&self) -> &'static str {
                $col_type
            }
        }
    )*};
}

impl_column_value! {
    String => "TEXT", |value| Value::Text(value.clone());
    i64 => "INTEGER", |value| Value::Integer(*value);
    i32 => "INTEGER", |value| Value::Integer(i64::from(*value));
    f64 => "FLOAT", |value| Value::Real(*value);
    f32 => "FLOAT", |value| Value::Real(f64::from(*value));
    bool => "BOOLEAN", |value| Value::Integer(if *value { 1 } else { 0 });
    serde_json::Value => "JSON", |value| Value::Text(value.to_string());
}

#[cfg(test)]
mod tests;
