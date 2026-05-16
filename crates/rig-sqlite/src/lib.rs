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
use rusqlite::types::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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

pub trait ColumnValue: Send + Sync {
    fn to_sql_string(&self) -> String;
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
    /// KNN candidate search. Simple comparison filters are pushed to sqlite-vec;
    /// more complex filters are applied on the joined document table.
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
/// sqlite-vec's L2 distance is squared L2 distance.
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
        let column_type = column_type
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_ascii_uppercase();

        match column_type.as_str() {
            "TEXT" => Some(Self::Text),
            "INTEGER" | "INT" | "INT64" | "INTEGER64" => Some(Self::Integer),
            "FLOAT" | "REAL" | "DOUBLE" | "FLOAT64" | "F64" => Some(Self::Float),
            "BOOLEAN" | "BOOL" => Some(Self::Boolean),
            _ => None,
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
    value: String,
}

impl Display for SqliteMetadataValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "could not convert value `{}` for SQLite metadata column `{} {}`",
            self.value,
            self.column_name,
            self.column_type.vec0_name()
        )
    }
}

impl std::error::Error for SqliteMetadataValueError {}

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
        .to_sql_string();

    match column.metadata_type {
        SqliteMetadataType::Text => Ok(Value::Text(value)),
        SqliteMetadataType::Integer => value.parse::<i64>().map(Value::Integer).map_err(|_| {
            rusqlite::Error::ToSqlConversionFailure(Box::new(SqliteMetadataValueError {
                column_name: column.name,
                column_type: column.metadata_type,
                value,
            }))
        }),
        SqliteMetadataType::Float => value.parse::<f64>().map(Value::Real).map_err(|_| {
            rusqlite::Error::ToSqlConversionFailure(Box::new(SqliteMetadataValueError {
                column_name: column.name,
                column_type: column.metadata_type,
                value,
            }))
        }),
        SqliteMetadataType::Boolean => match value.to_ascii_lowercase().as_str() {
            "true" | "1" => Ok(Value::Integer(1)),
            "false" | "0" => Ok(Value::Integer(0)),
            _ => Err(rusqlite::Error::ToSqlConversionFailure(Box::new(
                SqliteMetadataValueError {
                    column_name: column.name,
                    column_type: column.metadata_type,
                    value,
                },
            ))),
        },
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
        let mut last_id = 0;
        let embedding_columns = std::iter::once("rowid")
            .chain(std::iter::once("embedding"))
            .chain(self.metadata_columns.iter().map(|column| column.name))
            .collect::<Vec<_>>();
        let embedding_placeholders = (1..=embedding_columns.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>();
        let embeddings_sql = format!(
            "INSERT INTO {table_name}_embeddings ({}) VALUES ({})",
            embedding_columns.join(", "),
            embedding_placeholders.join(", ")
        );

        for (doc, embeddings) in &documents {
            debug!("Storing document with id {}", doc.id());

            let values = doc.column_values();
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
                rusqlite::params_from_iter(values.iter().map(|(_, val)| val.to_sql_string())),
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
                let mut params = Vec::with_capacity(2 + metadata_values.len());
                params.push(Value::Integer(last_id));
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
        lo: String,
        hi: String,
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

#[derive(Clone, Copy, Deserialize, Serialize, Debug)]
enum SqliteComparisonOp {
    Eq,
    Gt,
    Lt,
}

impl SqliteComparisonOp {
    fn as_sql(self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::Gt => ">",
            Self::Lt => "<",
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
    Document,
}

impl SqliteFilterTarget {
    fn alias(self) -> &'static str {
        match self {
            Self::VectorMetadata => "e",
            Self::Document => "d",
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
    pub fn not(self) -> Self {
        Self {
            expr: SqliteSearchFilterExpr::Not(Box::new(self.expr)),
        }
    }

    /// Tests whether the value at `key` is contained in the range
    pub fn between<N>(key: String, range: RangeInclusive<N>) -> Self
    where
        N: Ord + rusqlite::ToSql + std::fmt::Display,
    {
        let lo = range.start();
        let hi = range.end();

        Self {
            expr: SqliteSearchFilterExpr::Between {
                key,
                lo: lo.to_string(),
                hi: hi.to_string(),
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

    // String ops
    /// Tests whether the value at `key` satisfies the glob pattern
    /// `pattern` should be a valid SQLite glob pattern
    pub fn glob<'a, S>(key: String, pattern: S) -> Self
    where
        S: AsRef<&'a str>,
    {
        Self {
            expr: SqliteSearchFilterExpr::Pattern {
                key,
                op: SqlitePatternOp::Glob,
                pattern: pattern.as_ref().to_string(),
            },
        }
    }

    /// Tests whether the value at `key` satisfies the "like" pattern
    /// `pattern` should be a valid SQLite like pattern
    pub fn like<'a, S>(key: String, pattern: S) -> Self
    where
        S: AsRef<&'a str>,
    {
        Self {
            expr: SqliteSearchFilterExpr::Pattern {
                key,
                op: SqlitePatternOp::Like,
                pattern: pattern.as_ref().to_string(),
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
        let metadata_column_names = metadata_columns
            .iter()
            .map(|column| column.name)
            .collect::<HashSet<_>>();

        self.expr.render_split(&metadata_column_names)
    }
}

impl SqliteSearchFilterExpr {
    fn render_split(
        &self,
        metadata_column_names: &HashSet<&'static str>,
    ) -> Result<SqliteRenderedFilters, FilterError> {
        match self {
            Self::Comparison { key, .. }
                if metadata_column_names.contains(key.as_str())
                    && !sqlite_key_is_qualified(key) =>
            {
                Ok(SqliteRenderedFilters {
                    native: vec![self.render(SqliteFilterTarget::VectorMetadata)?],
                    post: Vec::new(),
                })
            }
            Self::And(lhs, rhs) => {
                let mut rendered = lhs.render_split(metadata_column_names)?;
                rendered.extend(rhs.render_split(metadata_column_names)?);
                Ok(rendered)
            }
            Self::Or(_, _) | Self::Not(_) => Ok(SqliteRenderedFilters {
                native: Vec::new(),
                post: vec![self.render(SqliteFilterTarget::Document)?],
            }),
            _ => Ok(SqliteRenderedFilters {
                native: Vec::new(),
                post: vec![self.render(SqliteFilterTarget::Document)?],
            }),
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
                condition: format!("{} between {lo} and {hi}", sqlite_qualify_key(key, target)),
                params: Vec::new(),
            }),
            Self::NullCheck { key, negated } => {
                let operator = if *negated { "is not null" } else { "is null" };
                Ok(SqliteRenderedFilter {
                    condition: format!("{} {operator}", sqlite_qualify_key(key, target)),
                    params: Vec::new(),
                })
            }
            Self::Pattern { key, op, pattern } => Ok(SqliteRenderedFilter {
                condition: format!(
                    "{} {} {}",
                    sqlite_qualify_key(key, target),
                    op.as_sql(),
                    pattern
                ),
                params: Vec::new(),
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

fn sqlite_filter_param(value: serde_json::Value) -> Result<Value, FilterError> {
    use serde_json::Value::*;

    match value {
        Null => Ok(Value::Null),
        Bool(b) => Ok(Value::Integer(b as i64)),
        String(s) => Ok(Value::Text(s)),
        Number(n) => Ok(if let Some(float) = n.as_f64() {
            Value::Real(float)
        } else if let Some(int) = n.as_i64() {
            Value::Integer(int)
        } else if let Some(int) = n.as_u64() {
            Value::Integer(int as i64)
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
) -> Result<(String, Vec<Value>), FilterError> {
    let score_expression = distance_metric.score_expression("?1", "e.embedding");
    let threshold_filter = req.threshold().map(|threshold| {
        SqliteSearchFilter::raw(format!("{score_expression} > ?"), vec![threshold.into()])
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
    let samples = req.samples() as u32;

    let mut params = vec![query_vec.clone(), query_vec, samples.into()];
    params.extend(filters.native.into_iter().flat_map(|filter| filter.params));
    params.extend(filters.post.into_iter().flat_map(|filter| filter.params));

    Ok((where_clause, params))
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
        let embedding = self.embedding_model.embed_text(req.query()).await?;
        let query_vec: Vec<f32> = serialize_embedding(&embedding);
        let table_name = T::name();

        // Get all column names from SqliteVectorStoreTable
        let columns = T::schema();
        let column_names: Vec<&str> = columns.iter().map(|column| column.name).collect();

        // Build SELECT statement with all columns
        let select_cols = column_names
            .iter()
            .map(|column| format!("d.{column}"))
            .collect::<Vec<_>>()
            .join(", ");

        let distance_metric = self.store.distance_metric;
        let score_expression = distance_metric.score_expression("?1", "e.embedding");
        let (where_clause, params) = build_where_clause(
            &req,
            query_vec,
            distance_metric,
            &self.store.metadata_columns,
        )?;

        let rows = self
            .store
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(&format!(
                    "SELECT {select_cols}, {score_expression} as score
                    FROM {table_name}_embeddings e
                    JOIN {table_name} d ON e.rowid = d.rowid
                    {where_clause}
                    ORDER BY score DESC"
                ))?;

                let rows = stmt
                    .query_map(rusqlite::params_from_iter(params), |row| {
                        // Create a map of column names to values
                        let mut map = serde_json::Map::new();
                        for (i, col_name) in column_names.iter().enumerate() {
                            let value: String = row.get(i)?;
                            map.insert(col_name.to_string(), serde_json::Value::String(value));
                        }
                        let score: f64 = row.get(column_names.len())?;
                        let id: String = row.get(0)?; // Assuming id is always first column

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
        let embedding = self.embedding_model.embed_text(req.query()).await?;
        let query_vec = serialize_embedding(&embedding);
        let table_name = T::name();

        let distance_metric = self.store.distance_metric;
        let score_expression = distance_metric.score_expression("?1", "e.embedding");
        let (where_clause, params) = build_where_clause(
            &req,
            query_vec,
            distance_metric,
            &self.store.metadata_columns,
        )?;

        let results = self
            .store
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(&format!(
                    "SELECT d.id, {score_expression} as score
                     FROM {table_name}_embeddings e
                     JOIN {table_name} d ON e.rowid = d.rowid
                     {where_clause}
                     ORDER BY score DESC"
                ))?;

                let results = stmt
                    .query_map(rusqlite::params_from_iter(params), |row| {
                        Ok((row.get::<_, f64>(1)?, row.get::<_, String>(0)?))
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
    fn to_sql_string(&self) -> String {
        self.clone()
    }

    fn column_type(&self) -> &'static str {
        "TEXT"
    }
}

impl ColumnValue for i64 {
    fn to_sql_string(&self) -> String {
        self.to_string()
    }

    fn column_type(&self) -> &'static str {
        "INTEGER"
    }
}

impl ColumnValue for i32 {
    fn to_sql_string(&self) -> String {
        self.to_string()
    }

    fn column_type(&self) -> &'static str {
        "INTEGER"
    }
}

impl ColumnValue for f64 {
    fn to_sql_string(&self) -> String {
        self.to_string()
    }

    fn column_type(&self) -> &'static str {
        "FLOAT"
    }
}

impl ColumnValue for f32 {
    fn to_sql_string(&self) -> String {
        self.to_string()
    }

    fn column_type(&self) -> &'static str {
        "FLOAT"
    }
}

impl ColumnValue for bool {
    fn to_sql_string(&self) -> String {
        self.to_string()
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
    use std::sync::Once;
    use tokio_rusqlite::Connection;

    fn test_metadata_columns() -> Vec<SqliteMetadataColumn> {
        vec![SqliteMetadataColumn {
            name: "category",
            metadata_type: SqliteMetadataType::Text,
        }]
    }

    #[test]
    fn threshold_filter_uses_computed_similarity_expression() -> anyhow::Result<()> {
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .threshold(0.95)
            .build();

        let (where_clause, params) =
            build_where_clause(&req, vec![1.0, 0.0], SqliteDistanceMetric::Cosine, &[])?;

        anyhow::ensure!(
            where_clause.contains("e.embedding MATCH ?"),
            "missing vector match constraint: {where_clause}"
        );
        anyhow::ensure!(
            where_clause.contains("k = ?"),
            "missing vector k constraint: {where_clause}"
        );
        anyhow::ensure!(
            where_clause.contains("(1 - vec_distance_cosine(?1, e.embedding)) > ?"),
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
            build_where_clause(&req, vec![1.0, 0.0], SqliteDistanceMetric::L2, &[])?;

        anyhow::ensure!(
            where_clause.contains("(-vec_distance_l2(?1, e.embedding)) > ?"),
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
            build_where_clause(&req, vec![1.0, 0.0], SqliteDistanceMetric::Cosine, &[])?;

        anyhow::ensure!(
            where_clause == "WHERE e.embedding MATCH ? AND k = ?",
            "unexpected where clause: {where_clause}"
        );
        anyhow::ensure!(params.len() == 3, "unexpected params: {params:?}");

        Ok(())
    }

    #[test]
    fn no_threshold_or_filter_keeps_vector_constraints_grouped() -> anyhow::Result<()> {
        let filter = SqliteSearchFilter::eq("category", serde_json::json!("docs")).or(
            SqliteSearchFilter::eq("category", serde_json::json!("archive")),
        );

        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .filter(filter)
            .build();

        let (where_clause, params) = build_where_clause(
            &req,
            vec![1.0, 0.0],
            SqliteDistanceMetric::Cosine,
            &test_metadata_columns(),
        )?;

        anyhow::ensure!(
            where_clause
                == "WHERE e.embedding MATCH ? AND k = ? AND ((d.category = ?) OR (d.category = ?))",
            "unexpected where clause: {where_clause}"
        );
        anyhow::ensure!(params.len() == 5, "unexpected params: {params:?}");

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
    async fn live_or_filter_without_threshold_does_not_bypass_vector_constraints()
    -> anyhow::Result<()> {
        let index = live_test_index(
            "live_or_filter_without_threshold_does_not_bypass_vector_constraints",
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

        let results = index.top_n::<TestDocument>(req).await?;

        anyhow::ensure!(
            results.is_empty(),
            "OR filter should not return rows outside the top-k vector match set: {results:?}"
        );

        Ok(())
    }

    type SqliteExtensionFn =
        unsafe extern "C" fn(*mut sqlite3, *mut *mut i8, *const sqlite3_api_routines) -> i32;

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
