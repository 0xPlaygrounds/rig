//! PostgreSQL and pgvector vector store for Rig.
//!
//! [`PostgresVectorStore`] searches a table holding a `pgvector` embedding
//! column using a [`PgVectorDistanceFunction`] and optional [`PgSearchFilter`]
//! conditions. The `rig` facade re-exports this crate as `rig::postgres` under
//! the `postgres` feature.

use std::{fmt::Display, ops::RangeInclusive};

use rig_core::{
    Embed,
    embeddings::Embedding,
    vector_store::{
        InsertDocuments, VectorSearchIdResult, VectorSearchResult, VectorStoreError,
        VectorStoreIndex,
        request::{SearchFilter, SqlCondition, VectorSearchRequest},
    },
    wasm_compat::WasmCompatSend,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use sqlx::{PgPool, Postgres, postgres::PgArguments, query::QueryAs};
use uuid::Uuid;

/// Vector store over a Postgres table. Queries are embedded with the same model
/// that populated the table, so results are meaningless under another model.
pub struct PostgresVectorStore {
    model: rig_core::DynModel<rig_core::operation::Embedding>,
    pg_pool: PgPool,
    documents_table: String,
    distance_function: PgVectorDistanceFunction,
}

/// pgvector distance operators. `Hamming` and `Jaccard` apply to binary vectors,
/// and all operators except L2, inner product, and cosine require pgvector 0.7.
pub enum PgVectorDistanceFunction {
    L2,
    InnerProduct,
    Cosine,
    L1,
    Hamming,
    Jaccard,
}

impl Display for PgVectorDistanceFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PgVectorDistanceFunction::L2 => write!(f, "<->"),
            PgVectorDistanceFunction::InnerProduct => write!(f, "<#>"),
            PgVectorDistanceFunction::Cosine => write!(f, "<=>"),
            PgVectorDistanceFunction::L1 => write!(f, "<+>"),
            PgVectorDistanceFunction::Hamming => write!(f, "<~>"),
            PgVectorDistanceFunction::Jaccard => write!(f, "<%>"),
        }
    }
}

impl PgVectorDistanceFunction {
    /// Builds a SQL expression increasing with similarity so a minimum-similarity
    /// threshold applies as `score >= $n`. `embedding` and `query` are spliced
    /// verbatim and must not carry untrusted input.
    fn score_expression(&self, embedding: &str, query: &str) -> String {
        match self {
            PgVectorDistanceFunction::Cosine | PgVectorDistanceFunction::Jaccard => {
                format!("1 - ({embedding} {self} {query})")
            }
            PgVectorDistanceFunction::InnerProduct
            | PgVectorDistanceFunction::L2
            | PgVectorDistanceFunction::L1
            | PgVectorDistanceFunction::Hamming => format!("-({embedding} {self} {query})"),
        }
    }
}

/// Bind placeholder token. Query rendering numbers the placeholders that
/// [`SqlCondition`] recorded, never other `$` characters in the text.
const PLACEHOLDER: &str = "$";

/// Postgres `WHERE` fragment with its bind values.
///
/// Values, `like`/`similar_to` patterns, and `between` bounds are bound as
/// parameters. Keys are spliced into the SQL verbatim so that expressions such
/// as `document->>'kind'` work, which means a key is executable SQL: never
/// build one from untrusted input.
#[derive(Clone, Default, Serialize, Deserialize, Debug)]
pub struct PgSearchFilter(SqlCondition<serde_json::Value>);

impl SearchFilter for PgSearchFilter {
    type Value = serde_json::Value;

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

impl PgSearchFilter {
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

    pub fn is_null(key: &str) -> Self {
        Self(SqlCondition::raw(format!("{key} is null")))
    }

    pub fn is_not_null(key: &str) -> Self {
        Self(SqlCondition::raw(format!("{key} is not null")))
    }

    /// SQL `BETWEEN` with both bounds bound as parameters. A non-finite float
    /// bound binds as `NULL`, so the condition matches no rows.
    pub fn between<T: Into<Value>>(key: impl AsRef<str>, range: RangeInclusive<T>) -> Self {
        let (lo, hi) = range.into_inner();
        Self(SqlCondition::between(
            key,
            PLACEHOLDER,
            lo.into(),
            hi.into(),
        ))
    }

    pub fn member(key: &str, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        Self(SqlCondition::list(key, "IN", PLACEHOLDER, values))
    }

    // String matching ops

    /// Case-sensitive SQL `LIKE` match against a bound pattern, with `%` and `_`
    /// as wildcards. Pass the bare pattern, such as `%apple%`, without quotes.
    pub fn like(key: impl AsRef<str>, pattern: impl Into<String>) -> Self {
        Self(SqlCondition::binary(
            key,
            "like",
            PLACEHOLDER,
            Value::String(pattern.into()),
        ))
    }

    /// SQL `SIMILAR TO` match against a bound pattern. Pass the bare pattern
    /// without quotes.
    pub fn similar_to(key: impl AsRef<str>, pattern: impl Into<String>) -> Self {
        Self(SqlCondition::binary(
            key,
            "similar to",
            PLACEHOLDER,
            Value::String(pattern.into()),
        ))
    }
}

fn bind_value<S>(
    builder: QueryAs<'_, Postgres, S, PgArguments>,
    value: Value,
) -> QueryAs<'_, Postgres, S, PgArguments> {
    match value {
        Value::Null => builder.bind(Option::<String>::None),
        Value::Bool(b) => builder.bind(b),
        Value::Number(num) => {
            if let Some(n) = num.as_f64() {
                builder.bind(n)
            } else if let Some(n) = num.as_i64() {
                builder.bind(n)
            } else if let Some(n) = num.as_u64() {
                builder.bind(n as i64)
            } else {
                builder.bind(num.to_string())
            }
        }
        Value::String(s) => builder.bind(s),
        Value::Array(xs) => {
            if let Some(xs) = xs
                .iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()
            {
                builder.bind(xs)
            } else if let Some(xs) = xs.iter().map(Value::as_f64).collect::<Option<Vec<_>>>() {
                builder.bind(xs)
            } else if let Some(xs) = xs.iter().map(Value::as_i64).collect::<Option<Vec<_>>>() {
                builder.bind(xs)
            } else if let Some(xs) = xs.iter().map(Value::as_bool).collect::<Option<Vec<_>>>() {
                builder.bind(xs)
            } else {
                builder.bind(Value::Array(xs))
            }
        }
        object => builder.bind(object),
    }
}

#[derive(Debug, Deserialize, sqlx::FromRow)]
pub struct SearchResult {
    id: Uuid,
    document: Value,
    distance: f64,
}

#[derive(Debug, Deserialize, sqlx::FromRow)]
pub struct SearchResultOnlyId {
    id: Uuid,
    distance: f64,
}

impl SearchResult {
    pub fn into_result<T: DeserializeOwned>(
        self,
    ) -> Result<VectorSearchResult<T>, VectorStoreError> {
        Ok(VectorSearchResult {
            score: self.distance,
            id: self.id.to_string(),
            document: serde_json::from_value(self.document).map_err(VectorStoreError::JsonError)?,
        })
    }
}

impl PostgresVectorStore {
    pub fn new(
        model: impl Into<rig_core::DynModel<rig_core::operation::Embedding>>,
        pg_pool: PgPool,
        documents_table: Option<String>,
        distance_function: PgVectorDistanceFunction,
    ) -> Self {
        Self {
            model: model.into(),
            pg_pool,
            documents_table: documents_table.unwrap_or_else(|| String::from("documents")),
            distance_function,
        }
    }

    pub fn with_defaults(
        model: impl Into<rig_core::DynModel<rig_core::operation::Embedding>>,
        pg_pool: PgPool,
    ) -> Self {
        Self::new(model, pg_pool, None, PgVectorDistanceFunction::Cosine)
    }

    /// Embeds the query and runs the search, returning one row per result.
    /// Errors when the requested sample count exceeds `i64::MAX`.
    async fn run_search<R>(
        &self,
        req: &VectorSearchRequest<PgSearchFilter>,
        with_document: bool,
    ) -> Result<Vec<R>, VectorStoreError>
    where
        R: for<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> + Send + Unpin,
    {
        if req.samples() > i64::MAX as u64 {
            return Err(VectorStoreError::SamplesOutOfRange {
                requested: req.samples(),
                max: i64::MAX as u64,
            });
        }

        let embedded_query: pgvector::Vector = self
            .model
            .embed_text(req.query())
            .await?
            .vec
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<f32>>()
            .into();

        let (search_query, params) = render_search_query(
            &self.distance_function,
            &self.documents_table,
            with_document,
            req,
        );
        let builder = sqlx::query_as(sqlx::AssertSqlSafe(search_query))
            .bind(embedded_query)
            .bind(req.samples() as i64);

        let builder = params.iter().cloned().fold(builder, bind_value);

        builder
            .fetch_all(&self.pg_pool)
            .await
            .map_err(VectorStoreError::datastore)
    }
}

/// Renders the search SQL and the values bound after `$1` (query vector) and
/// `$2` (limit).
///
/// The threshold filters on a similarity expression inside the inner `SELECT`,
/// since the `distance` alias is only visible to the outer query. Returned
/// scores remain raw distances ordered ascending regardless of threshold.
fn render_search_query(
    distance_function: &PgVectorDistanceFunction,
    documents_table: &str,
    with_document: bool,
    req: &VectorSearchRequest<PgSearchFilter>,
) -> (String, Vec<serde_json::Value>) {
    let document = if with_document { ", document" } else { "" };

    // The threshold binds before filter values.
    let mut params = Vec::new();
    let mut conditions = Vec::new();

    if let Some(threshold) = req.threshold() {
        let score = distance_function.score_expression("embedding", "$1");
        conditions.push(format!("({score} >= $3)"));
        params.push(serde_json::Value::from(threshold));
    }

    if let Some(filter) = req.filter() {
        let first = 3 + params.len();
        let expr = filter.0.render_placeholders(|i| format!("${}", first + i));
        conditions.push(format!("({expr})"));
        params.extend_from_slice(filter.0.params());
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };

    let query = format!(
        "
            SELECT id{}, distance FROM ( \
              SELECT DISTINCT ON (id) id{}, embedding {} $1 as distance \
              FROM {} \
              {where_clause} \
              ORDER BY id, distance \
            ) as d \
            ORDER BY distance \
            LIMIT $2",
        document, document, distance_function, documents_table
    );

    (query, params)
}

impl InsertDocuments for PostgresVectorStore {
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        for (document, embeddings) in documents {
            let id = Uuid::new_v4();
            let json_document = serde_json::to_value(&document)?;

            for embedding in embeddings {
                let embedding_text = embedding.document;
                let embedding: Vec<f64> = embedding.vec;

                sqlx::query(sqlx::AssertSqlSafe(format!(
                    "INSERT INTO {} (id, document, embedded_text, embedding) VALUES ($1, $2, $3, $4)",
                    self.documents_table
                )))
                .bind(id)
                .bind(&json_document)
                .bind(&embedding_text)
                .bind(&embedding)
                .execute(&self.pg_pool)
                .await
                .map_err(VectorStoreError::datastore)?;
            }
        }

        Ok(())
    }
}

impl VectorStoreIndex for PostgresVectorStore {
    type Filter = PgSearchFilter;

    /// Returns up to `samples` documents scored by distance, ordered by
    /// ascending distance. Rows whose documents fail to deserialize are skipped.
    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<PgSearchFilter>,
    ) -> Result<Vec<VectorSearchResult<T>>, VectorStoreError> {
        let rows: Vec<SearchResult> = self.run_search(&req, true).await?;

        let rows: Vec<VectorSearchResult<T>> = rows
            .into_iter()
            .flat_map(SearchResult::into_result)
            .collect();

        Ok(rows)
    }

    /// Like `top_n` but returns only scores and ids, without deserializing documents.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<PgSearchFilter>,
    ) -> Result<Vec<VectorSearchIdResult>, VectorStoreError> {
        let rows: Vec<SearchResultOnlyId> = self.run_search(&req, false).await?;

        let rows: Vec<VectorSearchIdResult> = rows
            .into_iter()
            .map(|row| VectorSearchIdResult {
                score: row.distance,
                id: row.id.to_string(),
            })
            .collect();

        Ok(rows)
    }
}

#[cfg(test)]
mod tests;
