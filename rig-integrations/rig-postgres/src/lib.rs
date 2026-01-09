use std::{fmt::Display, ops::RangeInclusive};

use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{SearchFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use sqlx::{PgPool, Postgres, postgres::PgArguments, query::QueryAs};
use uuid::Uuid;

pub struct PostgresVectorStore<Model: EmbeddingModel> {
    model: Model,
    pg_pool: PgPool,
    documents_table: String,
    distance_function: PgVectorDistanceFunction,
}

/* PgVector supported distances
<-> - L2 distance
<#> - (negative) inner product
<=> - cosine distance
<+> - L1 distance (added in 0.7.0)
<~> - Hamming distance (binary vectors, added in 0.7.0)
<%> - Jaccard distance (binary vectors, added in 0.7.0)
 */
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

#[derive(Clone, Default, Serialize, Deserialize, Debug)]
pub struct PgSearchFilter {
    condition: String,
    values: Vec<serde_json::Value>,
}

impl SearchFilter for PgSearchFilter {
    type Value = serde_json::Value;

    fn eq(key: String, value: Self::Value) -> Self {
        Self {
            condition: format!("{key} = $"),
            values: vec![value],
        }
    }

    fn gt(key: String, value: Self::Value) -> Self {
        Self {
            condition: format!("{key} > $"),
            values: vec![value],
        }
    }

    fn lt(key: String, value: Self::Value) -> Self {
        Self {
            condition: format!("{key} < $"),
            values: vec![value],
        }
    }

    fn and(self, rhs: Self) -> Self {
        Self {
            condition: format!("({}) AND ({})", self.condition, rhs.condition),
            values: self.values.into_iter().chain(rhs.values).collect(),
        }
    }

    fn or(self, rhs: Self) -> Self {
        Self {
            condition: format!("({}) OR ({})", self.condition, rhs.condition),
            values: self.values.into_iter().chain(rhs.values).collect(),
        }
    }
}

impl PgSearchFilter {
    fn into_clause(self) -> (String, Vec<serde_json::Value>) {
        (self.condition, self.values)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self {
            condition: format!("NOT ({})", self.condition),
            values: self.values,
        }
    }

    pub fn gte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self {
            condition: format!("{key} >= ?"),
            values: vec![value],
        }
    }

    pub fn lte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self {
            condition: format!("{key} <= ?"),
            values: vec![value],
        }
    }

    pub fn is_null(key: String) -> Self {
        Self {
            condition: format!("{key} is null"),
            ..Default::default()
        }
    }

    pub fn is_not_null(key: String) -> Self {
        Self {
            condition: format!("{key} is not null"),
            ..Default::default()
        }
    }

    pub fn between<T>(key: String, range: RangeInclusive<T>) -> Self
    where
        T: std::fmt::Display + Into<serde_json::Number> + Copy,
    {
        let lo = range.start();
        let hi = range.end();

        Self {
            condition: format!("{key} between {lo} and {hi}"),
            ..Default::default()
        }
    }

    pub fn member(key: String, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        let placeholders = values.iter().map(|_| "?").collect::<Vec<&str>>().join(",");

        Self {
            condition: format!("{key} is in ({placeholders})"),
            values,
        }
    }

    // String matching ops

    /// Tests whether the value at `key` matches the (case-sensitive) pattern
    /// `pattern` should be a valid SQL string pattern, with '%' and '_' as wildcards
    pub fn like(key: String, pattern: &'static str) -> Self {
        Self {
            condition: format!("{key} like {pattern}"),
            ..Default::default()
        }
    }

    /// Tests whether the value at `key` matches the SQL regex pattern
    /// `pattern` should be a valid regex
    pub fn similar_to(key: String, pattern: &'static str) -> Self {
        Self {
            condition: format!("{key} similar to {pattern}"),
            ..Default::default()
        }
    }
}

fn bind_value<S>(
    builder: QueryAs<'_, Postgres, S, PgArguments>,
    value: Value,
) -> QueryAs<'_, Postgres, S, PgArguments> {
    match value {
        Value::Null => unreachable!(),
        Value::Bool(b) => builder.bind(b),
        Value::Number(num) => {
            if let Some(n) = num.as_f64() {
                builder.bind(n)
            } else if let Some(n) = num.as_i64() {
                builder.bind(n)
            } else {
                unreachable!()
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
        // Will always be JSONB
        object => builder.bind(object),
    }
}

#[derive(Debug, Deserialize, sqlx::FromRow)]
pub struct SearchResult {
    id: Uuid,
    document: Value,
    //embedded_text: String,
    distance: f64,
}

#[derive(Debug, Deserialize, sqlx::FromRow)]
pub struct SearchResultOnlyId {
    id: Uuid,
    distance: f64,
}

impl SearchResult {
    pub fn into_result<T: DeserializeOwned>(self) -> Result<(f64, String, T), VectorStoreError> {
        let document: T =
            serde_json::from_value(self.document).map_err(VectorStoreError::JsonError)?;
        Ok((self.distance, self.id.to_string(), document))
    }
}

impl<Model> PostgresVectorStore<Model>
where
    Model: EmbeddingModel,
{
    pub fn new(
        model: Model,
        pg_pool: PgPool,
        documents_table: Option<String>,
        distance_function: PgVectorDistanceFunction,
    ) -> Self {
        Self {
            model,
            pg_pool,
            documents_table: documents_table.unwrap_or(String::from("documents")),
            distance_function,
        }
    }

    pub fn with_defaults(model: Model, pg_pool: PgPool) -> Self {
        Self::new(model, pg_pool, None, PgVectorDistanceFunction::Cosine)
    }

    fn search_query_full(
        &self,
        req: &VectorSearchRequest<PgSearchFilter>,
    ) -> (String, Vec<serde_json::Value>) {
        self.search_query(true, req)
    }

    fn search_query_only_ids(
        &self,
        req: &VectorSearchRequest<PgSearchFilter>,
    ) -> (String, Vec<serde_json::Value>) {
        self.search_query(false, req)
    }

    fn search_query(
        &self,
        with_document: bool,
        req: &VectorSearchRequest<PgSearchFilter>,
    ) -> (String, Vec<serde_json::Value>) {
        let document = if with_document { ", document" } else { "" };

        let thresh = req
            .threshold()
            .map(|t| PgSearchFilter::gt("distance".into(), t.into()));
        let filter = match (thresh, req.filter()) {
            (Some(thresh), Some(filt)) => Some(thresh.and(filt.clone())),
            (Some(thresh), _) => Some(thresh),
            (_, Some(filt)) => Some(filt.clone()),
            _ => None,
        };
        let (where_clause, params) = match filter {
            Some(f) => {
                let (expr, params) = f.into_clause();
                (String::from("WHERE") + &expr, params)
            }
            None => (Default::default(), Default::default()),
        };

        let mut counter = 3;
        let mut buf = String::with_capacity(where_clause.len() * 2);

        for c in where_clause.chars() {
            buf.push(c);

            if c == '$' {
                buf.push_str(counter.to_string().as_str());
                counter += 1;
            }
        }

        let where_clause = buf;

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
            document, document, self.distance_function, self.documents_table
        );

        (query, params)
    }
}

impl<Model> InsertDocuments for PostgresVectorStore<Model>
where
    Model: EmbeddingModel + Send + Sync,
{
    async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        for (document, embeddings) in documents {
            let id = Uuid::new_v4();
            let json_document = serde_json::to_value(&document).unwrap();

            for embedding in embeddings {
                let embedding_text = embedding.document;
                let embedding: Vec<f64> = embedding.vec;

                sqlx::query(
                    format!(
                        "INSERT INTO {} (id, document, embedded_text, embedding) VALUES ($1, $2, $3, $4)",
                        self.documents_table
                    )
                    .as_str(),
                )
                .bind(id)
                .bind(&json_document)
                .bind(&embedding_text)
                .bind(&embedding)
                .execute(&self.pg_pool)
                .await
                .map_err(|e| VectorStoreError::DatastoreError(e.into()))?;
            }
        }

        Ok(())
    }
}

impl<Model> VectorStoreIndex for PostgresVectorStore<Model>
where
    Model: EmbeddingModel,
{
    type Filter = PgSearchFilter;

    /// Get the top n documents based on the distance to the given query.
    /// The result is a list of tuples of the form (score, id, document)
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<PgSearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        if req.samples() > i64::MAX as u64 {
            return Err(VectorStoreError::DatastoreError(
                format!(
                    "The maximum amount of samples to return with the `rig` Postgres integration cannot be larger than {}",
                    i64::MAX
                )
                .into(),
            ));
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

        let (search_query, params) = self.search_query_full(&req);
        let builder = sqlx::query_as(search_query.as_str())
            .bind(embedded_query)
            .bind(req.samples() as i64);

        let builder = params.iter().cloned().fold(builder, bind_value);

        let rows = builder
            .fetch_all(&self.pg_pool)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows: Vec<(f64, String, T)> = rows
            .into_iter()
            .flat_map(SearchResult::into_result)
            .collect();

        Ok(rows)
    }

    /// Same as `top_n` but returns the document ids only.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<PgSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        if req.samples() > i64::MAX as u64 {
            return Err(VectorStoreError::DatastoreError(
                format!(
                    "The maximum amount of samples to return with the `rig` Postgres integration cannot be larger than {}",
                    i64::MAX
                )
                .into(),
            ));
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

        let (search_query, params) = self.search_query_only_ids(&req);
        let builder = sqlx::query_as(search_query.as_str())
            .bind(embedded_query)
            .bind(req.samples() as i64);

        let builder = params.iter().cloned().fold(builder, bind_value);

        let rows: Vec<SearchResultOnlyId> = builder
            .fetch_all(&self.pg_pool)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows: Vec<(f64, String)> = rows
            .into_iter()
            .map(|row| (row.distance, row.id.to_string()))
            .collect();

        Ok(rows)
    }
}
