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
//! LanceDB vector store for Rig.
//!
//! [`LanceDbVectorIndex`] searches an existing LanceDB table, exactly or
//! approximately according to [`SearchParams`], narrowed by [`LanceDBFilter`]
//! SQL predicates. The `rig` facade re-exports this crate as `rig::lancedb`
//! under the `lancedb` feature.

use std::ops::Range;

use lancedb::{
    DistanceType,
    query::{QueryBase, VectorQuery},
};
use rig_core::{
    embeddings::embedding::EmbeddingModel,
    vector_store::{
        VectorStoreError, VectorStoreIndex,
        request::{FilterError, SearchFilter, VectorSearchRequest},
    },
    wasm_compat::WasmCompatSend,
};
use serde::de::DeserializeOwned;
use serde_json::Value;
use utils::{FilterTableColumns, QueryToJson};

mod utils;

/// Vector index over a LanceDB table.
///
/// Queries are embedded with the same model `M` that populated the table, so
/// results are meaningless under another model. See [`LanceDbVectorIndex::top_n`]
/// for a worked example.
pub struct LanceDbVectorIndex<M> {
    model: M,
    table: lancedb::Table,
    /// Column holding each record's id.
    id_field: String,
    search_params: SearchParams,
}

impl<M: EmbeddingModel> LanceDbVectorIndex<M> {
    /// Creates an index over an existing table whose ids live in `id_field`.
    /// The table is not inspected, so a wrong column surfaces at query time.
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

    /// Applies the configured search parameters. Probe and refinement settings
    /// apply only when approximate search is requested explicitly.
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
            query = query.column(column.as_str());
        }

        query
    }
}

/// Which search LanceDB performs. See
/// [LanceDB vector search](https://lancedb.github.io/lancedb/search/).
#[derive(Debug, Clone)]
pub enum SearchType {
    /// Exhaustive search, bypassing any vector index.
    Flat,
    /// Approximate nearest neighbor search.
    Approximate,
}

/// SQL predicate applied to a search, carrying any error raised while building
/// it until the filter is used.
///
/// Column names and array bounds are spliced in verbatim, so they must not carry
/// untrusted input; string values are quote-escaped.
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

    pub fn not(self) -> Self {
        Self(self.0.map(|s| format!("NOT ({s})")))
    }

    /// Matches rows whose `key` equals one of `values`.
    pub fn in_values(key: &str, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        Self(
            values
                .into_iter()
                .map(escape_value)
                .collect::<Result<Vec<_>, FilterError>>()
                .map(|xs| xs.join(","))
                .map(|xs| format!("{key} IN ({xs})")),
        )
    }

    /// Matches `key` against a case-sensitive SQL `LIKE` pattern.
    pub fn like<S>(key: &str, pattern: S) -> Self
    where
        S: AsRef<str>,
    {
        Self(
            escape_value(serde_json::Value::String(pattern.as_ref().into()))
                .map(|pat| format!("{key} LIKE {pat}")),
        )
    }

    /// Matches `key` against a case-insensitive `LIKE` pattern.
    pub fn ilike<S>(key: &str, pattern: S) -> Self
    where
        S: AsRef<str>,
    {
        Self(
            escape_value(serde_json::Value::String(pattern.as_ref().into()))
                .map(|pat| format!("{key} ILIKE {pat}")),
        )
    }

    /// Matches rows whose `key` is null.
    pub fn is_null(key: &str) -> Self {
        Self(Ok(format!("{key} IS NULL")))
    }

    /// Matches rows whose `key` is not null.
    pub fn is_not_null(key: &str) -> Self {
        Self(Ok(format!("{key} IS NOT NULL")))
    }

    /// Matches list columns sharing at least one element with `values`.
    pub fn array_has_any(key: &str, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        Self(
            values
                .into_iter()
                .map(escape_value)
                .collect::<Result<Vec<_>, FilterError>>()
                .map(|xs| xs.join(","))
                .map(|xs| format!("array_has_any({key}, ARRAY[{xs}])")),
        )
    }

    /// Matches list columns containing every element of `values`.
    pub fn array_has_all(key: &str, values: Vec<<Self as SearchFilter>::Value>) -> Self {
        Self(
            values
                .into_iter()
                .map(escape_value)
                .collect::<Result<Vec<_>, FilterError>>()
                .map(|xs| xs.join(","))
                .map(|xs| format!("array_has_all({key}, ARRAY[{xs}])")),
        )
    }

    /// Matches list columns holding exactly `length` elements.
    pub fn array_length(key: &str, length: i32) -> Self {
        Self(Ok(format!("array_length({key}) = {length}")))
    }

    /// Matches `key` within the inclusive bounds of `start` and `end`, which are
    /// spliced into the predicate verbatim.
    pub fn between<T>(key: &str, Range { start, end }: Range<T>) -> Self
    where
        T: PartialOrd + std::fmt::Display + Into<serde_json::Number>,
    {
        Self(Ok(format!("{key} BETWEEN {start} AND {end}")))
    }
}

/// Search tuning applied to every query.
///
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
    /// Sets the distance measure, which must match the one the index was built
    /// with. LanceDB defaults to L2.
    pub fn distance_type(mut self, distance_type: DistanceType) -> Self {
        self.distance_type = Some(distance_type);
        self
    }

    /// Forces exact or approximate search. When unset, LanceDB searches
    /// approximately if the table has a vector index and exhaustively otherwise.
    pub fn search_type(mut self, search_type: SearchType) -> Self {
        self.search_type = Some(search_type);
        self
    }

    /// Sets how many partitions approximate search probes. Ignored unless
    /// approximate search is requested explicitly. See
    /// [LanceDB ANN search](https://lancedb.github.io/lancedb/ann_indexes/#querying-an-ann-index).
    pub fn nprobes(mut self, nprobes: usize) -> Self {
        self.nprobes = Some(nprobes);
        self
    }

    /// Sets how many extra candidates approximate search reranks. Ignored unless
    /// approximate search is requested explicitly.
    pub fn refine_factor(mut self, refine_factor: u32) -> Self {
        self.refine_factor = Some(refine_factor);
        self
    }

    /// Applies filters after the vector search rather than before. See
    /// [LanceDB pre- and post-filtering](https://lancedb.github.io/lancedb/sql/#pre-and-post-filtering).
    pub fn post_filter(mut self, post_filter: bool) -> Self {
        self.post_filter = Some(post_filter);
        self
    }

    /// Names the embedding column to search. Needed only when the table has more
    /// than one float-list column.
    pub fn column(mut self, column: impl Into<String>) -> Self {
        self.column = Some(column.into());
        self
    }
}

impl<M: EmbeddingModel> VectorStoreIndex for LanceDbVectorIndex<M> {
    type Filter = LanceDBFilter;

    /// Returns matches as `(distance, id, row)` with embedding columns projected
    /// out. A row missing its distance scores zero, and a row whose id column is
    /// absent or non-textual gets a placeholder id.
    ///
    /// # Example
    /// ```no_run
    /// use rig_core::providers::openai::{self, wire::OpenAI};
    /// use rig_core::vector_store::VectorStoreIndex;
    /// use rig_core::vector_store::request::VectorSearchRequest;
    /// use rig_lancedb::{LanceDbVectorIndex, SearchParams};
    /// use rig_reqwest::prelude::*;
    ///
    /// # async fn example(table: lancedb::Table) -> Result<(), anyhow::Error> {
    /// let openai_client = OpenAI::from_env()?.bound()?;
    /// let model = openai_client.embedding(openai::TEXT_EMBEDDING_ADA_002, None);
    /// let vector_store_index =
    ///     LanceDbVectorIndex::new(table, model, "id", SearchParams::default()).await?;
    ///
    /// let req = VectorSearchRequest::builder()
    ///     .query("My boss says I zindle too much, what does that mean?")
    ///     .samples(1)
    ///     .build();
    ///
    /// let results = vector_store_index.top_n::<String>(req).await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<LanceDBFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_text(req.query()).await?;

        let mut query = self
            .table
            .vector_search(prompt_embedding.vec.clone())
            .map_err(VectorStoreError::datastore)?
            .limit(req.samples() as usize)
            .distance_range(None, req.threshold().map(|x| x as f32))
            .select(lancedb::query::Select::Columns(
                self.table
                    .schema()
                    .await
                    .map_err(VectorStoreError::datastore)?
                    .filter_embeddings(),
            ));

        if let Some(filter) = req.filter() {
            query = query.only_if(filter.clone().into_inner()?);
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
                    match value.get(self.id_field.as_str()) {
                        Some(Value::String(id)) => id.clone(),
                        _ => format!("unknown{i}"),
                    },
                    serde_json::from_value(value).map_err(VectorStoreError::JsonError)?,
                ))
            })
            .collect()
    }

    /// Like [`LanceDbVectorIndex::top_n`] but projects only the id column,
    /// returning an empty id when the column is absent or non-textual.
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
            .map_err(VectorStoreError::datastore)?
            .distance_range(None, req.threshold().map(|x| x as f32))
            .limit(req.samples() as usize);

        if let Some(filter) = req.filter() {
            query = query.only_if(filter.clone().into_inner()?);
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
                    match value.get(self.id_field.as_str()) {
                        Some(Value::String(id)) => id.clone(),
                        _ => "".to_string(),
                    },
                ))
            })
            .collect()
    }
}
