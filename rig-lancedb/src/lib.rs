use lancedb::{
    DistanceType,
    query::{QueryBase, VectorQuery},
};
use rig::{
    embeddings::embedding::EmbeddingModel,
    vector_store::{
        VectorStoreError, VectorStoreIndex,
        request::{FilterError, SearchFilter, VectorSearchRequest},
    },
};
use serde::Deserialize;
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
/// # Example
/// ```
/// use rig_lancedb::{LanceDbVectorIndex, SearchParams};
/// use rig::providers::openai::{Client, TEXT_EMBEDDING_ADA_002, EmbeddingModel};
///
/// let openai_client = Client::from_env();
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

fn zip_result(
    l: Result<String, FilterError>,
    r: Result<String, FilterError>,
) -> Result<(String, String), FilterError> {
    l.and_then(|l| r.map(|r| (l, r)))
}

impl SearchFilter for LanceDBFilter {
    type Value = serde_json::Value;

    fn eq(key: String, value: Self::Value) -> Self {
        Self(escape_value(value).map(|s| format!("{key} = {s}")))
    }

    fn gt(key: String, value: Self::Value) -> Self {
        Self(escape_value(value).map(|s| format!("{key} > {s}")))
    }

    fn lt(key: String, value: Self::Value) -> Self {
        Self(escape_value(value).map(|s| format!("{key} < {s}")))
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
    /// ```
    /// use rig_lancedb::{LanceDbVectorIndex, SearchParams};
    /// use rig::providers::openai::{EmbeddingModel, Client, TEXT_EMBEDDING_ADA_002};
    ///
    /// let openai_client = Client::from_env();
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
    /// ```
    /// use rig_lancedb::{LanceDbVectorIndex, SearchParams};
    /// use rig::providers::openai::{Client, TEXT_EMBEDDING_ADA_002, EmbeddingModel};
    ///
    /// let openai_client = Client::from_env();
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
