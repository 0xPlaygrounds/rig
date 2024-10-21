use std::marker::PhantomData;

use lancedb::{
    query::{QueryBase, VectorQuery},
    DistanceType,
};
use rig::{
    embeddings::embedding::EmbeddingModel,
    vector_store::{VectorStoreError, VectorStoreIndex},
};
use serde::Deserialize;
use serde_json::Value;
use utils::QueryToJson;

mod utils;

fn lancedb_to_rig_error(e: lancedb::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

fn serde_to_rig_error(e: serde_json::Error) -> VectorStoreError {
    VectorStoreError::JsonError(e)
}

/// A vector index on a LanceDB table.
/// # Example
/// ```
/// use std::{env, sync::Arc};

/// use arrow_array::RecordBatchIterator;
/// use fixture::{as_record_batch, fake_definitions, schema, FakeDefinition};
/// use lancedb::index::vector::IvfPqIndexBuilder;
/// use rig::vector_store::VectorStoreIndex;
/// use rig::{
///     embeddings::{builder::EmbeddingsBuilder, embedding::EmbeddingModel},
///     providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
/// };
/// use rig_lancedb::{LanceDbVectorIndex, SearchParams};
///
/// #[path = "../examples/fixtures/lib.rs"]
/// mod fixture;
///
/// // Initialize OpenAI client. Use this to generate embeddings (and generate test data for RAG demo).
/// let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
/// let openai_client = Client::new(&openai_api_key);
///
/// // Select an embedding model.
/// let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
///
/// // Initialize LanceDB locally.
/// let db = lancedb::connect("data/lancedb-store").execute().await?;
///
/// // Generate embeddings for the test data.
/// let embeddings = EmbeddingsBuilder::new(model.clone())
///     .documents(fake_definitions())?
///     // Note: need at least 256 rows in order to create an index so copy the definition 256 times for testing purposes.
///     .documents(
///         (0..256)
///         .map(|i| FakeDefinition {
///             id: format!("doc{}", i),
///             definition: "Definition of *flumbuzzle (noun)*: A sudden, inexplicable urge to rearrange or reorganize small objects, such as desk items or books, for no apparent reason.".to_string()
///         })
///         .collect(),
///     )?
///     .build()
///     .await?;
///
/// // Create table with embeddings.
/// let record_batch = as_record_batch(embeddings, model.ndims());
/// let table = db
///     .create_table(
///        "definitions",
///        RecordBatchIterator::new(vec![record_batch], Arc::new(schema(model.ndims()))),
///     )
///     .execute()
///     .await?;
///
/// // See [LanceDB indexing](https://lancedb.github.io/lancedb/concepts/index_ivfpq/#product-quantization) for more information
/// table
///     .create_index(
///         &["embedding"],
///         lancedb::index::Index::IvfPq(IvfPqIndexBuilder::default()),
///     )
///     .execute()
///     .await?;
///
/// // Define search_params params that will be used by the vector store to perform the vector search.
/// let search_params = SearchParams::default();
/// let vector_store = LanceDbVectorIndex::new(table, model, "id", search_params).await?;
///
/// // Query the index
/// let results = vector_store
///    .top_n::<FakeDefinition>("My boss says I zindle too much, what does that mean?", 1)
///    .await?;
///
/// println!("Results: {:?}", results);
/// ```
pub struct LanceDbVectorIndex<M: EmbeddingModel, T> {
    _t: PhantomData<T>,
    /// Defines which model is used to generate embeddings for the vector store.
    model: M,
    /// LanceDB table containing embeddings.
    table: lancedb::Table,
    /// Column name in `table` that contains the id of a record.
    id_field: String,
    /// Vector search params that are used during vector search operations.
    search_params: SearchParams,
}

impl<M: EmbeddingModel, T: for<'a> Deserialize<'a>> LanceDbVectorIndex<M, T> {
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
            _t: PhantomData,
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

/// Parameters used to perform a vector search on a LanceDb table.
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

impl<M: EmbeddingModel + Sync + Send, T: for<'a> Deserialize<'a> + Sync + Send> VectorStoreIndex<T>
    for LanceDbVectorIndex<M, T>
{
    async fn top_n(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_document(query).await?;

        let query = self
            .table
            .vector_search(prompt_embedding.vec.clone())
            .map_err(lancedb_to_rig_error)?
            .limit(n);

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

    async fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_document(query).await?;

        let query = self
            .table
            .query()
            .select(lancedb::query::Select::Columns(vec![self.id_field.clone()]))
            .nearest_to(prompt_embedding.vec.clone())
            .map_err(lancedb_to_rig_error)?
            .limit(n);

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
