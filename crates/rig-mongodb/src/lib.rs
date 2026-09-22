//! MongoDB vector store for Rig.
//!
//! [`MongoDbVectorIndex`] queries an existing MongoDB Atlas Vector Search index
//! through an aggregation pipeline, filtered by [`MongoDbSearchFilter`]. The
//! `rig` facade re-exports this crate as `rig::mongodb` under the `mongodb`
//! feature.

use futures::StreamExt;
use mongodb::bson::{self, Bson, Document, doc, to_bson};

use rig_core::{
    Embed,
    embeddings::embedding::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{DynamicSearchFilter, Filter, FilterError, SearchFilter, VectorSearchRequest},
    },
    wasm_compat::WasmCompatSend,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchIndex {
    id: String,
    name: String,
    #[serde(rename = "type")]
    index_type: String,
    status: String,
    queryable: bool,
    latest_definition: LatestDefinition,
}

impl SearchIndex {
    async fn get_search_index<C: Send + Sync>(
        collection: mongodb::Collection<C>,
        index_name: &str,
    ) -> Result<SearchIndex, VectorStoreError> {
        collection
            .list_search_indexes()
            .name(index_name)
            .await
            .map_err(VectorStoreError::datastore)?
            .with_type::<SearchIndex>()
            .next()
            .await
            .transpose()
            .map_err(VectorStoreError::datastore)?
            .ok_or_else(|| VectorStoreError::DatastoreError("Index not found".into()))
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct LatestDefinition {
    fields: Vec<Field>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Field {
    #[serde(rename = "type")]
    field_type: String,
    path: String,
    num_dimensions: i32,
    similarity: String,
}

/// Vector index over a MongoDB collection.
///
/// Queries are embedded with the same model `M` that populated the collection,
/// so results are meaningless under another model.
///
/// # Example
/// ```no_run
/// use rig_mongodb::{MongoDbVectorIndex, SearchParams};
/// use rig_core::{providers::openai::{self, wire::OpenAI}, vector_store::{VectorStoreIndex, VectorSearchRequest}};
/// use rig_reqwest::prelude::*;
///
/// # async fn example() -> anyhow::Result<()> {
/// #[derive(serde::Deserialize, serde::Serialize, Debug)]
/// struct WordDefinition {
///     #[serde(rename = "_id")]
///     id: String,
///     definition: String,
///     embedding: Vec<f64>,
/// }
///
/// let mongodb_client = mongodb::Client::with_uri_str("mongodb://localhost:27017").await?; // <-- replace with your mongodb uri.
/// let openai = OpenAI::from_env()?.bound()?;
///
/// let collection = mongodb_client.database("db").collection::<WordDefinition>(""); // <-- replace with your mongodb collection.
///
/// let model = openai.embedding(openai::TEXT_EMBEDDING_ADA_002, None); // <-- replace with your embedding model.
/// let index = MongoDbVectorIndex::new(
///     collection,
///     model,
///     "vector_index", // <-- replace with the name of the index in your mongodb collection.
///     SearchParams::new(), // <-- field name in `Document` that contains the embeddings.
/// )
/// .await?;
///
/// let req = VectorSearchRequest::builder()
///     .query("My boss says I zindle too much, what does that mean?")
///     .samples(1)
///     .build();
///
/// // Query the index
/// let definitions = index
///     .top_n::<WordDefinition>(req)
///     .await?;
/// # Ok(())
/// # }
/// # let _ = example();
/// ```
pub struct MongoDbVectorIndex<C, M>
where
    C: Send + Sync,
{
    collection: mongodb::Collection<C>,
    model: M,
    index_name: String,
    embedded_field: String,
    search_params: SearchParams,
}

impl<C, M: EmbeddingModel> MongoDbVectorIndex<C, M>
where
    C: Send + Sync,
{
    /// Builds the `$vectorSearch` stage. Any request threshold becomes a
    /// `score >= threshold` condition combined with the request filter, and
    /// `numCandidates` defaults to ten times the requested sample count.
    fn pipeline_search_stage(
        &self,
        prompt_embedding: &Embedding,
        req: &VectorSearchRequest<MongoDbSearchFilter>,
    ) -> bson::Document {
        let SearchParams {
            exact,
            num_candidates,
        } = &self.search_params;

        let samples = req.samples() as usize;

        let thresh = req
            .threshold()
            .map(|thresh| MongoDbSearchFilter::gte("score", thresh.into()));

        let filter = match (thresh, req.filter()) {
            (Some(thresh), Some(filt)) => thresh.and(filt.clone()).into_inner(),
            (Some(thresh), _) => thresh.into_inner(),
            (_, Some(filt)) => filt.clone().into_inner(),
            _ => Default::default(),
        };

        doc! {
          "$vectorSearch": {
            "index": &self.index_name,
            "path": self.embedded_field.clone(),
            "queryVector": &prompt_embedding.vec,
            "numCandidates": num_candidates.unwrap_or((samples * 10) as u32),
            "limit": samples as u32,
            "filter": filter,
            "exact": exact.unwrap_or(false)
          }
        }
    }

    /// Embeds the query and runs the pipeline with the given `$project` stage.
    /// Errors when a result lacks a numeric `score` or an `_id`, whose BSON
    /// rendering (including quotes for strings) becomes the returned id.
    async fn run_search_pipeline(
        &self,
        req: &VectorSearchRequest<MongoDbSearchFilter>,
        project_stage: bson::Document,
    ) -> Result<Vec<(f64, String, serde_json::Value)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_text(req.query()).await?;

        let pipeline = vec![
            self.pipeline_search_stage(&prompt_embedding, req),
            self.pipeline_score_stage(),
            project_stage,
        ];

        let mut cursor = self
            .collection
            .aggregate(pipeline)
            .await
            .map_err(VectorStoreError::datastore)?
            .with_type::<serde_json::Value>();

        let mut results = Vec::new();
        while let Some(doc) = cursor.next().await {
            let doc = doc.map_err(VectorStoreError::datastore)?;
            let score = doc
                .get("score")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    VectorStoreError::DatastoreError(
                        "MongoDB vector search result missing numeric score".into(),
                    )
                })?;
            let id = doc
                .get("_id")
                .ok_or_else(|| {
                    VectorStoreError::MissingIdError(
                        "MongoDB vector search result missing _id".to_string(),
                    )
                })?
                .to_string();
            results.push((score, id, doc));
        }

        tracing::info!(target: "rig",
            "Selected documents: {}",
            results.iter()
                .map(|(distance, id, _)| format!("{id} ({distance})"))
                .collect::<Vec<String>>()
                .join(", ")
        );

        Ok(results)
    }

    /// Builds the stage exposing the vector search score as a `score` field.
    fn pipeline_score_stage(&self) -> bson::Document {
        doc! {
          "$addFields": {
            "score": { "$meta": "vectorSearchScore" }
          }
        }
    }
}

impl<C, M: EmbeddingModel> MongoDbVectorIndex<C, M>
where
    C: Send + Sync,
{
    /// Creates an index handle after confirming the named search index exists and
    /// is queryable. The embedded field is taken from the index's first defined
    /// field. Errors when the index is missing, not queryable, or defines no
    /// fields. See the MongoDB [documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/)
    /// on creating vector indexes.
    pub async fn new(
        collection: mongodb::Collection<C>,
        model: M,
        index_name: &str,
        search_params: SearchParams,
    ) -> Result<Self, VectorStoreError> {
        let search_index = SearchIndex::get_search_index(collection.clone(), index_name).await?;

        if !search_index.queryable {
            return Err(VectorStoreError::DatastoreError(
                "Index is not queryable".into(),
            ));
        }

        let embedded_field = search_index
            .latest_definition
            .fields
            .into_iter()
            .map(|field| field.path)
            .next()
            .ok_or(VectorStoreError::DatastoreError(
                "No embedded fields found".into(),
            ))?;

        Ok(Self {
            collection,
            model,
            index_name: index_name.to_string(),
            embedded_field,
            search_params,
        })
    }
}

/// Backend search tuning for the `$vectorSearch` stage. See
/// [MongoDB Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/).
#[derive(Default)]
pub struct SearchParams {
    exact: Option<bool>,
    num_candidates: Option<u32>,
}

impl SearchParams {
    /// Initializes a new `SearchParams` with default values.
    pub fn new() -> Self {
        Self {
            exact: None,
            num_candidates: None,
        }
    }

    /// Selects exact (ENN) search instead of the default approximate (ANN) search.
    pub fn exact(mut self, exact: bool) -> Self {
        self.exact = Some(exact);
        self
    }

    /// Sets how many nearest neighbors approximate search considers. MongoDB
    /// rejects this alongside exact search.
    pub fn num_candidates(mut self, num_candidates: u32) -> Self {
        self.num_candidates = Some(num_candidates);
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MongoDbSearchFilter(Document);

impl SearchFilter for MongoDbSearchFilter {
    type Value = Bson;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        let key = key.as_ref().to_owned();
        Self(doc! { key: value })
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        let key = key.as_ref().to_owned();
        Self(doc! { key: { "$gt": value } })
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        let key = key.as_ref().to_owned();
        Self(doc! { key: { "$lt": value } })
    }

    fn and(self, rhs: Self) -> Self {
        Self(doc! { "$and": [ self.0, rhs.0 ]})
    }

    fn or(self, rhs: Self) -> Self {
        Self(doc! { "$or": [ self.0, rhs.0 ]})
    }
}

impl MongoDbSearchFilter {
    fn into_inner(self) -> Document {
        self.0
    }

    pub fn gte(key: impl Into<String>, value: <Self as SearchFilter>::Value) -> Self {
        let key = key.into();
        Self(doc! { key: { "$gte": value } })
    }

    pub fn lte(key: impl Into<String>, value: <Self as SearchFilter>::Value) -> Self {
        let key = key.into();
        Self(doc! { key: { "$lte": value } })
    }

    pub fn not(self) -> Self {
        Self(doc! { "$nor": [self.0] })
    }

    /// Matches values at `key` whose BSON type alias is `typ`.
    pub fn is_type(key: impl Into<String>, typ: &'static str) -> Self {
        let key = key.into();
        Self(doc! { key: { "$type": typ } })
    }

    pub fn size(key: impl Into<String>, size: i32) -> Self {
        let key = key.into();
        Self(doc! { key: { "$size": size } })
    }

    /// Matches arrays at `key` containing every one of `values`.
    pub fn all(key: impl Into<String>, values: Vec<Bson>) -> Self {
        let key = key.into();
        Self(doc! { key: { "$all": values } })
    }

    /// Matches arrays at `key` with at least one element satisfying `condition`.
    pub fn any(key: impl Into<String>, condition: Document) -> Self {
        let key = key.into();
        Self(doc! { key: { "$elemMatch": condition } })
    }
}

impl From<Filter<serde_json::Value>> for MongoDbSearchFilter {
    /// Values that cannot be represented in BSON become `Bson::Null`.
    fn from(value: Filter<serde_json::Value>) -> Self {
        value.interpret_with(|v| to_bson(&v).unwrap_or(Bson::Null))
    }
}

impl DynamicSearchFilter for MongoDbSearchFilter {
    fn from_dynamic_filter(filter: Filter<serde_json::Value>) -> Result<Self, FilterError> {
        Ok(filter.into())
    }
}

impl<C, M: EmbeddingModel> VectorStoreIndex for MongoDbVectorIndex<C, M>
where
    C: Sync + Send,
{
    type Filter = MongoDbSearchFilter;

    /// Returns matches as `(score, id, document)`. The embedding field is
    /// projected out, so `T` must not require it.
    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<MongoDbSearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let project_stage = doc! {
            "$project": {
                self.embedded_field.clone(): 0
            }
        };

        self.run_search_pipeline(&req, project_stage)
            .await?
            .into_iter()
            .map(|(score, id, doc)| {
                let doc_t: T = serde_json::from_value(doc).map_err(VectorStoreError::JsonError)?;
                Ok((score, id, doc_t))
            })
            .collect()
    }

    /// Like `top_n` but projects only the id and score.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<MongoDbSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let project_stage = doc! {
            "$project": {
                "_id": 1,
                "score": 1
            },
        };

        Ok(self
            .run_search_pipeline(&req, project_stage)
            .await?
            .into_iter()
            .map(|(score, id, _)| (score, id))
            .collect())
    }
}

impl<C, M: EmbeddingModel> InsertDocuments for MongoDbVectorIndex<C, M>
where
    C: Send + Sync,
{
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let mongo_documents = rig_core::vector_store::flatten_embedded(
            documents,
            |json_doc, embedding| {
                Ok(doc! {
                    "document": mongodb::bson::to_bson(json_doc).map_err(VectorStoreError::datastore)?,
                    "embedding": embedding.vec,
                    "embedded_text": embedding.document,
                })
            },
        )?;

        let collection = self.collection.clone_with_type::<mongodb::bson::Document>();

        collection
            .insert_many(mongo_documents)
            .await
            .map_err(VectorStoreError::datastore)?;

        Ok(())
    }
}
