//! MongoDB vector store integration for Rig.
//!
//! This crate provides [`MongoDbVectorIndex`], a Rig vector store backed
//! by MongoDB Atlas Vector Search or compatible MongoDB vector search indexes.
//!
//! Queries arrive pre-embedded via [`VectorSearchRequest`]; the store never
//! embeds text itself.
//!
//! The root `rig` facade re-exports this crate as `rig::mongodb` when the
//! `mongodb` feature is enabled.

use futures::StreamExt;
use mongodb::bson::{self, Bson, Document, doc, to_bson};

use rig_core::{
    OneOrMany,
    embeddings::embedding::Embedding,
    vector_store::{
        SearchHit, StoreRecord, VectorStoreError,
        request::{Filter, SearchFilter, VectorSearchRequest},
    },
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
            .map_err(mongodb_to_rig_error)?
            .with_type::<SearchIndex>()
            .next()
            .await
            .transpose()
            .map_err(mongodb_to_rig_error)?
            .ok_or(VectorStoreError::DatastoreError("Index not found".into()))
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

fn mongodb_to_rig_error(e: mongodb::error::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

/// A vector index for a MongoDB collection.
///
/// Queries arrive pre-embedded via [`VectorSearchRequest`]; the index never
/// embeds text itself.
///
/// # Example
/// ```no_run
/// use rig_mongodb::{MongoDbVectorIndex, MongoDbSearchFilter, SearchParams};
/// use rig_core::{providers::openai, vector_store::VectorSearchRequest, client::{ProviderClient, EmbeddingsClient}, embeddings::embedding::EmbeddingModel};
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
/// let openai_client = openai::Client::from_env()?;
///
/// let collection = mongodb_client.database("db").collection::<WordDefinition>(""); // <-- replace with your mongodb collection.
///
/// let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002); // <-- replace with your embedding model.
/// let index = MongoDbVectorIndex::new(
///     collection,
///     "vector_index", // <-- replace with the name of the index in your mongodb collection.
///     SearchParams::new(),
/// )
/// .await?;
///
/// // Embed the query outside the store, then search with the pre-embedded query.
/// let query = model
///     .embed_text("My boss says I zindle too much, what does that mean?")
///     .await?;
/// let req = VectorSearchRequest::<MongoDbSearchFilter>::builder()
///     .query(query)
///     .samples(1)
///     .build();
///
/// // Query the index
/// let definitions = index
///     .top_n_as::<WordDefinition>(req)
///     .await?;
/// # Ok(())
/// # }
/// # let _ = example();
/// ```
pub struct MongoDbVectorIndex<C>
where
    C: Send + Sync,
{
    collection: mongodb::Collection<C>,
    index_name: String,
    embedded_field: String,
    search_params: SearchParams,
}

impl<C> MongoDbVectorIndex<C>
where
    C: Send + Sync,
{
    /// Vector search stage of aggregation pipeline of mongoDB collection.
    /// Used by the `top_n` and `top_n_ids` methods on [`MongoDbVectorIndex`].
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
            .map(|thresh| MongoDbSearchFilter::gte("score".into(), thresh.into()));

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

    /// Score declaration stage of aggregation pipeline of mongoDB collection.
    /// Used by the `top_n` and `top_n_ids` methods on [`MongoDbVectorIndex`].
    fn pipeline_score_stage(&self) -> bson::Document {
        doc! {
          "$addFields": {
            "score": { "$meta": "vectorSearchScore" }
          }
        }
    }

    /// Create a new `MongoDbVectorIndex`.
    ///
    /// The index (of type "vector") must already exist for the MongoDB collection.
    /// See the MongoDB [documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/) for more information on creating indexes.
    pub async fn new(
        collection: mongodb::Collection<C>,
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
            // This error shouldn't occur if the index is queryable
            .ok_or(VectorStoreError::DatastoreError(
                "No embedded fields found".into(),
            ))?;

        Ok(Self {
            collection,
            index_name: index_name.to_string(),
            embedded_field,
            search_params,
        })
    }

    /// Returns the top N most similar documents for a pre-embedded query.
    ///
    /// MongoDB `$vectorSearch` takes a single query vector, so only the first
    /// embedding of the request's query is used. Scores are Atlas
    /// `vectorSearchScore` similarity scores: higher is better.
    ///
    /// Each [`SearchHit`]'s payload is the full MongoDB document (minus the
    /// embedded field), including its `_id` and `score` fields.
    pub async fn top_n(
        &self,
        req: VectorSearchRequest<MongoDbSearchFilter>,
    ) -> Result<Vec<SearchHit>, VectorStoreError> {
        let prompt_embedding = req.query().first();

        let pipeline = vec![
            self.pipeline_search_stage(&prompt_embedding, &req),
            self.pipeline_score_stage(),
            doc! {
                "$project": {
                    self.embedded_field.clone(): 0
                }
            },
        ];

        let mut cursor = self
            .collection
            .aggregate(pipeline)
            .await
            .map_err(mongodb_to_rig_error)?
            .with_type::<serde_json::Value>();

        let mut results = Vec::new();
        while let Some(doc) = cursor.next().await {
            let doc = doc.map_err(mongodb_to_rig_error)?;
            let score = doc
                .get("score")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    VectorStoreError::DatastoreError(Box::new(std::io::Error::other(
                        "MongoDB vector search result missing numeric score",
                    )))
                })?;
            let id = doc.get("_id").ok_or_else(|| {
                VectorStoreError::DatastoreError(Box::new(std::io::Error::other(
                    "MongoDB vector search result missing _id",
                )))
            })?;
            let id = id.to_string();
            results.push(SearchHit {
                id,
                score,
                payload: doc,
            });
        }

        tracing::info!(target: "rig",
            "Selected documents: {}",
            results.iter()
                .map(|hit| format!("{} ({})", hit.id, hit.score))
                .collect::<Vec<String>>()
                .join(", ")
        );

        Ok(results)
    }

    /// Returns the top N most similar document IDs as `(score, id)` tuples.
    pub async fn top_n_ids(
        &self,
        req: VectorSearchRequest<MongoDbSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let prompt_embedding = req.query().first();

        let pipeline = vec![
            self.pipeline_search_stage(&prompt_embedding, &req),
            self.pipeline_score_stage(),
            doc! {
                "$project": {
                    "_id": 1,
                    "score": 1
                },
            },
        ];

        let mut cursor = self
            .collection
            .aggregate(pipeline)
            .await
            .map_err(mongodb_to_rig_error)?
            .with_type::<serde_json::Value>();

        let mut results = Vec::new();
        while let Some(doc) = cursor.next().await {
            let doc = doc.map_err(mongodb_to_rig_error)?;
            let score = doc
                .get("score")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    VectorStoreError::DatastoreError(Box::new(std::io::Error::other(
                        "MongoDB vector search result missing numeric score",
                    )))
                })?;
            let id = doc.get("_id").ok_or_else(|| {
                VectorStoreError::DatastoreError(Box::new(std::io::Error::other(
                    "MongoDB vector search result missing _id",
                )))
            })?;
            let id = id.to_string();
            results.push((score, id));
        }

        tracing::info!(target: "rig",
            "Selected documents: {}",
            results.iter()
                .map(|(distance, id)| format!("{id} ({distance})"))
                .collect::<Vec<String>>()
                .join(", ")
        );

        Ok(results)
    }

    /// Returns the top N most similar documents deserialized into `T` as
    /// `(score, id, document)` tuples. Sugar over [`Self::top_n`].
    pub async fn top_n_as<T: DeserializeOwned>(
        &self,
        req: VectorSearchRequest<MongoDbSearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.top_n(req)
            .await?
            .into_iter()
            .map(|hit| {
                let doc: T =
                    serde_json::from_value(hit.payload).map_err(VectorStoreError::JsonError)?;
                Ok((hit.score, hit.id, doc))
            })
            .collect()
    }

    /// Insert precomputed records into the collection.
    ///
    /// Each embedding of a record becomes one MongoDB document with the shape
    /// `{ "id": ..., "document": <payload>, "embedding": [...], "embedded_text": ... }`.
    ///
    /// Note that [`StoreRecord::id`] is stored in the `id` field only; search
    /// hits identify documents by the MongoDB-assigned `_id`, not this field.
    pub async fn insert(&self, records: Vec<StoreRecord>) -> Result<(), VectorStoreError> {
        let mongo_documents = records
            .into_iter()
            .map(
                |record| -> Result<Vec<mongodb::bson::Document>, VectorStoreError> {
                    let StoreRecord {
                        id,
                        payload,
                        embeddings,
                    } = record;

                    let payload = mongodb::bson::to_bson(&payload)
                        .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

                    embeddings
                        .into_iter()
                        .map(
                            |embedding| -> Result<mongodb::bson::Document, VectorStoreError> {
                                Ok(doc! {
                                    "id": id.clone(),
                                    "document": payload.clone(),
                                    "embedding": embedding.vec,
                                    "embedded_text": embedding.document,
                                })
                            },
                        )
                        .collect::<Result<Vec<_>, _>>()
                },
            )
            .collect::<Result<Vec<Vec<_>>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let collection = self.collection.clone_with_type::<mongodb::bson::Document>();

        collection
            .insert_many(mongo_documents)
            .await
            .map_err(mongodb_to_rig_error)?;

        Ok(())
    }

    /// Serializes each document and inserts it. Sugar over [`Self::insert`].
    pub async fn insert_as<T: Serialize>(
        &self,
        docs: Vec<(String, T, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let records = docs
            .into_iter()
            .map(|(id, doc, embeddings)| StoreRecord::new(id, &doc, embeddings))
            .collect::<Result<Vec<_>, _>>()?;
        self.insert(records).await
    }
}

/// See [MongoDB Vector Search](`https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/`) for more information
/// on each of the fields
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

    /// Sets the exact field of the search params.
    /// If exact is true, an ENN vector search will be performed, otherwise, an ANN search will be performed.
    /// By default, exact is false.
    /// See [MongoDB vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/) for more information.
    pub fn exact(mut self, exact: bool) -> Self {
        self.exact = Some(exact);
        self
    }

    /// Sets the num_candidates field of the search params.
    /// Only set this field if exact is set to false.
    /// Number of nearest neighbors to use during the search.
    /// See [MongoDB vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/) for more information.
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

    pub fn gte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self(doc! { key: { "$gte": value } })
    }

    pub fn lte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self(doc! { key: { "$lte": value } })
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(doc! { "$nor": [self.0] })
    }

    /// Tests whether the value at `key` is the BSON type `typ`
    pub fn is_type(key: String, typ: &'static str) -> Self {
        Self(doc! { key: { "$type": typ } })
    }

    pub fn size(key: String, size: i32) -> Self {
        Self(doc! { key: { "$size": size } })
    }

    // Array ops
    pub fn all(key: String, values: Vec<Bson>) -> Self {
        Self(doc! { key: { "$all": values } })
    }

    pub fn any(key: String, condition: Document) -> Self {
        Self(doc! { key: { "$elemMatch": condition } })
    }
}

impl From<Filter<serde_json::Value>> for MongoDbSearchFilter {
    fn from(value: Filter<serde_json::Value>) -> Self {
        fn serde_json_value_to_bson(v: &serde_json::Value) -> Bson {
            to_bson(v).unwrap_or(Bson::Null)
        }

        match value {
            Filter::Eq(k, val) => {
                let bson_val = serde_json_value_to_bson(&val);
                MongoDbSearchFilter::eq(k, bson_val)
            }
            Filter::Gt(k, val) => {
                let bson_val = serde_json_value_to_bson(&val);
                MongoDbSearchFilter::gt(k, bson_val)
            }
            Filter::Lt(k, val) => {
                let bson_val = serde_json_value_to_bson(&val);
                MongoDbSearchFilter::lt(k, bson_val)
            }
            Filter::And(l, r) => Self::from(*l).and(Self::from(*r)),
            Filter::Or(l, r) => Self::from(*l).or(Self::from(*r)),
        }
    }
}
