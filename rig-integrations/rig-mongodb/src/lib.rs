use futures::StreamExt;
use mongodb::bson::{self, Bson, Document, doc};

use rig::{
    Embed, OneOrMany,
    embeddings::embedding::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{SearchFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize};

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
/// # Example
/// ```rust
/// use rig_mongodb::{MongoDbVectorIndex, SearchParams};
/// use rig::{providers::openai, vector_store::{VectorStoreIndex, VectorSearchRequest}, client::{ProviderClient, EmbeddingsClient}};
///
/// # tokio_test::block_on(async {
/// #[derive(serde::Deserialize, serde::Serialize, Debug)]
/// struct WordDefinition {
///     #[serde(rename = "_id")]
///     id: String,
///     definition: String,
///     embedding: Vec<f64>,
/// }
///
/// let mongodb_client = mongodb::Client::with_uri_str("mongodb://localhost:27017").await?; // <-- replace with your mongodb uri.
/// let openai_client = openai::Client::from_env();
///
/// let collection = mongodb_client.database("db").collection::<WordDefinition>(""); // <-- replace with your mongodb collection.
///
/// let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002); // <-- replace with your embedding model.
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
///     .build()
///     .unwrap();
///
/// // Query the index
/// let definitions = index
///     .top_n::<WordDefinition>(req)
///     .await?;
/// # Ok::<_, anyhow::Error>(())
/// # }).unwrap()
/// ```
pub struct MongoDbVectorIndex<C, M>
where
    C: Send + Sync,
    M: EmbeddingModel,
{
    collection: mongodb::Collection<C>,
    model: M,
    index_name: String,
    embedded_field: String,
    search_params: SearchParams,
}

impl<C, M> MongoDbVectorIndex<C, M>
where
    C: Send + Sync,
    M: EmbeddingModel,
{
    /// Vector search stage of aggregation pipeline of mongoDB collection.
    /// To be used by implementations of top_n and top_n_ids methods on VectorStoreIndex trait for MongoDbVectorIndex.
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
    /// /// To be used by implementations of top_n and top_n_ids methods on VectorStoreIndex trait for MongoDbVectorIndex.
    fn pipeline_score_stage(&self) -> bson::Document {
        doc! {
          "$addFields": {
            "score": { "$meta": "vectorSearchScore" }
          }
        }
    }
}

impl<C, M> MongoDbVectorIndex<C, M>
where
    M: EmbeddingModel,
    C: Send + Sync,
{
    /// Create a new `MongoDbVectorIndex`.
    ///
    /// The index (of type "vector") must already exist for the MongoDB collection.
    /// See the MongoDB [documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/) for more information on creating indexes.
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
            // This error shouldn't occur if the index is queryable
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

    fn eq(key: String, value: Self::Value) -> Self {
        Self(doc! { key: value })
    }

    fn gt(key: String, value: Self::Value) -> Self {
        Self(doc! { key: { "$gt": value } })
    }

    fn lt(key: String, value: Self::Value) -> Self {
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

impl<C, M> VectorStoreIndex for MongoDbVectorIndex<C, M>
where
    C: Sync + Send,
    M: EmbeddingModel + Sync + Send,
{
    type Filter = MongoDbSearchFilter;

    /// Implement the `top_n` method of the `VectorStoreIndex` trait for `MongoDbVectorIndex`.
    ///
    /// `VectorSearchRequest` similarity search threshold filter gets ignored here because it is already present and can already be added in the MongoDB vector store struct.
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<MongoDbSearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_text(req.query()).await?;

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
            let score = doc.get("score").expect("score").as_f64().expect("f64");
            let id = doc.get("_id").expect("_id").to_string();
            let doc_t: T = serde_json::from_value(doc).map_err(VectorStoreError::JsonError)?;
            results.push((score, id, doc_t));
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

    /// Implement the `top_n_ids` method of the `VectorStoreIndex` trait for `MongoDbVectorIndex`.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<MongoDbSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_text(req.query()).await?;

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
            let score = doc.get("score").expect("score").as_f64().expect("f64");
            let id = doc.get("_id").expect("_id").to_string();
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
}

impl<C, M> InsertDocuments for MongoDbVectorIndex<C, M>
where
    C: Send + Sync,
    M: EmbeddingModel + Send + Sync,
{
    async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let mongo_documents = documents
            .into_iter()
            .map(|(document, embeddings)| -> Result<Vec<mongodb::bson::Document>, VectorStoreError> {
                let json_doc = serde_json::to_value(&document)?;

                embeddings.into_iter().map(|embedding| -> Result<mongodb::bson::Document, VectorStoreError> {
                    Ok(doc! {
                        "document": mongodb::bson::to_bson(&json_doc).map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?,
                        "embedding": embedding.vec,
                        "embedded_text": embedding.document,
                    })
                }).collect::<Result<Vec<_>, _>>()
            })
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
}
