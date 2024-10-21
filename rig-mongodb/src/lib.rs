use std::marker::PhantomData;

use futures::StreamExt;
use mongodb::bson::{self, doc};

use rig::{
    embeddings::embedding::{Embedding, EmbeddingModel},
    vector_store::{VectorStoreError, VectorStoreIndex},
};
use serde::Deserialize;

fn mongodb_to_rig_error(e: mongodb::error::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

/// A vector index on a MongoDB collection.
/// # Example
/// ```
/// use mongodb::{bson::doc, options::ClientOptions, Client as MongoClient, Collection};
/// use rig::providers::openai::TEXT_EMBEDDING_ADA_002;
/// use serde::{Deserialize, Serialize};
/// use std::env;
///
/// use rig::Embeddable;
/// use rig::{
///     embeddings::EmbeddingsBuilder, providers::openai::Client, vector_store::VectorStoreIndex,
/// };
/// use rig_mongodb::{MongoDbVectorStore, SearchParams};
///
/// // Shape of data that needs to be RAG'ed.
/// // The definition field will be used to generate embeddings.
/// #[derive(Embeddable, Clone, Deserialize, Debug)]
/// struct FakeDefinition {
///     #[serde(rename = "_id")]
///     id: String,
///     #[embed]
///     definition: String,
/// }
///
/// #[derive(Clone, Deserialize, Debug, Serialize)]
/// struct Link {
///     word: String,
///     link: String,
/// }
///
/// // Shape of the document to be stored in MongoDB, with embeddings.
/// #[derive(Serialize, Debug)]
/// struct Document {
///     #[serde(rename = "_id")]
///     id: String,
///     definition: String,
///     embedding: Vec<f64>,
/// }
/// // Initialize OpenAI client
/// let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
/// let openai_client = Client::new(&openai_api_key);
///
/// // Initialize MongoDB client
/// let mongodb_connection_string =
///     env::var("MONGODB_CONNECTION_STRING").expect("MONGODB_CONNECTION_STRING not set");
/// let options = ClientOptions::parse(mongodb_connection_string)
///     .await
///     .expect("MongoDB connection string should be valid");
///
/// let mongodb_client =
///     MongoClient::with_options(options).expect("MongoDB client options should be valid");
///
/// // Initialize MongoDB vector store
/// let collection: Collection<Document> = mongodb_client
///     .database("knowledgebase") // <-- Use your database name here!
///     .collection("context"); // <-- Use your collection name here!
///
/// // Select the embedding model and generate our embeddings
/// let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
///
/// let fake_definitions = vec![
///     FakeDefinition {
///         id: "doc0".to_string(),
///         definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
///     },
///     FakeDefinition {
///         id: "doc1".to_string(),
///         definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
///     },
///     FakeDefinition {
///         id: "doc2".to_string(),
///         definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
///     }
/// ];
///
/// let embeddings = EmbeddingsBuilder::new(model.clone())
///     .documents(fake_definitions)?
///     .build()
///     .await?;
///
/// let mongo_documents = embeddings
///     .iter()
///     .map(
///         |(FakeDefinition { id, definition, .. }, embedding)| Document {
///             id: id.clone(),
///             definition: definition.clone(),
///             embedding: embedding.first().vec.clone(),
///         },
///     )
///     .collect::<Vec<_>>();
///
/// match collection.insert_many(mongo_documents, None).await {
///     Ok(_) => println!("Documents added successfully"),
///     Err(e) => println!("Error adding documents: {:?}", e),
/// };
///
/// // Note: an index of type vector called "vector_index" must exist on the MongoDB collection you are querying.
/// // IMPORTANT: Reuse the same model that was used to generate the embeddings
/// let index = MongoDbVectorIndex::<_, _, FakeDefinition>::new(
///     model,
///     collection,
///     "vector_index",
///     SearchParams::new("embedding"),
/// );
///
/// // Query the index
/// let results = index
///     .top_n::<FakeDefinition>("What is a linglingdong?", 1)
///     .await?;
///
/// println!("Results: {:?}", results);
///
/// let id_results = index
///     .top_n_ids("What is a linglingdong?", 1)
///     .await?
///     .into_iter()
///     .map(|(score, id)| (score, id))
///     .collect::<Vec<_>>();
///
/// println!("ID results: {:?}", id_results);
/// ```
pub struct MongoDbVectorIndex<M, I, T> {
    _t: PhantomData<T>,
    collection: mongodb::Collection<I>,
    model: M,
    index_name: String,
    search_params: SearchParams,
}

impl<M: EmbeddingModel, I, T: for<'a> Deserialize<'a>> MongoDbVectorIndex<M, I, T> {
    /// Create a new `MongoDbVectorIndex` from a mongodb collection and index.
    /// Note: this is a rig concept, NOT a mongoDB cloud concept.
    /// Make sure you have a vector index on your Mongodb collection called `index_name`.
    ///
    /// See the MongoDB [documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/) for more information on creating indexes.
    pub fn new(
        model: M,
        collection: mongodb::Collection<I>,
        index_name: &str,
        search_params: SearchParams,
    ) -> MongoDbVectorIndex<M, I, T> {
        Self {
            _t: PhantomData,
            collection,
            model,
            index_name: index_name.to_string(),
            search_params,
        }
    }

    /// Vector search stage of aggregation pipeline of mongoDB collection.
    /// To be used by implementations of top_n and top_n_ids methods on VectorStoreIndex trait for MongoDbVectorIndex.
    fn pipeline_search_stage(&self, prompt_embedding: &Embedding, n: usize) -> bson::Document {
        let SearchParams {
            filter,
            exact,
            num_candidates,
            path,
        } = &self.search_params;

        doc! {
          "$vectorSearch": {
            "index": &self.index_name,
            "path": path,
            "queryVector": &prompt_embedding.vec,
            "numCandidates": num_candidates.unwrap_or((n * 10) as u32),
            "limit": n as u32,
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

/// See [MongoDB Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/) for more information
/// on each of the fields
pub struct SearchParams {
    filter: mongodb::bson::Document,
    path: String,
    exact: Option<bool>,
    num_candidates: Option<u32>,
}

impl SearchParams {
    /// Initializes a new `SearchParams` with default values.
    pub fn new(path: &str) -> Self {
        Self {
            filter: doc! {},
            exact: None,
            num_candidates: None,
            path: path.to_string(),
        }
    }

    /// Sets the pre-filter field of the search params.
    /// See [MongoDB vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/) for more information.
    pub fn filter(mut self, filter: mongodb::bson::Document) -> Self {
        self.filter = filter;
        self
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

impl<M: EmbeddingModel + Sync + Send, I: Sync + Send, T: for<'a> Deserialize<'a> + Sync + Send>
    VectorStoreIndex<T> for MongoDbVectorIndex<M, I, T>
{
    async fn top_n(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_document(query).await?;

        let mut cursor = self
            .collection
            .aggregate(
                [
                    self.pipeline_search_stage(&prompt_embedding, n),
                    self.pipeline_score_stage(),
                ],
                None,
            )
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
                .map(|(distance, id, _)| format!("{} ({})", id, distance))
                .collect::<Vec<String>>()
                .join(", ")
        );

        Ok(results)
    }

    async fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let prompt_embedding = self.model.embed_document(query).await?;

        let mut cursor = self
            .collection
            .aggregate(
                [
                    self.pipeline_search_stage(&prompt_embedding, n),
                    self.pipeline_score_stage(),
                    doc! {
                        "$project": {
                            "_id": 1,
                            "score": 1
                        },
                    },
                ],
                None,
            )
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
                .map(|(distance, id)| format!("{} ({})", id, distance))
                .collect::<Vec<String>>()
                .join(", ")
        );

        Ok(results)
    }
}
