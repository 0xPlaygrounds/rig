//! Weaviate vector store for Rig.
//!
//! [`WeaviateVectorStore`] inserts objects through Weaviate's batch REST
//! endpoint and searches them with a GraphQL `nearVector` query, narrowed by a
//! [`WeaviateFilter`]. Embeddings come from a Rig embedding model rather than a
//! Weaviate vectorizer, and each object keeps its document as JSON text in a
//! `document` property. The `rig` facade re-exports this crate as
//! `rig::weaviate` under the `weaviate` feature.
//!
//! ```no_run
//! use rig_core::providers::openai::{self, OpenAI};
//! use rig_core::vector_store::{VectorStoreIndex, request::VectorSearchRequest};
//! use rig_weaviate::WeaviateVectorStore;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAI::from_env()?.embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
//! let store = WeaviateVectorStore::new(model, "http://localhost:8080", "Word")?;
//! let request = VectorSearchRequest::builder()
//!     .query("What is a flurbo?")
//!     .samples(2)
//!     .build();
//! let results = store.top_n::<serde_json::Value>(request).await?;
//! # let _ = results;
//! # Ok(())
//! # }
//! ```

mod filter;

pub use filter::{WeaviateFilter, WeaviateValue};

use std::collections::{BTreeMap, HashMap};

use rig_core::{
    DynModel, Embed,
    embeddings::Embedding,
    http_client::{self, Bytes, DynHttpClient, HttpClientExt, Request},
    operation,
    vector_store::{
        InsertDocuments, VectorSearchIdResult, VectorSearchResult, VectorStoreError,
        VectorStoreIndex, flatten_embedded, request::VectorSearchRequest,
    },
    wasm_compat::WasmCompatSend,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// Objects sent per batch request.
const BATCH_SIZE: usize = 100;

/// Why a Weaviate request failed after Weaviate answered it. These reach
/// callers as [`VectorStoreError::DatastoreError`].
#[derive(Debug, thiserror::Error)]
pub enum WeaviateError {
    /// The collection name is not a valid Weaviate class name.
    #[error("invalid Weaviate collection name {0:?}: use a letter, then letters, digits or `_`")]
    InvalidCollectionName(String),
    /// Weaviate rejected objects of a batch insert. Objects of the same batch
    /// without errors were stored.
    #[error("Weaviate rejected {} object(s): {}", .0.len(), .0.join("; "))]
    Rejected(Vec<String>),
    /// The GraphQL search reported errors.
    #[error("Weaviate search failed: {}", .0.join("; "))]
    Query(Vec<String>),
    /// A search result carries no `document` property.
    #[error("Weaviate object {0} has no `document` property")]
    MissingDocument(String),
}

/// Vector store backed by a [Weaviate](https://weaviate.io/) collection.
///
/// Queries are embedded with the same model that populated the collection, so
/// results are meaningless under another model. Scores are `1 - distance`,
/// which under Weaviate's default cosine metric is the cosine similarity; a
/// request threshold is a minimum score.
#[derive(Clone)]
pub struct WeaviateVectorStore {
    model: DynModel<operation::Embedding>,
    http: DynHttpClient,
    base_url: String,
    collection: String,
    api_key: Option<String>,
    target_vector: Option<String>,
}

impl std::fmt::Debug for WeaviateVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeaviateVectorStore")
            .field("base_url", &self.base_url)
            .field("collection", &self.collection)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("target_vector", &self.target_vector)
            .finish_non_exhaustive()
    }
}

impl WeaviateVectorStore {
    /// A store over `collection` at `base_url` (such as
    /// `http://localhost:8080`), sending through the shared reqwest client.
    ///
    /// # Errors
    ///
    /// See [`WeaviateVectorStore::with_http`].
    #[cfg(feature = "reqwest")]
    pub fn new(
        model: impl Into<DynModel<operation::Embedding>>,
        base_url: impl Into<String>,
        collection: impl AsRef<str>,
    ) -> Result<Self, WeaviateError> {
        Self::with_http(rig_reqwest::shared(), model, base_url, collection)
    }

    /// A store over `collection` at `base_url`, sending through `http`.
    ///
    /// Weaviate capitalizes collection names, so `word` addresses the `Word`
    /// collection. Inserting into a missing collection creates it when the
    /// server's auto-schema is enabled, its default.
    ///
    /// # Errors
    ///
    /// When `collection` is not a letter followed by letters, digits or `_`.
    pub fn with_http(
        http: impl HttpClientExt + 'static,
        model: impl Into<DynModel<operation::Embedding>>,
        base_url: impl Into<String>,
        collection: impl AsRef<str>,
    ) -> Result<Self, WeaviateError> {
        let collection = collection_name(collection.as_ref())?;
        let base_url = base_url.into().trim_end_matches('/').to_owned();
        Ok(Self {
            model: model.into(),
            http: DynHttpClient::new(http),
            base_url,
            collection,
            api_key: None,
            target_vector: None,
        })
    }

    /// Sends `key` as a bearer token, as Weaviate Cloud and API-key
    /// authentication expect.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Writes and searches the named vector `name`. Without one, objects use
    /// the collection's only vector.
    pub fn target_vector(mut self, name: impl Into<String>) -> Self {
        self.target_vector = Some(name.into());
        self
    }

    /// The collection name as Weaviate spells it.
    pub fn collection(&self) -> &str {
        &self.collection
    }

    /// Posts `body` as JSON to `path` and decodes the reply. Non-success
    /// statuses arrive as [`VectorStoreError::Http`].
    async fn post<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &impl Serialize,
    ) -> Result<T, VectorStoreError> {
        let mut request = Request::post(format!("{}{path}", self.base_url))
            .header("content-type", "application/json");
        if let Some(key) = &self.api_key {
            request = request.header("authorization", format!("Bearer {key}"));
        }
        let request = request
            .body(serde_json::to_vec(body)?)
            .map_err(http_client::Error::from)?;
        let reply = self
            .http
            .send::<_, Bytes>(request)
            .await?
            .into_body()
            .await?;
        Ok(serde_json::from_slice(&reply)?)
    }

    /// Embeds the query and runs the search, returning the selected fields
    /// of each hit, nearest first.
    async fn search(
        &self,
        req: &VectorSearchRequest<WeaviateFilter>,
        with_document: bool,
    ) -> Result<Vec<Hit>, VectorStoreError> {
        let embedding = self.model.embed_text(req.query()).await?;
        let query = self.graphql(&embedding.vec, req, with_document)?;
        let reply: GraphQlReply = self.post("/v1/graphql", &GraphQlRequest { query }).await?;
        if !reply.errors.is_empty() {
            let messages = reply.errors.into_iter().map(|e| e.message).collect();
            return Err(VectorStoreError::datastore(WeaviateError::Query(messages)));
        }
        Ok(reply
            .data
            .and_then(|mut data| data.get.remove(&self.collection))
            .flatten()
            .unwrap_or_default())
    }

    /// The GraphQL `Get` query for `vector`. A threshold becomes the maximum
    /// distance `1 - threshold`.
    fn graphql(
        &self,
        vector: &[f64],
        req: &VectorSearchRequest<WeaviateFilter>,
        with_document: bool,
    ) -> Result<String, VectorStoreError> {
        let mut near = format!("vector: {}", serde_json::to_string(vector)?);
        if let Some(threshold) = req.threshold() {
            near.push_str(&format!(", distance: {}", 1.0 - threshold));
        }
        if let Some(name) = &self.target_vector {
            near.push_str(&format!(
                ", targetVectors: [{}]",
                serde_json::to_string(name)?
            ));
        }
        let mut arguments = format!("nearVector: {{{near}}}, limit: {}", req.samples());
        if let Some(filter) = req.filter() {
            arguments.push_str(&format!(", where: {}", filter.to_graphql()?));
        }
        let fields = if with_document {
            "document _additional { id distance }"
        } else {
            "_additional { id distance }"
        };
        Ok(format!(
            "{{ Get {{ {}({arguments}) {{ {fields} }} }} }}",
            self.collection
        ))
    }
}

/// `name` with its first letter capitalized, or an error when it is not a
/// Weaviate class name. It is spliced into GraphQL, so nothing else passes.
fn collection_name(name: &str) -> Result<String, WeaviateError> {
    let mut chars = name.chars();
    let valid = chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !valid {
        return Err(WeaviateError::InvalidCollectionName(name.to_owned()));
    }
    let mut chars = name.chars();
    Ok(chars
        .next()
        .map(|first| first.to_ascii_uppercase().to_string() + chars.as_str())
        .unwrap_or_default())
}

#[derive(Serialize)]
struct GraphQlRequest {
    query: String,
}

#[derive(Deserialize)]
struct GraphQlReply {
    data: Option<GetData>,
    #[serde(default)]
    errors: Vec<Message>,
}

#[derive(Deserialize)]
struct GetData {
    #[serde(rename = "Get")]
    get: HashMap<String, Option<Vec<Hit>>>,
}

#[derive(Deserialize)]
struct Hit {
    document: Option<String>,
    #[serde(rename = "_additional")]
    additional: Additional,
}

#[derive(Deserialize)]
struct Additional {
    id: String,
    distance: f64,
}

#[derive(Deserialize)]
struct Message {
    message: String,
}

#[derive(Serialize)]
struct Batch<'a> {
    objects: &'a [BatchObject<'a>],
}

#[derive(Serialize)]
struct BatchObject<'a> {
    class: &'a str,
    properties: Properties,
    #[serde(skip_serializing_if = "Option::is_none")]
    vector: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vectors: Option<BTreeMap<&'a str, Vec<f64>>>,
}

#[derive(Serialize)]
struct Properties {
    document: String,
}

#[derive(Deserialize)]
struct BatchReply {
    result: Option<BatchResult>,
}

#[derive(Deserialize)]
struct BatchResult {
    errors: Option<BatchErrors>,
}

#[derive(Deserialize)]
struct BatchErrors {
    #[serde(default)]
    error: Vec<Message>,
}

impl InsertDocuments for WeaviateVectorStore {
    /// Stores one object per embedding, in batches of 100. An error stops at
    /// the batch that failed; earlier batches stay stored.
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let objects = flatten_embedded(documents, |document, embedding| {
            let (vector, vectors) = match &self.target_vector {
                Some(name) => (None, Some(BTreeMap::from([(name.as_str(), embedding.vec)]))),
                None => (Some(embedding.vec), None),
            };
            Ok(BatchObject {
                class: &self.collection,
                properties: Properties {
                    document: serde_json::to_string(document)?,
                },
                vector,
                vectors,
            })
        })?;

        for objects in objects.chunks(BATCH_SIZE) {
            let replies: Vec<BatchReply> =
                self.post("/v1/batch/objects", &Batch { objects }).await?;
            let rejected: Vec<String> = replies
                .into_iter()
                .filter_map(|reply| reply.result?.errors)
                .flat_map(|errors| errors.error)
                .map(|error| error.message)
                .collect();
            if !rejected.is_empty() {
                return Err(VectorStoreError::datastore(WeaviateError::Rejected(
                    rejected,
                )));
            }
        }
        Ok(())
    }
}

impl VectorStoreIndex for WeaviateVectorStore {
    type Filter = WeaviateFilter;

    /// Returns the nearest objects with their `document` property decoded as
    /// `T`. Errors when an object has no `document` or it does not decode.
    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<WeaviateFilter>,
    ) -> Result<Vec<VectorSearchResult<T>>, VectorStoreError> {
        self.search(&req, true)
            .await?
            .into_iter()
            .map(|hit| {
                let Additional { id, distance } = hit.additional;
                let document = hit.document.ok_or_else(|| {
                    VectorStoreError::datastore(WeaviateError::MissingDocument(id.clone()))
                })?;
                Ok(VectorSearchResult {
                    score: 1.0 - distance,
                    id,
                    document: serde_json::from_str(&document)?,
                })
            })
            .collect()
    }

    /// Like `top_n` but returns only scores and object ids.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<WeaviateFilter>,
    ) -> Result<Vec<VectorSearchIdResult>, VectorStoreError> {
        Ok(self
            .search(&req, false)
            .await?
            .into_iter()
            .map(|hit| VectorSearchIdResult {
                score: 1.0 - hit.additional.distance,
                id: hit.additional.id,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests;
