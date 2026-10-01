//! HelixDB vector store for Rig.
//!
//! [`HelixDBVectorStore`] runs the `VectorSearch` and `InsertVector` HelixDB
//! queries through the [`HelixDB`] HTTP client. The `rig` facade re-exports
//! this crate as `rig::helixdb` under the `helixdb` feature.

use reqwest::{Client, StatusCode};
use rig_core::{
    vector_store::{InsertDocuments, VectorStoreError, VectorStoreIndex, request::Filter},
    wasm_compat::WasmCompatSend,
};
use serde::{Deserialize, Serialize};

/// HTTP client posting to HelixDB query endpoints, authenticating with an
/// `x-api-key` header when an API key is configured.
#[derive(Debug, Clone)]
pub struct HelixDB {
    port: Option<u16>,
    client: Client,
    endpoint: String,
    api_key: Option<String>,
}

impl HelixDB {
    /// Creates a client, defaulting the endpoint to `http://localhost`.
    pub fn new(endpoint: Option<&str>, port: Option<u16>, api_key: Option<&str>) -> Self {
        Self::with_client(endpoint, port, api_key, Client::new())
    }

    /// Creates a HelixDB client using a caller-provided reqwest client.
    pub fn with_client(
        endpoint: Option<&str>,
        port: Option<u16>,
        api_key: Option<&str>,
        client: Client,
    ) -> Self {
        Self {
            port,
            client,
            endpoint: endpoint.unwrap_or("http://localhost").to_string(),
            api_key: api_key.map(ToString::to_string),
        }
    }
}

/// Errors returned by the HelixDB HTTP client.
#[derive(Debug, thiserror::Error)]
pub enum HelixError {
    /// A request to HelixDB failed before a response body could be decoded.
    #[error("error communicating with server: {0}")]
    ReqwestError(#[from] reqwest::Error),

    /// HelixDB returned a non-200 response.
    #[error("got error from server: {details}")]
    RemoteError {
        /// Response body or status reason returned by HelixDB.
        details: String,
    },
}

impl HelixDB {
    /// Posts `data` to the named HelixDB query and decodes its response.
    pub async fn query<T, R>(&self, endpoint: &str, data: &T) -> Result<R, HelixError>
    where
        T: Serialize,
        R: for<'de> Deserialize<'de>,
    {
        let port = self.port.map(|port| format!(":{port}")).unwrap_or_default();
        let url = format!("{}{}/{}", self.endpoint, port, endpoint);

        let mut request = self.client.post(&url).json(data);
        if let Some(api_key) = &self.api_key {
            request = request.header("x-api-key", api_key);
        }

        let response = request.send().await?;

        match response.status() {
            StatusCode::OK => response.json().await.map_err(Into::into),
            code => match response.text().await {
                Ok(details) => Err(HelixError::RemoteError { details }),
                Err(_) => Err(HelixError::RemoteError {
                    details: code.canonical_reason().map_or_else(
                        || format!("unknown error with code: {code}"),
                        ToString::to_string,
                    ),
                }),
            },
        }
    }
}

/// Vector store backed by HelixDB queries.
///
/// Queries are embedded with the same model that populated the store, so
/// results are meaningless under another model.
///
/// ```no_run
/// use rig_core::providers::openai::OpenAI;
/// use rig_helixdb::{HelixDB, HelixDBVectorStore};
///
/// # fn example() -> anyhow::Result<()> {
/// let openai_model = OpenAI::from_env()?.embedding("text-embedding-ada-002", None);
///
/// let helixdb_client = HelixDB::new(None, Some(6969), None);
/// let vector_store = HelixDBVectorStore::new(helixdb_client, openai_model);
/// # let _ = vector_store;
/// # Ok(())
/// # }
/// ```
pub struct HelixDBVectorStore {
    client: HelixDB,
    model: rig_core::DynModel<rig_core::operation::Embedding>,
}

pub type HelixDBFilter = Filter<serde_json::Value>;

/// One `VectorSearch` hit, whose `score` is a cosine distance.
#[derive(Deserialize, Serialize, Clone, Debug)]
struct QueryResult {
    id: String,
    score: f64,
    doc: String,
    json_payload: String,
}

/// `VectorSearch` request body.
#[derive(Deserialize, Serialize, Clone, Debug)]
struct QueryInput {
    vector: Vec<f64>,
    limit: u64,
    threshold: f64,
}

/// `VectorSearch` response body.
#[derive(Serialize, Deserialize, Debug)]
struct VecResult {
    vec_docs: Vec<QueryResult>,
}

impl HelixDBVectorStore {
    /// Creates a new HelixDB vector store.
    pub fn new(
        client: HelixDB,
        model: impl Into<rig_core::DynModel<rig_core::operation::Embedding>>,
    ) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    /// Returns the underlying HelixDB client.
    pub fn client(&self) -> &HelixDB {
        &self.client
    }

    /// Embeds the query and runs `VectorSearch`. An absent request threshold is
    /// sent as zero.
    async fn vector_search(
        &self,
        req: &rig_core::vector_store::VectorSearchRequest<HelixDBFilter>,
    ) -> Result<Vec<QueryResult>, VectorStoreError> {
        let vector = self.model.embed_text(req.query()).await?.vec;

        let query_input = QueryInput {
            vector,
            limit: req.samples(),
            threshold: req.threshold().unwrap_or_default(),
        };

        let result: VecResult = self
            .client
            .query::<QueryInput, VecResult>("VectorSearch", &query_input)
            .await
            .map_err(VectorStoreError::datastore)?;

        Ok(result.vec_docs)
    }
}

impl InsertDocuments for HelixDBVectorStore {
    async fn insert_documents<Doc: Serialize + rig_core::Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, Vec<rig_core::embeddings::Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        #[derive(Serialize, Deserialize, Clone, Debug, Default)]
        struct QueryInput {
            vector: Vec<f64>,
            doc: String,
            json_payload: String,
        }

        #[derive(Serialize, Deserialize, Clone, Debug, Default)]
        struct QueryOutput {
            doc: String,
        }

        let queries =
            rig_core::vector_store::flatten_embedded(documents, |json_document, embedding| {
                Ok(QueryInput {
                    vector: embedding.vec,
                    doc: embedding.document,
                    json_payload: serde_json::to_string(json_document)?,
                })
            })?;

        for query in queries {
            self.client
                .query::<QueryInput, QueryOutput>("InsertVector", &query)
                .await
                .map_err(VectorStoreError::datastore)?;
        }
        Ok(())
    }
}

impl VectorStoreIndex for HelixDBVectorStore {
    type Filter = HelixDBFilter;

    // HelixDB reports cosine distance; `-(score - 1)` converts it to similarity.

    /// Returns matches as `(cosine similarity, id, document)`, discarding hits
    /// below the threshold or rejected by the request filter, which is evaluated
    /// client-side against each stored JSON payload.
    async fn top_n<T: for<'a> serde::Deserialize<'a> + WasmCompatSend>(
        &self,
        req: rig_core::vector_store::VectorSearchRequest<HelixDBFilter>,
    ) -> Result<Vec<(f64, String, T)>, rig_core::vector_store::VectorStoreError> {
        let docs = self
            .vector_search(&req)
            .await?
            .into_iter()
            .filter(|x| {
                let is_threshold = req.threshold().is_none_or(|t| -(x.score - 1.) >= t);

                is_threshold
                    && req
                        .filter()
                        .zip(serde_json::from_str(&x.json_payload).ok())
                        .is_none_or(
                            |(filter, payload): (&Filter<serde_json::Value>, serde_json::Value)| {
                                filter.satisfies(&payload)
                            },
                        )
            })
            .map(|x| {
                let doc: T = serde_json::from_str(&x.json_payload)?;

                Ok((-(x.score - 1.), x.id, doc))
            })
            .collect::<Result<Vec<_>, VectorStoreError>>()?;

        Ok(docs)
    }

    /// Like `top_n` but returns `(cosine similarity, id)` and ignores the
    /// request filter.
    async fn top_n_ids(
        &self,
        req: rig_core::vector_store::VectorSearchRequest<HelixDBFilter>,
    ) -> Result<Vec<(f64, String)>, rig_core::vector_store::VectorStoreError> {
        let docs = self
            .vector_search(&req)
            .await?
            .into_iter()
            .filter(|x| -(x.score - 1.) >= req.threshold().unwrap_or_default())
            .map(|x| Ok((-(x.score - 1.), x.id)))
            .collect::<Result<Vec<_>, VectorStoreError>>()?;

        Ok(docs)
    }
}
