//! Milvus vector store for Rig.
//!
//! [`MilvusVectorStore`] inserts and searches entities through the Milvus v2
//! HTTP API, narrowed by a [`Filter`] expression. The `rig` facade re-exports
//! this crate as `rig::milvus` under the `milvus` feature.

mod filter;

pub use filter::{Filter, MilvusValue};

use reqwest::StatusCode;
use rig_core::{
    Embed,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{SearchFilter, VectorSearchRequest},
    },
    wasm_compat::WasmCompatSend,
};
use rig_reqwest::from_reqwest;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// Vector store backed by a [Milvus](https://milvus.io/) collection.
///
/// Queries are embedded with the same model `M` that populated the collection,
/// so results are meaningless under another model.
pub struct MilvusVectorStore<M> {
    model: M,
    base_url: String,
    client: reqwest::Client,
    database_name: String,
    collection_name: String,
    token: Option<String>,
}

/// One row written by [`InsertDocuments`], holding the document serialized as a
/// JSON string alongside its embedding and source text.
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateRecord {
    document: String,
    embedded_text: String,
    embedding: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InsertRequest<'a> {
    data: Vec<CreateRecord>,
    collection_name: &'a str,
    db_name: &'a str,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchRequest<'a> {
    collection_name: &'a str,
    db_name: &'a str,
    data: Vec<f64>,
    #[serde(skip_serializing_if = "String::is_empty")]
    filter: String,
    anns_field: &'a str,
    limit: usize,
    output_fields: Vec<&'a str>,
}

/// Milvus search response envelope, generic over the row shape.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResult<Row> {
    code: i64,
    data: Vec<Row>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultData<T> {
    id: i64,
    distance: f64,
    document: T,
}

/// Row shape for the id-only search path.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultDataOnlyId {
    id: i64,
    distance: f64,
}

impl<M: EmbeddingModel> MilvusVectorStore<M> {
    /// Creates a store over a collection reached at `base_url`, which is the
    /// Milvus instance or Zilliz cluster endpoint. Requests are unauthenticated
    /// until [`MilvusVectorStore::auth`] supplies credentials.
    pub fn new(model: M, base_url: String, database_name: String, collection_name: String) -> Self {
        Self {
            model,
            base_url,
            client: reqwest::Client::new(),
            database_name,
            collection_name,
            token: None,
        }
    }

    /// Sets the credentials sent as a `Bearer username:password` authorization
    /// header on every request.
    pub fn auth(mut self, username: &str, password: &str) -> Self {
        let str = format!("{username}:{password}");
        self.token = Some(str);

        self
    }

    fn create_insert_request(&self, data: Vec<CreateRecord>) -> InsertRequest<'_> {
        InsertRequest {
            data,
            collection_name: &self.collection_name,
            db_name: &self.database_name,
        }
    }

    /// Builds the search body. Any request threshold becomes a
    /// `distance >= threshold` condition combined with the request filter.
    fn create_search_request(
        &self,
        data: Vec<f64>,
        req: &VectorSearchRequest<Filter>,
        id_only: bool,
    ) -> SearchRequest<'_> {
        const OUTPUT_FIELDS: [&str; 4] = ["id", "distance", "document", "embeddedText"];
        const OUTPUT_FIELDS_ID_ONLY: [&str; 2] = ["id", "distance"];

        let output_fields = if id_only {
            OUTPUT_FIELDS_ID_ONLY.to_vec()
        } else {
            OUTPUT_FIELDS.to_vec()
        };

        let threshold = req
            .threshold()
            .map(|thresh| Filter::gte("distance", thresh.into()));

        let filter = match (threshold, req.filter()) {
            (Some(thresh), Some(filter)) => thresh.and(filter.clone()).into_inner(),
            (Some(thresh), _) => thresh.into_inner(),
            (_, Some(filter)) => filter.clone().into_inner(),
            _ => String::new(),
        };

        SearchRequest {
            collection_name: &self.collection_name,
            db_name: &self.database_name,
            data,
            filter,
            anns_field: "embedding",
            limit: req.samples() as usize,
            output_fields,
        }
    }

    /// Embeds the query and posts it to the search endpoint, returning the
    /// decoded body. Any non-200 status is returned as an error with its body.
    async fn search<T: for<'a> Deserialize<'a>>(
        &self,
        req: &VectorSearchRequest<Filter>,
        id_only: bool,
    ) -> Result<T, VectorStoreError> {
        let embedding = self.model.embed_text(req.query()).await?;
        let url = format!(
            "{base_url}/v2/vectordb/entities/search",
            base_url = self.base_url
        );

        let body = self.create_search_request(embedding.vec, req, id_only);

        let mut client = self.client.post(url);
        if let Some(ref token) = self.token {
            client = client.header("Authorization", format!("Bearer {token}"));
        }

        let body = serde_json::to_string(&body)?;

        let res = client.body(body).send().await.map_err(from_reqwest)?;

        if res.status() != StatusCode::OK {
            let status = res.status();
            let text = res.text().await.map_err(from_reqwest)?;

            return Err(VectorStoreError::ExternalAPIError(status, text));
        }

        Ok(res.json().await.map_err(from_reqwest)?)
    }
}

impl<M: EmbeddingModel> InsertDocuments for MilvusVectorStore<M> {
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let url = format!(
            "{base_url}/v2/vectordb/entities/insert",
            base_url = self.base_url
        );

        let data =
            rig_core::vector_store::flatten_embedded(documents, |json_document, embedding| {
                Ok(CreateRecord {
                    document: serde_json::to_string(json_document)?,
                    embedded_text: embedding.document,
                    embedding: embedding.vec,
                })
            })?;

        let mut client = self.client.post(url);
        if let Some(ref token) = self.token {
            client = client.header("Authorization", format!("Bearer {token}"));
        }

        let insert_request = self.create_insert_request(data);

        let body = serde_json::to_string(&insert_request)?;

        let res = client.body(body).send().await.map_err(from_reqwest)?;

        if res.status() != StatusCode::OK {
            let status = res.status();
            let text = res.text().await.map_err(from_reqwest)?;

            return Err(VectorStoreError::ExternalAPIError(status, text));
        }

        Ok(())
    }
}

impl<M: EmbeddingModel> VectorStoreIndex for MilvusVectorStore<M> {
    type Filter = Filter;

    /// Returns matches as `(distance, id, document)` in the order Milvus reports.
    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let json: SearchResult<SearchResultData<T>> = self.search(&req, false).await?;

        let res = json
            .data
            .into_iter()
            .map(|x| (x.distance, x.id.to_string(), x.document))
            .collect();

        Ok(res)
    }

    /// Like `top_n` but returns `(distance, id)` without requesting documents.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let json: SearchResult<SearchResultDataOnlyId> = self.search(&req, true).await?;

        let res = json
            .data
            .into_iter()
            .map(|x| (x.distance, x.id.to_string()))
            .collect();

        Ok(res)
    }
}
