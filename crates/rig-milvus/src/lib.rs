//! Milvus vector store for Rig.
//!
//! [`MilvusVectorStore`] inserts and searches entities through the Milvus v2
//! HTTP API, narrowed by a [`Filter`] expression. The `rig` facade re-exports
//! this crate as `rig::milvus` under the `milvus` feature.

mod filter;
#[cfg(test)]
mod tests;

pub use filter::{Filter, MilvusValue};

use reqwest::StatusCode;
use rig_core::{
    Embed,
    embeddings::Embedding,
    vector_store::{
        InsertDocuments, VectorSearchIdResult, VectorSearchResult, VectorStoreError,
        VectorStoreIndex,
        request::{FilterError, VectorSearchRequest},
    },
    wasm_compat::WasmCompatSend,
};
use rig_reqwest::from_reqwest;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

/// Vector store backed by a [Milvus](https://milvus.io/) collection.
///
/// Queries are embedded with the same model that populated the collection,
/// so results are meaningless under another model.
pub struct MilvusVectorStore {
    model: rig_core::DynModel<rig_core::operation::Embedding>,
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
    data: [Vec<f64>; 1],
    #[serde(skip_serializing_if = "String::is_empty")]
    filter: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_params: Option<Value>,
    anns_field: &'a str,
    limit: usize,
    output_fields: Vec<&'a str>,
}

/// The application status Milvus reports on every response, including HTTP 200.
#[derive(Deserialize)]
struct ResponseCode {
    code: i64,
}

/// The payload of a response whose application code is zero.
#[derive(Deserialize)]
struct ResponseData<Data> {
    data: Data,
}

/// Row shape for document searches.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultData {
    id: i64,
    distance: f64,
    document: Value,
}

/// Row shape for the id-only search path.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultDataOnlyId {
    id: i64,
    distance: f64,
}

/// The largest `f32` below `threshold` rounded to `f32`, refusing a threshold
/// outside the `f32` range or whose radius is not finite.
///
/// Scores are compared at `f32` precision, the precision Milvus scores in. Milvus
/// excludes the radius, so a `COSINE` or `IP` score equal to the rounded threshold
/// stays. On `L2` the radius is a maximum, so a distance within one `f32` step
/// under the threshold can be left out.
fn radius(threshold: f64) -> Result<f32, FilterError> {
    let radius = (threshold as f32).next_down();
    if threshold.abs() > f64::from(f32::MAX) || !radius.is_finite() {
        return Err(FilterError::Must(
            format!("{threshold:e}"),
            "have a radius that fits in f32".into(),
        ));
    }
    Ok(radius)
}

/// Decodes a returned document as `T`.
///
/// A string is the JSON text insert writes, so it is parsed as JSON and plain
/// text is an error. Any other value, such as an object in a dynamic field, is
/// decoded as it stands.
fn decode_document<T: DeserializeOwned>(document: Value) -> Result<T, VectorStoreError> {
    Ok(match document {
        Value::String(text) => serde_json::from_str(&text)?,
        value => serde_json::from_value(value)?,
    })
}

impl MilvusVectorStore {
    /// Creates a store over a collection reached at `base_url`, which is the
    /// Milvus instance or Zilliz cluster endpoint. Requests are unauthenticated
    /// until [`MilvusVectorStore::auth`] supplies credentials.
    pub fn new(
        model: impl Into<rig_core::DynModel<rig_core::operation::Embedding>>,
        base_url: String,
        database_name: String,
        collection_name: String,
    ) -> Self {
        Self {
            model: model.into(),
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

    /// Builds the search body. A `radius` from the request threshold makes it a
    /// Milvus range search, so it applies to the score in the collection's own
    /// metric and the limit still fills from every match.
    fn create_search_request(
        &self,
        data: Vec<f64>,
        req: &VectorSearchRequest<Filter>,
        radius: Option<f32>,
        id_only: bool,
    ) -> SearchRequest<'_> {
        // Milvus always returns the id and distance, and refuses them as output
        // fields unless the collection stores a field of that name.
        let output_fields = if id_only { vec![] } else { vec!["document"] };

        let search_params = radius.map(|radius| json!({ "params": { "radius": radius } }));

        let filter = req.filter().cloned().map(Filter::into_inner);

        SearchRequest {
            collection_name: &self.collection_name,
            db_name: &self.database_name,
            data: [data],
            filter: filter.unwrap_or_default(),
            search_params,
            anns_field: "embedding",
            limit: req.samples() as usize,
            output_fields,
        }
    }

    /// Posts `body` to an entities endpoint and returns the response body.
    ///
    /// Milvus reports most failures as HTTP 200 with a nonzero `code`. Those and
    /// any non-200 status return [`VectorStoreError::ExternalAPIError`] with the
    /// response body.
    async fn post(
        &self,
        endpoint: &str,
        body: &impl Serialize,
    ) -> Result<String, VectorStoreError> {
        let url = format!(
            "{base_url}/v2/vectordb/entities/{endpoint}",
            base_url = self.base_url
        );

        let mut client = self.client.post(url);
        if let Some(ref token) = self.token {
            client = client.header("Authorization", format!("Bearer {token}"));
        }

        let body = serde_json::to_string(body)?;

        let res = client.body(body).send().await.map_err(from_reqwest)?;
        let status = res.status();
        let text = res.text().await.map_err(from_reqwest)?;

        if status != StatusCode::OK || serde_json::from_str::<ResponseCode>(&text)?.code != 0 {
            return Err(VectorStoreError::ExternalAPIError(status, text));
        }

        Ok(text)
    }

    /// Embeds the query and returns the matching rows. A threshold `f32` cannot
    /// hold is refused before the query is embedded.
    async fn search<Row: DeserializeOwned>(
        &self,
        req: &VectorSearchRequest<Filter>,
        id_only: bool,
    ) -> Result<Vec<Row>, VectorStoreError> {
        let radius = req.threshold().map(radius).transpose()?;
        let embedding = self.model.embed_text(req.query()).await?;
        let body = self.create_search_request(embedding.vec, req, radius, id_only);
        let text = self.post("search", &body).await?;

        Ok(serde_json::from_str::<ResponseData<Vec<Row>>>(&text)?.data)
    }
}

impl InsertDocuments for MilvusVectorStore {
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let data =
            rig_core::vector_store::flatten_embedded(documents, |json_document, embedding| {
                Ok(CreateRecord {
                    document: serde_json::to_string(json_document)?,
                    embedded_text: embedding.document,
                    embedding: embedding.vec,
                })
            })?;

        if data.is_empty() {
            return Ok(());
        }

        self.post("insert", &self.create_insert_request(data))
            .await?;

        Ok(())
    }
}

impl VectorStoreIndex for MilvusVectorStore {
    type Filter = Filter;

    /// Returns matches scored by distance, in the order Milvus reports.
    /// A threshold is a minimum for `COSINE` and `IP`, an exclusive maximum for `L2`.
    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<Filter>,
    ) -> Result<Vec<VectorSearchResult<T>>, VectorStoreError> {
        let rows: Vec<SearchResultData> = self.search(&req, false).await?;

        rows.into_iter()
            .map(|x| {
                Ok(VectorSearchResult {
                    score: x.distance,
                    id: x.id.to_string(),
                    document: decode_document(x.document)?,
                })
            })
            .collect()
    }

    /// Like `top_n` but returns only scores and ids, without requesting documents.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Filter>,
    ) -> Result<Vec<VectorSearchIdResult>, VectorStoreError> {
        let rows: Vec<SearchResultDataOnlyId> = self.search(&req, true).await?;

        let res = rows
            .into_iter()
            .map(|x| VectorSearchIdResult {
                score: x.distance,
                id: x.id.to_string(),
            })
            .collect();

        Ok(res)
    }
}
