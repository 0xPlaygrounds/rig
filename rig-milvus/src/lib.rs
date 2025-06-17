use reqwest::StatusCode;
use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{VectorStoreError, VectorStoreIndex},
};
use serde::{Deserialize, Serialize};

/// Represents a vector store implementation using Milvus - <https://milvus.io/> as the backend.
pub struct MilvusVectorStore<M> {
    /// Model used to generate embeddings for the vector store
    model: M,
    base_url: String,
    client: reqwest::Client,
    database_name: String,
    collection_name: String,
    token: Option<String>,
}

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
    anns_field: &'a str,
    limit: usize,
    output_fields: Vec<&'a str>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResult<T> {
    code: i64,
    data: Vec<SearchResultData<T>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultData<T> {
    id: i64,
    distance: f64,
    document: T,
    embedded_text: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultOnlyId {
    code: i64,
    data: Vec<SearchResultDataOnlyId>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultDataOnlyId {
    id: i64,
    distance: f64,
}

impl<M: EmbeddingModel> MilvusVectorStore<M> {
    /// Creates a new instance of `MilvusVectorStore`.
    ///
    /// # Arguments
    /// * `model` - Embedding model instance
    /// * `base_url` - The URL of where your Milvus instance is located. Alternatively if you're using the Milvus offering provided by Zilliz, your cluster endpoint.
    /// * `database_name` - The name of your database
    /// * `collection_name` - The name of your collection
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

    /// Forms the auth token for Milvus from your username and password. Required if using a Milvus instance that requires authentication.
    pub fn auth(mut self, username: String, password: String) -> Self {
        let str = format!("{username}:{password}");
        self.token = Some(str);

        self
    }

    /// Creates a Milvus insertion request.
    fn create_insert_request(&self, data: Vec<CreateRecord>) -> InsertRequest {
        InsertRequest {
            data,
            collection_name: &self.collection_name,
            db_name: &self.database_name,
        }
    }

    /// Creates a Milvus semantic search request.
    fn create_search_request(&self, data: Vec<f64>, limit: usize) -> SearchRequest {
        SearchRequest {
            collection_name: &self.collection_name,
            db_name: &self.database_name,
            data,
            anns_field: "embedding",
            limit,
            output_fields: vec!["id", "distance", "document", "embeddedText"],
        }
    }

    /// Creates a semantic search request, but only for IDs.
    fn create_search_request_id_only(&self, data: Vec<f64>, limit: usize) -> SearchRequest {
        SearchRequest {
            collection_name: &self.collection_name,
            db_name: &self.database_name,
            data,
            anns_field: "embedding",
            limit,
            output_fields: vec!["id", "distance"],
        }
    }

    /// Insert vectors (with metadata) into your Milvus instance.
    pub async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let url = format!(
            "{base_url}/v2/vectordb/entities/insert",
            base_url = self.base_url
        );

        let data = documents
            .into_iter()
            .map(|(document, embeddings)| {
                let json_document: serde_json::Value = serde_json::to_value(&document)?;
                let json_document_as_string = serde_json::to_string(&json_document)?;

                let embeddings = embeddings
                    .into_iter()
                    .map(|embedding| {
                        let embedded_text = embedding.document;
                        let embedding: Vec<f64> = embedding.vec;

                        CreateRecord {
                            document: json_document_as_string.clone(),
                            embedded_text,
                            embedding,
                        }
                    })
                    .collect::<Vec<CreateRecord>>();
                Ok(embeddings)
            })
            .collect::<Result<Vec<Vec<CreateRecord>>, VectorStoreError>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<CreateRecord>>();

        let mut client = self.client.post(url);
        if let Some(ref token) = self.token {
            client = client.header("Authentication", format!("Bearer {token}"));
        }

        let insert_request = self.create_insert_request(data);

        let body = serde_json::to_string(&insert_request).unwrap();

        let res = client.body(body).send().await?;

        if res.status() != StatusCode::OK {
            let status = res.status();
            let text = res.text().await?;

            return Err(VectorStoreError::ExternalAPIError(status, text));
        }

        Ok(())
    }
}

impl<M: EmbeddingModel> VectorStoreIndex for MilvusVectorStore<M> {
    /// Search for the top `n` nearest neighbors to the given query within the Milvus vector store.
    /// Returns a vector of tuples containing the score, ID, and payload of the nearest neighbors.
    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let embedding = self.model.embed_text(query).await?;
        let url = format!(
            "{base_url}/v2/vectordb/entities/search",
            base_url = self.base_url
        );

        let body = self.create_search_request(embedding.vec, n);

        let mut client = self.client.post(url);
        if let Some(ref token) = self.token {
            client = client.header("Authentication", format!("Bearer {token}"));
        }

        let body = serde_json::to_string(&body)?;

        let res = client.body(body).send().await?;

        if res.status() != StatusCode::OK {
            let status = res.status();
            let text = res.text().await?;

            return Err(VectorStoreError::ExternalAPIError(status, text));
        }

        let json: SearchResult<T> = res.json().await?;

        let res = json
            .data
            .into_iter()
            .map(|x| (x.distance, x.id.to_string(), x.document))
            .collect();

        Ok(res)
    }

    /// Search for the top `n` nearest neighbors to the given query within the Milvus vector store.
    /// Returns a vector of tuples containing the score and ID of the nearest neighbors.
    async fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let embedding = self.model.embed_text(query).await?;
        let url = format!(
            "{base_url}/v2/vectordb/entities/search",
            base_url = self.base_url
        );

        let body = self.create_search_request_id_only(embedding.vec, n);

        let mut client = self.client.post(url);
        if let Some(ref token) = self.token {
            client = client.header("Authentication", format!("Bearer {token}"));
        }

        let body = serde_json::to_string(&body)?;

        let res = client.body(body).send().await?;

        if res.status() != StatusCode::OK {
            let status = res.status();
            let text = res.text().await?;

            return Err(VectorStoreError::ExternalAPIError(status, text));
        }

        let json: SearchResultOnlyId = res.json().await?;

        let res = json
            .data
            .into_iter()
            .map(|x| (x.distance, x.id.to_string()))
            .collect();

        Ok(res)
    }
}
