//! YandexGPT OpenAI-compatible provider.
//!
//! This provider reuses the OpenAI-compatible request/response shapes with
//! a custom base URL and required `OpenAI-Project` header. The final model
//! identifier is assembled as `gpt://<folder-id>/<model-name>` where the
//! folder ID is provided via [`ClientBuilder::folder`].

use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::completion::{self, CompletionError, CompletionRequest as CoreCompletionRequest};
use crate::embeddings::{self, EmbeddingError};
use crate::http_client;
use crate::http_client::HttpClientExt;
use crate::providers::openai;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use http::header::{HeaderName, HeaderValue};

const YANDEX_API_BASE_URL: &str = "https://llm.api.cloud.yandex.net/v1";
#[allow(dead_code)]
const YANDEX_RESPONSES_API_BASE_URL: &str = "https://rest-assistant.api.cloud.yandex.net/v1";

/// `yandexgpt-lite/latest` text model.
pub const YANDEXGPT_LITE_LATEST: &str = "yandexgpt-lite/latest";
/// `yandexgpt/latest` text model.
pub const YANDEXGPT_LATEST: &str = "yandexgpt/latest";
/// `yandexgpt/rc` (YandexGPT 5.1) text model.
pub const YANDEXGPT_RC: &str = "yandexgpt/rc";

/// `text-search-doc/latest` embedding model.
pub const YANDEX_EMBED_TEXT_SEARCH_DOC: &str = "text-search-doc/latest";
/// `text-search-query/latest` embedding model.
pub const YANDEX_EMBED_TEXT_SEARCH_QUERY: &str = "text-search-query/latest";
/// `text-embeddings/latest` embedding model.
pub const YANDEX_EMBED_TEXT_EMBEDDINGS: &str = "text-embeddings/latest";

#[derive(Debug, Clone, Default)]
pub struct YandexExt {
    folder: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct YandexExtBuilder {
    folder: Option<String>,
}

type YandexApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<YandexExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<YandexExtBuilder, YandexApiKey, H>;

impl YandexExt {
    fn qualify_completion_model(&self, model: impl Into<String>) -> String {
        let model = model.into();

        if model.starts_with("gpt://") {
            return model;
        }

        match &self.folder {
            Some(folder) => format!("gpt://{folder}/{model}"),
            None => model,
        }
    }

    fn qualify_embedding_model(&self, model: impl Into<String>) -> String {
        let model = model.into();

        if model.starts_with("emb://") {
            return model;
        }

        match &self.folder {
            Some(folder) => format!("emb://{folder}/{model}"),
            None => model,
        }
    }
}

impl From<YandexExtBuilder> for YandexExt {
    fn from(value: YandexExtBuilder) -> Self {
        Self {
            folder: value.folder,
        }
    }
}

impl DebugExt for YandexExt {
    fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn std::fmt::Debug)> {
        [("folder", (&self.folder as &dyn std::fmt::Debug))].into_iter()
    }
}

impl Provider for YandexExt {
    type Builder = YandexExtBuilder;

    const VERIFY_PATH: &'static str = "/models";

    fn build<H>(
        builder: &client::ClientBuilder<Self::Builder, YandexApiKey, H>,
    ) -> http_client::Result<Self> {
        Ok(builder.ext().clone().into())
    }
}

impl<H> Capabilities<H> for YandexExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Capable<EmbeddingModel<H>>;
    type Transcription = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl ProviderBuilder for YandexExtBuilder {
    type Output = YandexExt;
    type ApiKey = YandexApiKey;

    const BASE_URL: &'static str = YANDEX_API_BASE_URL;

    fn finish<H>(
        &self,
        mut builder: client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, Self::ApiKey, H>> {
        if let Some(folder) = &self.folder {
            builder.headers_mut().insert(
                HeaderName::from_static("openai-project"),
                HeaderValue::from_str(folder)?,
            );
        }

        *builder.ext_mut() = self.clone();

        Ok(builder)
    }
}

impl<H> ClientBuilder<H> {
    /// Set the folder ID used for the `OpenAI-Project` header and model path.
    pub fn folder(self, folder: impl Into<String>) -> Self {
        self.over_ext(|mut ext| {
            ext.folder = Some(folder.into());
            ext
        })
    }
}

impl<H> Client<H> {
    fn qualify_completion_model(&self, model: impl Into<String>) -> String {
        self.ext().qualify_completion_model(model)
    }

    fn qualify_embedding_model(&self, model: impl Into<String>) -> String {
        self.ext().qualify_embedding_model(model)
    }
}

impl ProviderClient for Client {
    type Input = YandexApiKey;

    /// Create a new YandexGPT client using `YANDEX_API_KEY` and optional `YANDEX_FOLDER_ID`.
    fn from_env() -> Self {
        let api_key = std::env::var("YANDEX_API_KEY").expect("YANDEX_API_KEY not set");
        let folder = std::env::var("YANDEX_FOLDER_ID").ok();
        let base_url = std::env::var("YANDEX_BASE_URL").ok();

        let mut builder = Client::builder().api_key(api_key);

        if let Some(folder) = folder {
            builder = builder.folder(folder);
        }

        if let Some(base_url) = base_url {
            builder = builder.base_url(base_url);
        }

        builder.build().unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(input).unwrap()
    }
}

fn to_openai_responses_client<T: Clone>(client: &Client<T>) -> openai::Client<T> {
    client::Client::from_parts(
        client.base_url().to_string(),
        client.headers().clone(),
        client.http_client().clone(),
        openai::client::OpenAIResponsesExt,
    )
}

fn to_openai_completions_client<T: Clone>(client: &Client<T>) -> openai::CompletionsClient<T> {
    client::Client::from_parts(
        client.base_url().to_string(),
        client.headers().clone(),
        client.http_client().clone(),
        openai::client::OpenAICompletionsExt,
    )
}

// ------------------------------------------------------------------
// Completion wrapper
// ------------------------------------------------------------------

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    inner: openai::CompletionModel<T>,
}

impl<T> CompletionModel<T> {
    fn new(client: &Client<T>, model: impl Into<String>) -> Self
    where
        T: Clone + Default + std::fmt::Debug + 'static,
    {
        let inner = openai::CompletionModel::new(
            to_openai_completions_client(client),
            client.qualify_completion_model(model),
        );

        Self { inner }
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt
        + Default
        + std::fmt::Debug
        + Clone
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    type Response = openai::CompletionResponse;
    type StreamingResponse = openai::streaming::StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client, model)
    }

    async fn completion(
        &self,
        completion_request: CoreCompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        self.inner.completion(completion_request).await
    }

    async fn stream(
        &self,
        request: CoreCompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        self.inner.stream(request).await
    }
}

// ------------------------------------------------------------------
// Embedding wrapper
// ------------------------------------------------------------------

#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    inner: openai::EmbeddingModel<T>,
    ndims: usize,
}

impl<T> EmbeddingModel<T> {
    fn new(client: &Client<T>, model: impl Into<String>, ndims: usize) -> Self
    where
        T: Clone + Default + std::fmt::Debug,
    {
        let inner = openai::EmbeddingModel::new(
            to_openai_responses_client(client),
            client.qualify_embedding_model(model),
            ndims,
        );

        Self { inner, ndims }
    }
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + Send + 'static,
{
    const MAX_DOCUMENTS: usize = 1024;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>, ndims: Option<usize>) -> Self {
        Self::new(client, model, ndims.unwrap_or_default())
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + crate::wasm_compat::WasmCompatSend,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        // Yandex embeddings endpoint only accepts one string per request.
        // Run per-item calls and reassemble.
        let docs: Vec<String> = documents.into_iter().collect();
        let mut results = Vec::with_capacity(docs.len());

        for doc in docs {
            let mut single = self.inner.embed_texts(vec![doc.clone()]).await?;
            let Some(embed) = single.pop() else {
                return Err(EmbeddingError::ResponseError(
                    "Empty embedding response".to_string(),
                ));
            };

            results.push(embeddings::Embedding {
                document: doc,
                vec: embed.vec,
            });
        }

        Ok(results)
    }
}
