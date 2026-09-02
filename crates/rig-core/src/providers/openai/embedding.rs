use super::{client::ApiResponse, completion::Usage};
use crate::embeddings;
use crate::embeddings::EmbeddingError;
#[cfg(test)]
use crate::http_client;
use crate::http_client::HttpClientExt;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::{Deserialize, Serialize};

// ================================================================
// OpenAI Embedding API
// ================================================================
/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

/// The OpenAI-compatible embeddings wire response as every provider on this
/// wire answers it: what [`GenericEmbeddingModel::raw_embed_texts`] returns.
/// `usage` is optional here because compatible providers may omit it; the
/// strict [`EmbeddingResponse`] above is OpenAI's own contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibleEmbeddingResponse {
    #[serde(default)]
    pub object: String,
    pub data: Vec<EmbeddingData>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl embeddings::NormalizeEmbeddingResponse for CompatibleEmbeddingResponse {
    fn normalize(
        self,
        provider: &str,
        documents: Vec<String>,
    ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
        if self.data.len() != documents.len() {
            return Err(EmbeddingError::ResponseError(
                "Response data length does not match input length".into(),
            ));
        }

        let usage = match &self.usage {
            Some(usage) => crate::completion::Usage {
                input_tokens: usage.prompt_tokens as u64,
                output_tokens: 0,
                total_tokens: usage.total_tokens as u64,
                cached_input_tokens: usage
                    .prompt_tokens_details
                    .as_ref()
                    .map_or(0, |details| details.cached_tokens as u64),
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            },
            None => crate::completion::Usage::new(),
        };

        let embeddings: Vec<embeddings::Embedding> = self
            .data
            .into_iter()
            .zip(documents)
            .map(|(embedding, document)| embeddings::Embedding {
                document,
                vec: embedding
                    .embedding
                    .into_iter()
                    .filter_map(|n| n.as_f64())
                    .collect(),
            })
            .collect();

        Ok(embeddings::EmbeddingResponse::new(embeddings, provider)
            .with_model(self.model)
            .with_usage(usage))
    }
}

/// Provider-specific spelling for an embedding dimension request field.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingDimensions {
    /// Serialize the value as the OpenAI-compatible `dimensions` field.
    Dimensions(usize),
    /// Serialize the value as Mistral's `output_dimension` field.
    OutputDimension(usize),
}

/// Contract for provider extensions that speak an OpenAI-compatible embeddings
/// wire format through [`GenericEmbeddingModel`].
#[doc(hidden)]
pub trait OpenAIEmbeddingsCompatible: crate::client::Provider {
    /// Provider name used in embedding request and response errors.
    const PROVIDER_NAME: &'static str;

    /// Whether successful responses from this provider must include usage.
    const REQUIRES_USAGE: bool = true;

    /// The provider's transport request-id response header, when it has one
    /// (OpenAI: `x-request-id`). `None` means the provider reports none.
    const REQUEST_ID_HEADER: Option<&'static str> = None;

    /// Whether the provider accepts the OpenAI-compatible `encoding_format` field.
    const SUPPORTS_ENCODING_FORMAT: bool = true;

    /// Whether the provider accepts the OpenAI-compatible `user` field.
    const SUPPORTS_USER: bool = true;

    /// Whether the model is sent as a `model` field in the request body.
    /// Azure routes the deployment through the URL and sends no model field.
    const SENDS_MODEL_FIELD: bool = true;

    /// Most inputs the provider accepts in one embeddings request.
    ///
    /// [`EmbeddingsBuilder`](crate::embeddings::EmbeddingsBuilder) chunks by
    /// this, so a value above the provider's real cap turns a large job into a
    /// rejected request rather than more round trips. OpenAI's 1024 is the
    /// default; providers with a smaller cap override it.
    const MAX_DOCUMENTS: usize = 1024;

    /// Output dimensions for a model the provider knows by name, used when the
    /// caller did not state them. The default consults OpenAI's own table;
    /// providers with their own models override it, because a model missing
    /// from every table reports `ndims() == 0`.
    fn default_ndims(model: &str) -> Option<usize> {
        model_dimensions_from_identifier(model)
    }

    /// The request path for embeddings, resolved against the client base URL.
    fn embeddings_path(&self) -> String {
        "/embeddings".to_string()
    }

    /// The request path for embeddings for a given model. Providers that
    /// route the model through the URL (Azure deployments) override this;
    /// everyone else inherits [`OpenAIEmbeddingsCompatible::embeddings_path`].
    fn embeddings_path_for_model(&self, _model: &str) -> String {
        self.embeddings_path()
    }

    /// Validate and select the provider's dimension field.
    fn embedding_dimensions(
        &self,
        model: &str,
        dimensions: Option<usize>,
    ) -> Result<Option<EmbeddingDimensions>, EmbeddingError> {
        // OpenAI's legacy Ada model does not accept `dimensions`. Keep that
        // OpenAI-specific exception in the provider hook so another
        // OpenAI-compatible provider can validate an identically named model.
        Ok((model != TEXT_EMBEDDING_ADA_002)
            .then_some(dimensions.map(EmbeddingDimensions::Dimensions))
            .flatten())
    }
}

impl OpenAIEmbeddingsCompatible for super::OpenAIResponses {
    const PROVIDER_NAME: &'static str = "openai";
    const REQUEST_ID_HEADER: Option<&'static str> = Some("x-request-id");
}

impl OpenAIEmbeddingsCompatible for super::OpenAICompletions {
    const PROVIDER_NAME: &'static str = "openai";
    const REQUEST_ID_HEADER: Option<&'static str> = Some("x-request-id");
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Serialize)]
struct CompatibleEmbeddingRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimension: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<EncodingFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<serde_json::Number>,
    pub index: usize,
}

#[doc(hidden)]
#[derive(Clone)]
pub struct GenericEmbeddingModel<Ext, H = crate::http_client::BoxedHttpClient> {
    client: crate::client::Client<Ext, H>,
    pub model: String,
    pub encoding_format: Option<EncodingFormat>,
    pub user: Option<String>,
    ndims: usize,
    dimensions_were_explicitly_set: bool,
}

/// The embedding model struct for OpenAI's Embeddings API.
///
/// This preserves the historical public generic shape where the first generic
/// parameter is the HTTP client type.
pub type EmbeddingModel<H = crate::http_client::BoxedHttpClient> =
    GenericEmbeddingModel<super::OpenAIResponses, H>;

/// Default dimensions for OpenAI's known embedding models (also used by
/// Azure OpenAI, which deploys the same models).
pub(crate) fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        TEXT_EMBEDDING_3_LARGE => Some(3_072),
        TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_ADA_002 => Some(1_536),
        _ => None,
    }
}

impl<Ext, H> GenericEmbeddingModel<Ext, H>
where
    crate::client::Client<Ext, H>: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync,
    Ext: OpenAIEmbeddingsCompatible + Clone,
{
    /// Perform the request and return the provider's native wire response
    /// instead of the normalized [`embeddings::EmbeddingResponse`]. Same
    /// request, transport, parser, and error path as
    /// [`embeddings::EmbeddingModel::embed_texts_response`].
    pub async fn raw_embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<CompatibleEmbeddingResponse, EmbeddingError> {
        self.raw_embed_texts_with_request_id(documents)
            .await
            .map(|(response, _)| response)
    }

    /// [`Self::raw_embed_texts`] plus the transport request id from the
    /// provider's request-id response header, when it carries one.
    pub async fn raw_embed_texts_with_request_id(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<(CompatibleEmbeddingResponse, Option<String>), EmbeddingError> {
        let documents: Vec<String> = documents.into_iter().collect();
        self.raw_embed_texts_slice(&documents).await
    }

    /// Borrow-shaped twin of [`Self::raw_embed_texts_with_request_id`]: the
    /// batch is only serialized into the request body, so callers that keep
    /// their documents (the normalize path) can lend them instead of cloning
    /// the batch.
    async fn raw_embed_texts_slice(
        &self,
        documents: &[String],
    ) -> Result<(CompatibleEmbeddingResponse, Option<String>), EmbeddingError> {
        if self.encoding_format == Some(EncodingFormat::Base64) {
            return Err(EmbeddingError::UnsupportedResponseEncoding {
                provider: Ext::PROVIDER_NAME,
                encoding_format: "base64",
            });
        }

        if self.encoding_format.is_some() && !Ext::SUPPORTS_ENCODING_FORMAT {
            return Err(EmbeddingError::UnsupportedParameter {
                provider: Ext::PROVIDER_NAME,
                parameter: "encoding_format",
            });
        }

        if self.user.is_some() && !Ext::SUPPORTS_USER {
            return Err(EmbeddingError::UnsupportedParameter {
                provider: Ext::PROVIDER_NAME,
                parameter: "user",
            });
        }

        let requested_dimensions =
            (self.dimensions_were_explicitly_set || self.ndims > 0).then_some(self.ndims);
        let dimensions = self
            .client
            .provider()
            .embedding_dimensions(&self.model, requested_dimensions)?;
        let (dimensions, output_dimension) = match dimensions {
            Some(EmbeddingDimensions::Dimensions(value)) => (Some(value), None),
            Some(EmbeddingDimensions::OutputDimension(value)) => (None, Some(value)),
            None => (None, None),
        };

        let body = serde_json::to_vec(&CompatibleEmbeddingRequest {
            model: Ext::SENDS_MODEL_FIELD.then_some(self.model.as_str()),
            input: documents,
            dimensions,
            output_dimension,
            encoding_format: self.encoding_format,
            user: self.user.as_deref(),
        })?;

        let req = self
            .client
            .post(
                self.client
                    .provider()
                    .embeddings_path_for_model(&self.model),
            )?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        let (parts, body) = response.into_parts();
        let status = parts.status;
        let provider_request_id =
            crate::providers::internal::transcription::request_id_from_headers(
                &parts.headers,
                Ext::REQUEST_ID_HEADER,
            );
        let response_body: Vec<u8> = body.await?;
        if status.is_success() {
            let parsed: ApiResponse<CompatibleEmbeddingResponse> =
                serde_json::from_slice(&response_body)?;

            match parsed {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "embedding token usage: {:?}",
                        response.usage
                    );
                    Ok((response, provider_request_id))
                }
                ApiResponse::Err(err) => {
                    tracing::warn!(message = %err.message, "provider returned an error response");
                    Err(EmbeddingError::from_http_response(
                        status,
                        String::from_utf8_lossy(&response_body).into_owned(),
                    ))
                }
            }
        } else {
            Err(EmbeddingError::from_http_response(
                status,
                String::from_utf8_lossy(&response_body).into_owned(),
            ))
        }
    }
}

impl<Ext, H> embeddings::EmbeddingModel for GenericEmbeddingModel<Ext, H>
where
    crate::client::Client<Ext, H>: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync,
    Ext: OpenAIEmbeddingsCompatible + Clone,
{
    fn max_documents(&self) -> usize {
        Ext::MAX_DOCUMENTS
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts_response(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
        crate::telemetry::instrument_modality(
            Ext::PROVIDER_NAME,
            &self.model,
            crate::telemetry::ModalityOperation::Embeddings,
            async {
                use embeddings::NormalizeEmbeddingResponse as _;

                let documents: Vec<String> = documents.into_iter().collect();
                let (response, provider_request_id) =
                    self.raw_embed_texts_slice(&documents).await?;

                if response.usage.is_none() && Ext::REQUIRES_USAGE {
                    return Err(EmbeddingError::MissingUsage {
                        provider: Ext::PROVIDER_NAME,
                    });
                }

                let captured = serde_json::to_value(&response)?;
                let normalized = response.normalize(Ext::PROVIDER_NAME, documents)?;
                let embeddings = &normalized.embeddings;

                // A width the caller *declared* must be the width they
                // got. Two carve-outs, and both matter:
                //
                // * the width must have been set explicitly — a handle
                //   built without one reports whatever the provider table
                //   says and has nothing to disagree with;
                // * and it must be non-zero. Zero is rig's sentinel for
                //   *unknown*, not a declaration: `default_ndims`
                //   returning `None` lands here through
                //   `unwrap_or_default()`, and
                //   `GenericEmbeddingModel::new(client, model, 0)` is the
                //   documented way to say "I do not know how wide this
                //   is". Treating it as a claim would turn every such
                //   handle into a hard error on its first request.
                //
                // The failure this catches is silent, because the
                // providers where it happens are the ones that *ignore*
                // `dimensions` rather than rejecting it — `llama-server`'s
                // embeddings handler reads no such field at all, so a
                // request for 128 answers 200 with 1024-wide vectors while
                // `ndims()` keeps reporting 128. A vector store sized from
                // `ndims()` then builds an index that cannot hold its own
                // vectors, and the first thing to notice is the store.
                if self.dimensions_were_explicitly_set
                    && self.ndims > 0
                    && let Some(returned) = embeddings
                        .iter()
                        .map(|embedding| embedding.vec.len())
                        .find(|width| *width != self.ndims)
                {
                    return Err(EmbeddingError::MismatchedDimensions {
                        provider: Ext::PROVIDER_NAME,
                        requested: self.ndims,
                        returned,
                    });
                }

                Ok(normalized
                    .with_optional_provider_request_id(provider_request_id)
                    .with_raw(captured))
            },
        )
        .await
    }
}

impl<Ext, H> GenericEmbeddingModel<Ext, H>
where
    crate::client::Client<Ext, H>: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync,
    Ext: OpenAIEmbeddingsCompatible + Clone,
{
    /// Build the model, defaulting `ndims` from the model identifier when the
    /// caller gave none — the body behind `EmbeddingsClient::embedding_model`.
    pub fn make(
        client: &crate::client::Client<Ext, H>,
        model: String,
        ndims: Option<usize>,
    ) -> Self {
        let dimensions_were_explicitly_set = ndims.is_some();
        let dims = ndims
            .or_else(|| Ext::default_ndims(&model))
            .unwrap_or_default();

        Self::from_parts(client.clone(), model, dims, dimensions_were_explicitly_set)
    }
}

impl<Ext, H> GenericEmbeddingModel<Ext, H>
where
    Ext: crate::client::Provider,
{
    pub fn new(
        client: crate::client::Client<Ext, H>,
        model: impl Into<String>,
        ndims: usize,
    ) -> Self {
        Self::from_parts(client, model, ndims, true)
    }

    fn from_parts(
        client: crate::client::Client<Ext, H>,
        model: impl Into<String>,
        ndims: usize,
        dimensions_were_explicitly_set: bool,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            encoding_format: None,
            ndims,
            dimensions_were_explicitly_set,
            user: None,
        }
    }

    pub fn with_model(client: crate::client::Client<Ext, H>, model: &str, ndims: usize) -> Self {
        Self::new(client, model, ndims)
    }

    pub fn with_encoding_format(
        client: crate::client::Client<Ext, H>,
        model: &str,
        ndims: usize,
        encoding_format: EncodingFormat,
    ) -> Self {
        Self::new(client, model, ndims).encoding_format(encoding_format)
    }

    pub fn encoding_format(mut self, encoding_format: EncodingFormat) -> Self {
        self.encoding_format = Some(encoding_format);
        self
    }

    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

#[cfg(test)]
mod tests;
