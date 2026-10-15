//! Claude on Vertex AI: Anthropic's Messages wire, sent to the `anthropic`
//! publisher's `rawPredict` endpoint (`streamRawPredict` when streamed) with
//! Google credentials. Requests and replies are the Messages API's, so the
//! Anthropic typed options, extras and history rules apply unchanged; the
//! model is named in the URL and `anthropic_version` travels in the body.
//!
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::providers::anthropic::CLAUDE_SONNET_4_6;
//! use rig_vertexai::VertexAi;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let claude = VertexAi::from_env()?.anthropic(CLAUDE_SONNET_4_6);
//! let reply = claude.call(CompletionRequest::new("Hello, Claude")).await?;
//! # let _ = reply;
//! # Ok(())
//! # }
//! ```

use google_cloud_auth::credentials::CacheableResource;
use rig_core::catalog::{Catalog, ModelFacts};
use rig_core::completion::CompletionRequest;
use rig_core::driver::{Exchange, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::Completion;
use rig_core::providers::anthropic::streaming::{MessagesDecoder, MessagesDocument};
use rig_core::providers::anthropic::{ANTHROPIC, Addressing, AnthropicConfig, Messages, Prepared};
use rig_core::providers::registry::ProviderId;
use rig_core::wire::{Descriptor, Encoded, Mode, Secret, Wire, WireFrame};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::VertexAi;
use crate::client::VertexAiClientError;

/// The `anthropic_version` Vertex AI takes in the request body.
pub const ANTHROPIC_VERSION_VERTEX: &str = "vertex-2023-10-16";

/// The `rawPredict` endpoint of one Claude model on Vertex AI.
///
/// [`messages`](Self::messages) is the Messages wire this encodes with. Its
/// provider's `base_url` is the publisher's models collection
/// (`https://{host}/v1/projects/{project}/locations/{location}/publishers/anthropic/models`),
/// its `version` the body's `anthropic_version`, and its `betas` the
/// `anthropic-beta` header. Its API key is never sent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawPredict {
    /// The Messages wire, on Anthropic's own dialect.
    pub messages: Messages,
}

impl RawPredict {
    /// `model` in `project` at `location`. `global` and the `us` and `eu`
    /// multi-regions have endpoints of their own; any other location is a
    /// region (`us-east5`).
    pub fn new(project: &str, location: &str, model: impl Into<String>) -> Self {
        let mut provider = AnthropicConfig::with_key(&ANTHROPIC, Secret::default())
            .with_version(ANTHROPIC_VERSION_VERTEX);
        provider.addressing = Addressing::Url;
        provider.base_url = format!(
            "https://{}/v1/projects/{project}/locations/{location}/publishers/anthropic/models",
            host(location)
        );
        let model = model.into();
        let messages = provider.completion(model.as_str());
        let messages = match snapshot_facts(&model) {
            Some(facts) => messages.with_facts(facts),
            None => messages,
        };
        Self { messages }
    }
}

/// The facts of a dated Vertex AI id (`claude-haiku-4-5@20251001`): the
/// catalog's Claude model before the `@`, bound to the id as written, so its
/// output limit, reasoning modes and pricing apply. `None` for an undated
/// id, which the catalog reads as it is.
fn snapshot_facts(model: &str) -> Option<ModelFacts> {
    let (base, _) = model.split_once('@')?;
    let anthropic = ProviderId::catalog(ANTHROPIC.name)?;
    let mut spec = Catalog::builtin().get_exact(anthropic, base)?.clone();
    spec.id = model.to_owned();
    Some(ModelFacts::new(spec))
}

/// The API host serving `location`.
fn host(location: &str) -> String {
    match location {
        "global" => "aiplatform.googleapis.com".to_owned(),
        "us" | "eu" => format!("aiplatform.{location}.rep.googleapis.com"),
        region => format!("{region}-aiplatform.googleapis.com"),
    }
}

impl Wire for RawPredict {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = MessagesDecoder;
    type Reassembler = MessagesDocument;

    fn describe(&self) -> Descriptor<'_> {
        self.messages.describe()
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        let provider = &self.messages.provider;
        self.messages.encode_with(request, mode, |prepared| {
            let Prepared {
                model, body, betas, ..
            } = prepared;
            let method = match mode {
                Mode::Unary => "rawPredict",
                Mode::Streaming => "streamRawPredict",
            };
            let mut builder =
                http::Request::post(format!("{}/{model}:{method}", provider.base_url))
                    .header(http::header::CONTENT_TYPE, "application/json");
            if !betas.is_empty() {
                builder = builder.header("anthropic-beta", betas.join(","));
            }
            Ok(builder.body(body.into_body())?)
        })
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        self.messages.decoder()
    }

    fn reassembler(&self) -> Self::Reassembler {
        self.messages.reassembler()
    }
}

/// Sends through the client's HTTP transport with the client's Google
/// credentials. A client built on a supplied prediction service holds no
/// credentials and fails every request.
impl Transport<RawPredict> for VertexAi {
    fn send(&self, mut payload: Encoded, exchange: Exchange) -> Opening<WireFrame> {
        let client = self.clone();
        Opening::new(async move {
            let credentials = client
                .credentials()
                .ok_or_else(|| ProviderError::request(VertexAiClientError::NoCredentials))?;
            let headers = match credentials.headers(http::Extensions::new()).await {
                Ok(CacheableResource::New { data, .. }) => data,
                // Only a request that names a cached entity tag gets this.
                Ok(CacheableResource::NotModified) => http::HeaderMap::new(),
                Err(error) => {
                    return Err(ProviderError::request(VertexAiClientError::Credentials(
                        Arc::new(error),
                    )));
                }
            };
            payload.request.headers_mut().extend(headers);
            <_ as Transport<RawPredict>>::send(client.http(), payload, exchange).await
        })
    }
}

#[cfg(test)]
mod tests;
