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
use rig_core::catalog::Catalog;
use rig_core::completion::CompletionRequest;
use rig_core::driver::{Exchange, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::Completion;
use rig_core::providers::anthropic::streaming::{MessagesDecoder, MessagesDocument, MessagesEvent};
use rig_core::providers::anthropic::{ANTHROPIC, Addressing, AnthropicConfig, Messages, Prepared};
use rig_core::providers::registry::ProviderId;
use rig_core::wire::{Descriptor, Encoded, Mode, Secret, Wire, WireFrame};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};

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
        let messages = match ProviderId::catalog(ANTHROPIC.name) {
            Some(anthropic) => messages.with_facts(catalog().facts(anthropic, &model)),
            None => messages,
        };
        Self { messages }
    }
}

/// The built-in catalog with every Claude model Vertex AI names by a dated
/// id (`claude-haiku-4-5@20251001`) also listed as Anthropic's model under
/// that id, so its output limit, reasoning modes and pricing apply to a
/// wire or a request that names it.
fn catalog() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        let builtin = Catalog::builtin();
        let mut catalog = builtin.clone();
        let Some(anthropic) = ProviderId::catalog(ANTHROPIC.name) else {
            return catalog;
        };
        let dated = builtin
            .iter()
            .filter(|spec| {
                spec.provider.vendor() == crate::types::completion_response::PROVIDER_NAME
            })
            .filter_map(|spec| {
                let (base, _) = spec.id.split_once('@')?;
                let mut claude = builtin.get_exact(anthropic, base)?.clone();
                claude.id = spec.id.clone();
                Some(claude)
            })
            .collect::<Vec<_>>();
        for claude in dated {
            catalog.insert(claude);
        }
        catalog
    })
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
    type Event = MessagesEvent;
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
