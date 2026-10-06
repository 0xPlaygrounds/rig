//! The Bedrock text-embedding wire over `InvokeModel`: one request per text.
//!
//! ```no_run
//! use rig_bedrock::{client::BedrockRuntime, embedding::{AMAZON_TITAN_EMBED_TEXT_V2_0, Embeddings}};
//! use rig_core::Model;
//!
//! let model = BedrockRuntime::from_env().embedding(AMAZON_TITAN_EMBED_TEXT_V2_0, Some(256));
//! # let _ = model;
//! ```

use aws_smithy_types::Blob;
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::embeddings::{self, Embedding};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::wire::{Capabilities, Decoder, Descriptor, Flow, Mode, Out, Wire, WireEvent};
use serde::{Deserialize, Serialize};

use crate::client::BedrockRuntime;
use crate::completion::PROVIDER_NAME;
use crate::types::errors::sdk_error;
use rig_core::Model;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingRequest {
    pub input_text: String,
    pub dimensions: usize,
    pub normalize: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingResponse {
    pub embedding: Vec<f64>,
    pub input_text_token_count: usize,
}

/// `amazon.titan-embed-text-v1`
pub const AMAZON_TITAN_EMBED_TEXT_V1: &str = "amazon.titan-embed-text-v1";
/// `amazon.titan-embed-image-v1` multimodal embedding model
pub const AMAZON_TITAN_EMBED_IMAGE_V1: &str = "amazon.titan-embed-image-v1";
/// `amazon.titan-embed-text-v2:0`
pub const AMAZON_TITAN_EMBED_TEXT_V2_0: &str = "amazon.titan-embed-text-v2:0";
/// `cohere.embed-english-v3`
pub const COHERE_EMBED_ENGLISH_V3: &str = "cohere.embed-english-v3";
/// `cohere.embed-multilingual-v3`
pub const COHERE_EMBED_MULTILINGUAL_V3: &str = "cohere.embed-multilingual-v3";

/// The embedding endpoint for one model, at a caller-chosen width.
#[derive(Clone, Debug, PartialEq)]
pub struct Embeddings {
    pub model: String,
    pub ndims: Option<usize>,
}

impl BedrockRuntime {
    /// The embedding model for `model`, `ndims` wide when set.
    pub fn embedding(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Model<Embeddings, Self> {
        Model::new(Embeddings::new(model, ndims), self.clone())
    }
}

impl Embeddings {
    pub fn new(model: impl Into<String>, ndims: Option<usize>) -> Self {
        Self {
            model: model.into(),
            ndims,
        }
    }
}

/// One text's embedding request: the model, the text, and its body.
pub struct EmbeddingBatch {
    model: String,
    texts: Vec<(String, String)>,
}

/// One text's reply, or the failure that took its place. Bedrock embeds one
/// text per request, and a failed text does not stop the rest.
pub enum EmbeddingFrame {
    Embedded {
        document: String,
        response: EmbeddingResponse,
    },
    Failed(ProviderError),
}

impl Wire for Embeddings {
    type Op = rig_core::operation::Embedding;
    type Payload = EmbeddingBatch;
    type Frame = EmbeddingFrame;
    type Decoder<'id> = EmbeddingsDecoder;
    type Reassembler = rig_core::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .capabilities(Capabilities::embedding(
                1024,
                self.ndims.unwrap_or_default(),
            ))
    }

    fn encode(&self, texts: Vec<String>, _mode: Mode) -> Result<EmbeddingBatch, EncodeError> {
        let texts = texts
            .into_iter()
            .map(|text| {
                let body = serde_json::to_string(&EmbeddingRequest {
                    input_text: text.clone(),
                    dimensions: self.ndims.unwrap_or_default(),
                    normalize: true,
                })?;
                Ok((text, body))
            })
            .collect::<Result<_, EncodeError>>()?;
        Ok(EmbeddingBatch {
            model: self.model.clone(),
            texts,
        })
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        EmbeddingsDecoder::default()
    }
}

impl Transport<Embeddings> for BedrockRuntime {
    fn send(&self, batch: EmbeddingBatch, _exchange: Exchange) -> Opening<EmbeddingFrame> {
        let runtime = self.clone();
        // Every call completes inside the send, so the calls run under the
        // attempt's span; sequential requests limit load against account
        // quotas.
        Opening::new(async move {
            let client = runtime.inner().await.clone();
            let mut frames = Vec::with_capacity(batch.texts.len());
            for (document, body) in batch.texts {
                let sent = client
                    .invoke_model()
                    .model_id(batch.model.as_str())
                    .content_type("application/json")
                    .accept("application/json")
                    .body(Blob::new(body))
                    .send()
                    .await;
                let reply = sent
                    .map_err(sdk_error)
                    .and_then(|response| {
                        String::from_utf8(response.body.into_inner())
                            .map_err(|error| ProviderError::Response(error.to_string()))
                    })
                    .and_then(|body| serde_json::from_str(&body).map_err(ProviderError::from));
                frames.push(Ok(match reply {
                    Ok(response) => EmbeddingFrame::Embedded { document, response },
                    Err(error) => EmbeddingFrame::Failed(error),
                }));
            }
            Ok(Opened::new(futures::stream::iter(frames)))
        })
    }
}

/// Collects every text's reply into one response; the first failure fails
/// the batch once every text was sent. The transport sends every text before
/// its frames end, so the batch ends there.
#[derive(Default)]
pub struct EmbeddingsDecoder {
    embeddings: Vec<Embedding>,
    raw: Vec<serde_json::Value>,
    usage: rig_core::completion::Usage,
    failure: Option<ProviderError>,
}

impl<'id> Decoder<'id, rig_core::operation::Embedding, EmbeddingFrame> for EmbeddingsDecoder {
    type Event = EmbeddingFrame;

    fn classify(&self, frame: EmbeddingFrame) -> WireEvent<EmbeddingFrame> {
        WireEvent::Known(frame)
    }

    fn decode(
        &mut self,
        frame: EmbeddingFrame,
        _out: Out<'id, rig_core::operation::Embedding>,
    ) -> Result<Flow, ProviderError> {
        let (document, response) = match frame {
            EmbeddingFrame::Embedded { document, response } => (document, response),
            EmbeddingFrame::Failed(error) => {
                self.failure.get_or_insert(error);
                return Ok(Flow::More);
            }
        };
        let tokens = response.input_text_token_count as u64;
        self.usage += rig_core::completion::Usage::new()
            .input_tokens(tokens)
            .total_tokens(tokens);
        self.raw.push(serde_json::to_value(&response)?);
        self.embeddings.push(Embedding {
            document,
            vec: response.embedding,
        });
        Ok(Flow::More)
    }

    fn eof(
        &mut self,
        mut out: Out<'id, rig_core::operation::Embedding>,
    ) -> Result<Flow, ProviderError> {
        if let Some(error) = self.failure.take() {
            return Err(ProviderError::Response(error.to_string()));
        }
        out.raw(serde_json::Value::Array(std::mem::take(&mut self.raw)));
        Ok(out.end(embeddings::EmbeddingResponse {
            usage: self.usage,
            ..embeddings::EmbeddingResponse::new(std::mem::take(&mut self.embeddings))
        }))
    }
}
