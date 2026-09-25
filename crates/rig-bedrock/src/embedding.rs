//! The Bedrock text-embedding wire over `InvokeModel`: one request per text.
//!
//! ```no_run
//! use rig_core::wire::Wire as _;
//! use rig_bedrock::{client::BedrockRuntime, embedding::{AMAZON_TITAN_EMBED_TEXT_V2_0, Embeddings}};
//!
//! let model = Embeddings::new(AMAZON_TITAN_EMBED_TEXT_V2_0, Some(256)).on(BedrockRuntime::from_env());
//! # let _ = model;
//! ```

use aws_smithy_types::Blob;
use rig_core::driver::{Observation, Opened, Transport};
use rig_core::embeddings::{self, Embedding};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::{EmbeddingCapabilities, One};
use rig_core::providers::internal::wire::{self, TypedEvent, WireEvent};
use rig_core::wire::{Decoder, Mode, Output, Sink, Wire};
use serde::{Deserialize, Serialize};

use crate::client::BedrockRuntime;
use crate::types::assistant_content::PROVIDER_NAME;
use crate::types::errors::AwsSdkInvokeModelError;

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

pub use crate::completion::{
    AMAZON_TITAN_EMBEDDINGS_G1_TEXT as AMAZON_TITAN_EMBED_TEXT_V1,
    AMAZON_TITAN_MULTIMODAL_EMBEDDINGS_G1 as AMAZON_TITAN_EMBED_IMAGE_V1,
    AMAZON_TITAN_TEXT_EMBEDDINGS_V2 as AMAZON_TITAN_EMBED_TEXT_V2_0,
    COHERE_EMBED_ENGLISH as COHERE_EMBED_ENGLISH_V3,
    COHERE_EMBED_MULTILINGUAL as COHERE_EMBED_MULTILINGUAL_V3,
};

/// The embedding endpoint for one model, at a caller-chosen width.
#[derive(Clone, Debug, PartialEq)]
pub struct Embeddings {
    pub model: String,
    pub ndims: Option<usize>,
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
    type Decoder = EmbeddingsDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn capabilities(&self) -> EmbeddingCapabilities {
        EmbeddingCapabilities::new(1024, self.ndims.unwrap_or_default())
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

    fn decoder(&self, _mode: Mode) -> EmbeddingsDecoder {
        EmbeddingsDecoder::default()
    }
}

impl Transport<Embeddings> for BedrockRuntime {
    fn send(
        &self,
        batch: EmbeddingBatch,
        _mode: Mode,
        _observation: Option<Observation>,
    ) -> Result<
        impl Future<Output = Opened<EmbeddingBatch, EmbeddingFrame>> + Send + 'static + use<>,
        ProviderError,
    > {
        let runtime = self.clone();
        // Every call completes inside the send, so the calls run under the
        // attempt's span; sequential requests limit load against account
        // quotas.
        Ok(async move {
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
                    .map_err(|sdk_error| ProviderError::from(AwsSdkInvokeModelError(sdk_error)))
                    .and_then(|response| {
                        String::from_utf8(response.body.into_inner())
                            .map_err(|error| ProviderError::Response(error.to_string()))
                    })
                    .and_then(|body| serde_json::from_str(&body).map_err(ProviderError::Json));
                frames.push(Ok(match reply {
                    Ok(response) => EmbeddingFrame::Embedded { document, response },
                    Err(error) => EmbeddingFrame::Failed(error),
                }));
            }
            Opened::new(futures::stream::iter(frames))
        })
    }
}

/// Collects every text's reply into one response; the first failure fails
/// the batch once every text was sent.
#[derive(Default)]
pub struct EmbeddingsDecoder {
    embeddings: Vec<Embedding>,
    raw: Vec<serde_json::Value>,
    usage: rig_core::completion::Usage,
    failure: Option<ProviderError>,
}

impl Decoder<rig_core::operation::Embedding, EmbeddingFrame> for EmbeddingsDecoder {
    type Event = EmbeddingFrame;

    fn classify(&self, frame: EmbeddingFrame) -> WireEvent<EmbeddingFrame> {
        wire::classify_typed_event(TypedEvent::Modeled(frame))
    }

    fn interpret(&mut self, frame: EmbeddingFrame, out: &mut One<rig_core::operation::Embedding>) {
        let EmbeddingFrame::Embedded { document, response } = frame else {
            if let EmbeddingFrame::Failed(error) = frame {
                self.failure.get_or_insert(error);
            }
            return;
        };
        let tokens = response.input_text_token_count as u64;
        self.usage += rig_core::completion::Usage {
            input_tokens: Some(tokens),
            total_tokens: Some(tokens),
            ..Default::default()
        };
        match serde_json::to_value(&response) {
            Ok(raw) => self.raw.push(raw),
            Err(error) => out.push(Err(error.into())),
        }
        self.embeddings.push(Embedding {
            document,
            vec: response.embedding,
        });
    }

    fn finish(&mut self, out: &mut Output<rig_core::operation::Embedding>) {
        out.push(match self.failure.take() {
            None => Ok(embeddings::EmbeddingResponse::new(
                std::mem::take(&mut self.embeddings),
                PROVIDER_NAME,
            )
            .with_usage(self.usage)
            .with_raw(serde_json::Value::Array(std::mem::take(&mut self.raw)))),
            Some(error) => Err(ProviderError::Response(error.to_string())),
        });
    }
}
