//! The Gemini text-embedding wire over gRPC: one `EmbedContent` call per text.
//!
//! ```no_run
//! use rig_core::{Model, RigError};
//! use rig_gemini_grpc::{GeminiGrpc, embedding::{EMBEDDING_004, Embeddings}};
//!
//! # async fn example() -> Result<(), RigError> {
//! let model = GeminiGrpc::new("API_KEY")
//!     .await
//!     .map_err(|error| RigError::other(&*error))?
//!     .embedding(EMBEDDING_004, None);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

/// `text-embedding-004` embedding model
pub const EMBEDDING_004: &str = "text-embedding-004";

use rig_core::driver::{Exchange, Opened, Sending, Transport};
use rig_core::embeddings;
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::Embedding;
use rig_core::providers::internal::wire;
use rig_core::wire::{
    Capabilities, Decoder, Descriptor, End, Mode, Out, TypedEvent, Wire, WireEvent,
};

use super::GeminiGrpc;
use super::proto::{self, EmbedContentRequest};

/// The embedding endpoint for one model, at a chosen width.
#[derive(Clone, Debug, PartialEq)]
pub struct Embeddings {
    pub model: String,
    pub ndims: usize,
}

impl Embeddings {
    pub fn new(model: impl Into<String>, dims: Option<usize>) -> Self {
        Self {
            model: model.into(),
            ndims: dims.unwrap_or(768), // Default embedding size for text-embedding-004
        }
    }
}

impl Wire for Embeddings {
    type Op = Embedding;
    type Payload = Vec<(String, EmbedContentRequest)>;
    type Frame = (String, proto::EmbedContentResponse);
    type Decoder = EmbeddingsDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::completion::PROVIDER_NAME)
            .model(self.model.as_str())
            .capabilities(Capabilities::embedding(100, self.ndims))
    }

    fn encode(
        &self,
        texts: Vec<String>,
        _mode: Mode,
    ) -> Result<Vec<(String, EmbedContentRequest)>, EncodeError> {
        Ok(texts
            .into_iter()
            .map(|text| {
                let request = EmbedContentRequest {
                    model: format!("models/{}", self.model),
                    content: Some(proto::Content {
                        parts: vec![super::completion::text_part(text.clone())],
                        role: String::new(),
                    }),
                    task_type: None,
                    title: None,
                    output_dimensionality: Some(self.ndims as i32),
                };
                (text, request)
            })
            .collect())
    }

    fn decoder(&self, _mode: Mode) -> EmbeddingsDecoder {
        EmbeddingsDecoder::default()
    }
}

impl Transport<Embeddings> for GeminiGrpc {
    fn send(
        &self,
        requests: Vec<(String, EmbedContentRequest)>,
        _exchange: Exchange,
    ) -> Result<Sending<(String, proto::EmbedContentResponse)>, ProviderError> {
        let mut client = self
            .grpc_client()
            .map_err(|error| ProviderError::Provider(error.to_string()))?;
        // Sequential calls, each completed inside the send so they run under
        // the attempt's span; the first RPC error ends the batch.
        Ok(Sending::later(async move {
            let mut frames = Vec::with_capacity(requests.len());
            for (text, request) in requests {
                match client.embed_content(request).await {
                    Ok(response) => frames.push(Ok((text, response.into_inner()))),
                    Err(status) => {
                        frames.push(Err(super::completion::rpc_error(&status)));
                        break;
                    }
                }
            }
            Opened::new(futures::stream::iter(frames))
        }))
    }
}

/// Collects every text's vector into one response.
#[derive(Default)]
pub struct EmbeddingsDecoder {
    embeddings: Vec<embeddings::Embedding>,
    failure: Option<ProviderError>,
}

impl Decoder<Embedding, (String, proto::EmbedContentResponse)> for EmbeddingsDecoder {
    type Event = (String, proto::EmbedContentResponse);

    fn classify(
        &self,
        frame: (String, proto::EmbedContentResponse),
    ) -> WireEvent<(String, proto::EmbedContentResponse)> {
        wire::classify_typed_event(TypedEvent::Modeled(frame))
    }

    fn interpret(&mut self, (document, response): Self::Event, _out: &mut Out<'_, Embedding>) {
        match response.embedding {
            Some(embedding) => self.embeddings.push(embeddings::Embedding {
                document,
                vec: embedding.values.into_iter().map(f64::from).collect(),
            }),
            None => {
                self.failure.get_or_insert(ProviderError::Response(
                    "No embedding in response".to_string(),
                ));
            }
        }
    }

    fn end(&mut self, out: &mut Out<'_, Embedding>, end: End) {
        if end != End::Eof {
            return;
        }
        // gRPC: the native answers are prost messages, not JSON, and
        // `EmbedContent` reports no usage or response id; `raw` stays `Null`.
        out.push(match self.failure.take() {
            None => Ok(embeddings::EmbeddingResponse::new(
                std::mem::take(&mut self.embeddings),
                super::completion::PROVIDER_NAME,
            )),
            Some(error) => Err(error),
        });
    }
}

#[cfg(test)]
mod tests;
