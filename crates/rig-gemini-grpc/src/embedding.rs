//! The Gemini text-embedding wire over gRPC: one `EmbedContent` call per text.
//!
//! ```no_run
//! use rig_core::Model;
//! use rig_gemini_grpc::{GeminiGrpc, embedding::{EMBEDDING_004, Embeddings}};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let model = GeminiGrpc::new("API_KEY").await?.embedding(EMBEDDING_004, None);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

/// `text-embedding-004` embedding model
pub const EMBEDDING_004: &str = "text-embedding-004";

use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::embeddings;
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::Embedding;
use rig_core::wire::{Capabilities, Decoder, Descriptor, Flow, Mode, Out, Wire, WireEvent};

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
    type Decoder<'id> = EmbeddingsDecoder;

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

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        EmbeddingsDecoder::default()
    }
}

impl Transport<Embeddings> for GeminiGrpc {
    fn send(
        &self,
        requests: Vec<(String, EmbedContentRequest)>,
        _exchange: Exchange,
    ) -> Opening<(String, proto::EmbedContentResponse)> {
        let mut client = match self.grpc_client() {
            Ok(client) => client,
            Err(error) => return Opening::failed(ProviderError::Provider(error.to_string())),
        };
        // Sequential calls, each completed inside the send so they run under
        // the attempt's span; the first RPC error ends the batch.
        Opening::new(async move {
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
            Ok(Opened::new(futures::stream::iter(frames)))
        })
    }
}

/// Collects every text's vector into one response.
#[derive(Default)]
pub struct EmbeddingsDecoder {
    embeddings: Vec<embeddings::Embedding>,
    failure: Option<ProviderError>,
}

impl<'id> Decoder<'id, Embedding, (String, proto::EmbedContentResponse)> for EmbeddingsDecoder {
    type Event = (String, proto::EmbedContentResponse);

    fn classify(
        &self,
        frame: (String, proto::EmbedContentResponse),
    ) -> WireEvent<(String, proto::EmbedContentResponse)> {
        WireEvent::Known(frame)
    }

    fn decode(
        &mut self,
        (document, response): Self::Event,
        _out: Out<'id, Embedding>,
    ) -> Result<Flow, ProviderError> {
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
        Ok(Flow::More)
    }

    /// The transport sends every text before its frames end, so the batch
    /// ends there.
    fn eof(&mut self, out: Out<'id, Embedding>) -> Result<Flow, ProviderError> {
        if let Some(error) = self.failure.take() {
            return Err(error);
        }
        // gRPC: the native answers are prost messages, not JSON, and
        // `EmbedContent` reports no usage or response id; `raw` stays `Null`.
        Ok(out.end(embeddings::EmbeddingResponse::new(
            std::mem::take(&mut self.embeddings),
            super::completion::PROVIDER_NAME,
        )))
    }
}

#[cfg(test)]
mod tests;
