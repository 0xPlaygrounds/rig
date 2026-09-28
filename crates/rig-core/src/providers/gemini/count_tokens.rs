//! `countTokens`: what a completion request would cost before sending it.
//! The body is the one `generateContent` would send, settings and cache
//! included.
//!
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::providers::gemini::{self, Gemini};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Gemini::from_env()?.completion(gemini::GEMINI_3_8_FLASH);
//! let count = model.count_tokens(CompletionRequest::new("How long is this?")).await?;
//! println!("{:?}", count.total_tokens);
//! # Ok(())
//! # }
//! ```

use serde::Serialize;

use super::api;
use super::completion::GenerateContent;
use super::generate_content::Body;
use crate::completion::CompletionRequest;
use crate::error::{EncodeError, ProviderError};
use crate::operation::TokenCount;
use crate::providers::internal::wire::classify_untyped_line;
use crate::wire::{
    Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent, WireFrame,
};

/// The `countTokens` wire of a GenerateContent model.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CountTokens {
    /// The model whose requests are counted.
    pub generate: GenerateContent,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CountBody<'a> {
    generate_content_request: WithModel<'a>,
}

#[derive(Serialize)]
struct WithModel<'a> {
    model: String,
    #[serde(flatten)]
    body: &'a Body,
}

impl Wire for CountTokens {
    type Op = TokenCount;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = CountTokensDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME).model(self.generate.model.as_str())
    }

    fn encode(&self, request: CompletionRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.generate.model.clone());
        let request = request.replayable_to(&[super::ISSUER])?;
        let body = super::generate_content::body(
            request,
            &self.generate.settings,
            self.generate.cached_content.as_ref(),
        )?;
        let count = CountBody {
            generate_content_request: WithModel {
                model: format!("models/{model}"),
                body: &body,
            },
        };
        let request = http::Request::post(
            self.generate
                .provider
                .uri(&format!("/v1beta/models/{model}:countTokens")),
        )
        .header("Content-Type", "application/json")
        .body(crate::wire::Body::Bytes(serde_json::to_vec(&count)?))?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        CountTokensDecoder
    }
}

/// Decodes one `countTokens` reply.
pub struct CountTokensDecoder;

impl<'id> Decoder<'id, TokenCount> for CountTokensDecoder {
    type Event = api::CountTokensResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_untyped_line(frame.as_str().as_bytes())
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, TokenCount>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(reply))
    }
}

#[cfg(test)]
mod tests;
