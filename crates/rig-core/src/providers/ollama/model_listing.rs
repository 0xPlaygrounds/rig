//! Ollama's model-listing wire, `GET /api/tags`.
//!
//! ```
//! use rig_core::providers::ollama::ListModelsResponse;
//! let reply: ListModelsResponse =
//!     serde_json::from_str(r#"{"models": [{"name": "qwen3:4b", "model": "qwen3:4b"}]}"#)?;
//! assert_eq!(reply.models.len(), 1);
//! # Ok::<(), serde_json::Error>(())
//! ```

use crate::error::{EncodeError, ProviderError};
use crate::model::{ModelInfo, ModelList};
use crate::operation::{ModelListing, ModelPage};
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent, WireFrame,
};
use serde::{Deserialize, Serialize};

use super::{OllamaConfig, PROVIDER_NAME};

/// The reply of `GET /api/tags`: every model the daemon has pulled.
#[derive(Debug, Deserialize)]
pub struct ListModelsResponse {
    /// The installed models, in the daemon's own order.
    pub models: Vec<ListModelEntry>,
}

/// One installed model.
#[derive(Debug, Deserialize)]
pub struct ListModelEntry {
    /// The tag as the daemon displays it (`qwen3:4b`).
    pub name: String,
    /// The identifier a request addresses.
    pub model: String,
}

impl From<ListModelEntry> for ModelInfo {
    fn from(value: ListModelEntry) -> Self {
        ModelInfo::new(value.model, value.name)
    }
}

impl OllamaConfig {
    /// The model-listing wire.
    pub(crate) fn models(&self) -> Models {
        Models {
            provider: self.clone(),
        }
    }
}

/// The model-listing wire: `GET /api/tags`, unpaged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// The daemon this wire speaks to.
    pub provider: OllamaConfig,
}

impl Wire for Models {
    type Op = ModelListing;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ModelsDecoder;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
    }

    fn encode(&self, _cursor: Option<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = self
            .provider
            .request(http::Method::GET, "/api/tags")
            .body(Body::empty())?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ModelsDecoder
    }
}

/// Decodes `GET /api/tags`. The daemon answers with every installed model at
/// once, so there is no cursor to follow.
pub struct ModelsDecoder;

impl<'id> Decoder<'id, ModelListing> for ModelsDecoder {
    type Event = ListModelsResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_marker_keyed_frame(&frame.as_str(), &["models"])
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, ModelListing>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(ModelPage {
            models: ModelList::new(reply.models.into_iter().map(ModelInfo::from).collect()),
            next: None,
        }))
    }
}

#[cfg(test)]
mod tests;
