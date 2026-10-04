//! Model listing and credential verification for a Messages-format
//! provider: `GET /v1/models`, paged by cursor, and the same endpoint read
//! for its status alone.
//!
//! ```no_run
//! use rig_core::providers::anthropic::Anthropic;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let models = Anthropic::from_env()?.list_models().await?;
//! # let _ = models;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

use super::wire::AnthropicConfig;
use crate::error::{EncodeError, ProviderError};
use crate::model::{ModelInfo, ModelList};
pub use crate::operation::VerifyDecoder;
use crate::operation::{ModelListing, ModelPage, Verify as VerifyOp};
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent, WireFrame,
};

impl AnthropicConfig {
    /// The model-listing wire.
    pub(crate) fn models(&self) -> Models {
        Models {
            provider: self.clone(),
        }
    }

    /// The credential-check wire.
    pub(crate) fn verify(&self) -> Verify {
        Verify {
            provider: self.clone(),
        }
    }
}

/// The model-listing wire: `GET /v1/models`, cursor-paged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// The provider this wire speaks to.
    pub provider: AnthropicConfig,
}

impl Wire for Models {
    type Op = ModelListing;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ModelsDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider.dialect.name)
    }

    fn encode(&self, cursor: Option<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        Ok(Encoded::new(
            self.models_request(cursor.as_deref())?,
            Framing::Whole,
        ))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ModelsDecoder
    }
}

impl Models {
    /// One page's request, after `cursor` when the previous page named one.
    fn models_request(&self, cursor: Option<&str>) -> Result<http::Request<Body>, EncodeError> {
        let uri = match cursor {
            Some(cursor) => format!(
                "{}{}",
                self.provider.base_url,
                crate::providers::internal::with_query_pairs("/v1/models", &[("after_id", cursor)],)
            ),
            None => format!("{}/v1/models", self.provider.base_url),
        };
        self.provider
            .headers(http::Request::get(uri))
            .body(Body::empty())
            .map_err(EncodeError::from)
    }
}

/// One page of `GET /v1/models`.
#[derive(Debug, Deserialize)]
#[doc(hidden)]
pub struct ModelsPage {
    data: Vec<ModelEntry>,
    #[serde(default)]
    has_more: bool,
    #[serde(default)]
    last_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    display_name: String,
}

impl From<ModelEntry> for ModelInfo {
    fn from(entry: ModelEntry) -> Self {
        ModelInfo::new(entry.id, entry.display_name)
    }
}

/// Decodes `GET /v1/models` and the cursor Anthropic names.
pub struct ModelsDecoder;

impl<'id> Decoder<'id, ModelListing> for ModelsDecoder {
    type Event = ModelsPage;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_marker_keyed_frame(&frame.as_str(), &["data"])
    }

    fn decode(
        &mut self,
        page: Self::Event,
        out: Out<'id, ModelListing>,
    ) -> Result<Flow, ProviderError> {
        // Missing or empty cursors would repeatedly fetch page one, even with has_more.
        let next = page
            .last_id
            .filter(|cursor| page.has_more && !cursor.is_empty());
        Ok(out.end(ModelPage {
            models: ModelList::new(page.data.into_iter().map(ModelInfo::from).collect()),
            next,
        }))
    }
}

/// The credential-check wire: `GET /v1/models`, status only, decoded by
/// the shared [`VerifyDecoder`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verify {
    /// The provider this wire speaks to.
    pub provider: AnthropicConfig,
}

impl Wire for Verify {
    type Op = VerifyOp;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = VerifyDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider.dialect.name)
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = self
            .provider
            .headers(http::Request::get(format!(
                "{}/v1/models",
                self.provider.base_url
            )))
            .body(Body::empty())?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        VerifyDecoder
    }
}
