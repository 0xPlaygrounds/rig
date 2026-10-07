//! Copilot's model-listing wire: `GET /models` with the editor envelope.
//!
//! ```no_run
//! use rig_core::providers::copilot::Copilot;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let models = Copilot::from_env()?.list_models().await?;
//! # let _ = models;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::CopilotIntent;
use super::PROVIDER_NAME;
use super::wire::{CopilotConfig, REQUEST_ID_HEADER, stamp};
use crate::error::{EncodeError, ProviderError};
use crate::json_utils::Lenient;
use crate::model::{ModelInfo, ModelList};
use crate::operation::{ModelListing, ModelPage};
use crate::providers::internal::wire::classify_untyped_line;
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent, WireFrame,
};

/// Catalogue endpoint returning all models without pagination.
pub(crate) const MODEL_LISTING_PATH: &str = "/models";

impl CopilotConfig {
    /// The model-listing wire.
    pub(crate) fn models(&self) -> Models {
        Models {
            provider: self.clone(),
        }
    }
}

/// Copilot's model-listing wire.
///
/// `GET /models` answers with the whole catalogue, so a page never names a
/// next one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// Which Copilot, and how to reach it.
    pub provider: CopilotConfig,
}

/// The model-listing decoder: the catalogue's `data`, each entry naming
/// its id, its name, its vendor and its modality under `capabilities.type`.
#[derive(Default)]
pub struct ModelsDecoder;

impl<'id> Decoder<'id, ModelListing> for ModelsDecoder {
    type Event = Value;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_untyped_line(frame.as_str().as_bytes())
    }

    fn decode(&mut self, event: Value, out: Out<'id, ModelListing>) -> Result<Flow, ProviderError> {
        let text = |entry: &Value, pointer: &str| {
            entry.at(pointer).and_then(Value::as_str).map(str::to_owned)
        };
        let models = event
            .arr("data")
            .iter()
            .filter_map(|entry| {
                let mut model = ModelInfo::from_id(entry.str("id")?);
                model.name = text(entry, "/name");
                model.owned_by = text(entry, "/vendor");
                model.r#type = text(entry, "/capabilities/type");
                Some(model)
            })
            .collect();
        Ok(out.end(ModelPage {
            models: ModelList::new(models),
            next: None,
        }))
    }
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
        let mut request = http::Request::get(self.provider.uri(MODEL_LISTING_PATH))
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::empty())?;
        stamp(
            &mut request,
            self.provider.api_key.expose(),
            "user",
            false,
            CopilotIntent::Panel,
        )?;
        Ok(Encoded::new(request, Framing::Whole).with_request_id_header(REQUEST_ID_HEADER))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ModelsDecoder
    }
}
