//! Copilot's model-listing wire: `GET /models` with the editor envelope.
//!
//! ```no_run
//! use rig_core::driver::Model;
//! use rig_core::providers::copilot::{DIALECT, model_listing::Models};
//! use rig_core::providers::openai::OpenAIConfig;
//!
//! # async fn run(http: rig_core::http_client::DynHttpClient) -> Result<(), Box<dyn std::error::Error>> {
//! let config = OpenAIConfig::with_key(&DIALECT, "tid=session-token");
//! let models = Model::new(Models::new(config), http).list().await?;
//! let tool_models: Vec<_> = models.iter()
//!     .filter(|model| model.supports_tool_calls == Some(true))
//!     .collect();
//! # let _ = tool_models;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::PROVIDER_NAME;
use super::wire::{REQUEST_ID_HEADER, stamp};
use crate::error::{EncodeError, ProviderError};
use crate::json_utils::Lenient;
use crate::model::{ModelInfo, ModelList};
use crate::operation::{ModelListing, ModelPage};
use crate::providers::internal::wire::classify_untyped_line;
use crate::providers::openai::OpenAIConfig;
use crate::wire::{Body, Descriptor, DocumentDecoder, Encoded, Framing, Mode, Wire};

/// Catalogue endpoint returning all models without pagination.
pub(crate) const MODEL_LISTING_PATH: &str = "/models";

/// Copilot's model-listing wire.
///
/// `GET /models` answers with the whole catalogue, so a page never names a
/// next one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// Which Copilot, and how to reach it.
    pub provider: OpenAIConfig,
}

impl Models {
    /// The catalogue of the Copilot `provider` names.
    pub fn new(provider: OpenAIConfig) -> Self {
        Self { provider }
    }
}

/// The catalogue's `data`, each entry naming its id, its name, its vendor
/// and its modality under `capabilities.type`.
fn page(event: Value) -> Result<ModelPage, ProviderError> {
    let text =
        |entry: &Value, pointer: &str| entry.at(pointer).and_then(Value::as_str).map(str::to_owned);
    let models = event
        .arr("data")
        .iter()
        .filter_map(|entry| {
            let mut model = ModelInfo::from_id(entry.str("id")?);
            model.name = text(entry, "/name");
            model.owned_by = text(entry, "/vendor");
            model.r#type = text(entry, "/capabilities/type");
            model.supports_tool_calls = entry
                .at("/capabilities/supports/tool_calls")
                .and_then(Value::as_bool);
            Some(model)
        })
        .collect();
    Ok(ModelPage {
        models: ModelList::new(models),
        next: None,
    })
}

impl Wire for Models {
    type Op = ModelListing;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Event = Value;
    type Decoder<'id> = DocumentDecoder<ModelListing, Value>;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
    }

    fn encode(&self, _cursor: Option<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let uri = format!(
            "{}{MODEL_LISTING_PATH}",
            self.provider.base_url.trim_end_matches('/')
        );
        let mut request = http::Request::get(uri)
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::empty())?;
        stamp(&mut request, self.provider.api_key.expose(), "user", false)?;
        Ok(Encoded::new(request, Framing::Whole).with_request_id_header(REQUEST_ID_HEADER))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        DocumentDecoder::new(|body| classify_untyped_line(body.as_bytes()), page)
    }
}

#[cfg(test)]
mod tests;
