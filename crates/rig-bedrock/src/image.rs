//! The Bedrock text-to-image wire over `InvokeModel`.
//!
//! ```no_run
//! use rig_bedrock::{client::BedrockRuntime, image::{AMAZON_NOVA_CANVAS, Images}};
//! use rig_core::Model;
//!
//! let model = BedrockRuntime::from_env().image_generation(AMAZON_NOVA_CANVAS);
//! # let _ = model;
//! ```

use crate::client::BedrockRuntime;
use crate::completion::PROVIDER_NAME;
use crate::types::errors::sdk_error;
use crate::types::text_to_image::{TextToImageGeneration, TextToImageResponse};
use aws_smithy_types::Blob;
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::image_generation::{ImageGenerationRequest, NormalizeImageGenerationResponse};
use rig_core::operation::ImageGeneration;
use rig_core::wire::{Decoder, Descriptor, Flow, Mode, Out, Wire, WireEvent};

pub use crate::completion::{
    AMAZON_NOVA_CANVAS, STABILITY_SD3_5_LARGE, STABILITY_STABLE_IMAGE_CORE_1_0,
    STABILITY_STABLE_IMAGE_ULTRA_1_0,
};

/// The image-generation endpoint for one model.
#[derive(Clone, Debug, PartialEq)]
pub struct Images {
    pub model: String,
}

impl Images {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

/// One `InvokeModel` request: the model and its JSON body.
pub struct InvokeModel {
    model: String,
    body: String,
}

impl Wire for Images {
    type Op = ImageGeneration;
    type Payload = InvokeModel;
    type Frame = Vec<u8>;
    type Decoder<'id> = ImagesDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME).model(self.model.as_str())
    }

    fn encode(
        &self,
        request: ImageGenerationRequest,
        _mode: Mode,
    ) -> Result<InvokeModel, EncodeError> {
        let request = TextToImageGeneration::new(request.prompt)
            .width(request.width)
            .height(request.height);
        Ok(InvokeModel {
            model: self.model.clone(),
            body: serde_json::to_string(&request)?,
        })
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ImagesDecoder
    }
}

impl Transport<Images> for BedrockRuntime {
    fn send(&self, payload: InvokeModel, _exchange: Exchange) -> Opening<Vec<u8>> {
        let runtime = self.clone();
        Opening::new(async move {
            let sent = runtime
                .inner()
                .await
                .invoke_model()
                .model_id(payload.model.as_str())
                .content_type("application/json")
                .accept("application/json")
                .body(Blob::new(payload.body))
                .send()
                .await;
            Ok(match sent {
                Ok(response) => {
                    let request_id =
                        aws_sdk_bedrockruntime::operation::RequestId::request_id(&response)
                            .map(str::to_owned);
                    Opened::new(futures::stream::iter([Ok(response.body.into_inner())]))
                        .with_request_id(request_id)
                }
                Err(error) => Opened::failed(sdk_error(error)),
            })
        })
    }
}

/// Decodes the one `TextToImageResponse` an image request returns.
pub struct ImagesDecoder;

impl<'id> Decoder<'id, ImageGeneration, Vec<u8>> for ImagesDecoder {
    type Event = Vec<u8>;

    fn classify(&self, body: Vec<u8>) -> WireEvent<Vec<u8>> {
        WireEvent::Known(body)
    }

    fn decode(
        &mut self,
        body: Vec<u8>,
        mut out: Out<'id, ImageGeneration>,
    ) -> Result<Flow, ProviderError> {
        let body =
            String::from_utf8(body).map_err(|error| ProviderError::Response(error.to_string()))?;
        let response = serde_json::from_str::<TextToImageResponse>(&body)
            .map_err(|error| ProviderError::Response(error.to_string()))?;
        out.raw(serde_json::to_value(&response)?);
        Ok(out.end(response.normalize()?))
    }
}
