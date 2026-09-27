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
use crate::types::assistant_content::PROVIDER_NAME;
use crate::types::errors::AwsSdkInvokeModelError;
use crate::types::text_to_image::{TextToImageGeneration, TextToImageResponse};
use aws_smithy_types::Blob;
use rig_core::driver::{Exchange, Opened, Sending, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::image_generation::{ImageGenerationRequest, NormalizeImageGenerationResponse};
use rig_core::operation::ImageGeneration;
use rig_core::providers::internal::wire;
use rig_core::wire::{Decoder, Descriptor, Mode, Out, TypedEvent, Wire, WireEvent};

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
    type Decoder = ImagesDecoder;

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

    fn decoder(&self, _mode: Mode) -> ImagesDecoder {
        ImagesDecoder
    }
}

impl Transport<Images> for BedrockRuntime {
    fn send(
        &self,
        payload: InvokeModel,
        _exchange: Exchange,
    ) -> Result<Sending<Vec<u8>>, ProviderError> {
        let runtime = self.clone();
        Ok(Sending::later(async move {
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
            match sent {
                Ok(response) => {
                    let request_id =
                        aws_sdk_bedrockruntime::operation::RequestId::request_id(&response)
                            .map(str::to_owned);
                    Opened::new(futures::stream::iter([Ok(response.body.into_inner())]))
                        .with_request_id(request_id)
                }
                Err(sdk_error) => Opened::failed(AwsSdkInvokeModelError(sdk_error).into()),
            }
        }))
    }
}

/// Decodes the one `TextToImageResponse` an image request returns.
pub struct ImagesDecoder;

impl Decoder<ImageGeneration, Vec<u8>> for ImagesDecoder {
    type Event = Vec<u8>;

    fn classify(&self, body: Vec<u8>) -> WireEvent<Vec<u8>> {
        wire::classify_typed_event(TypedEvent::Modeled(body))
    }

    fn interpret(&mut self, body: Vec<u8>, out: &mut Out<'_, ImageGeneration>) {
        let decoded = String::from_utf8(body)
            .map_err(|error| ProviderError::Response(error.to_string()))
            .and_then(|body| {
                serde_json::from_str::<TextToImageResponse>(&body)
                    .map_err(|error| ProviderError::Response(error.to_string()))
            })
            .and_then(|response| {
                out.document(serde_json::to_value(&response)?);
                response.normalize(PROVIDER_NAME)
            });
        out.push(decoded);
    }
}
