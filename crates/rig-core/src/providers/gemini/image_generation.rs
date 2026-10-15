//! Image generation through Gemini's `generateContent` endpoint. The
//! request's width and height pick the aspect ratio, its images are
//! references the model draws from, and [`ImageOptions`] sets the size and
//! asks for text beside the images. [`parts_of`] reads every image and text
//! part of a reply.
//!
//! ```no_run
//! use rig_core::image_generation::ImageGenerationRequestBuilder;
//! use rig_core::providers::gemini::{Gemini, ImageSize};
//! use rig_core::providers::gemini::image_generation::{GEMINI_3_1_FLASH_IMAGE, ImageOptions};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Gemini::from_env()?.image_generation(GEMINI_3_1_FLASH_IMAGE);
//! let request = ImageGenerationRequestBuilder::new("A banana in a tuxedo")
//!     .width(16)
//!     .height(9)
//!     .additional_params(ImageOptions::new().image_size(ImageSize::TwoK).params())
//!     .build();
//! let image = model.call(request).await?.image;
//! # let _ = image;
//! # Ok(())
//! # }
//! ```

use super::ImageSize;
use super::completion::usage_of;
use crate::error::{EncodeError, ProviderError};
use crate::image_generation;
use crate::image_generation::ImageGenerationRequest;
use crate::json_utils::Lenient;
use crate::message::{DocumentSourceKind, Image};
use crate::operation::ImageGeneration;
use crate::wire::{Body, Descriptor, DocumentDecoder, Encoded, Framing, Mode, Wire};
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde_json::{Value, json};

/// `gemini-nano-banana-2.1`, Nano Banana 2.1: the general image model.
pub const GEMINI_NANO_BANANA_2_1: &str = "gemini-nano-banana-2.1";
/// `gemini-3.1-flash-image`, Nano Banana 2.
pub const GEMINI_3_1_FLASH_IMAGE: &str = "gemini-3.1-flash-image";
/// `gemini-3.1-flash-lite-image`, Nano Banana 2 Lite: 1K images only.
pub const GEMINI_3_1_FLASH_LITE_IMAGE: &str = "gemini-3.1-flash-lite-image";
/// `gemini-3-pro-image`, Nano Banana Pro.
pub const GEMINI_3_PRO_IMAGE: &str = "gemini-3-pro-image";
/// `gemini-2.5-flash-image`, Nano Banana.
pub const GEMINI_2_5_FLASH_IMAGE: &str = "gemini-2.5-flash-image";

/// The aspect ratios the API takes, as `(width, height)`.
const ASPECT_RATIOS: [(u32, u32); 14] = [
    (1, 1),
    (1, 4),
    (4, 1),
    (1, 8),
    (8, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (4, 5),
    (5, 4),
    (9, 16),
    (16, 9),
    (21, 9),
];

/// Gemini's image options, sent as the request's `additional_params`
/// through [`params`](Self::params).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_size: Option<ImageSize>,
    #[serde(skip)]
    text: bool,
}

impl ImageOptions {
    /// No option set: the aspect ratio follows the request's width and
    /// height, the size is the model's default, and the reply is images
    /// only.
    pub fn new() -> Self {
        Self::default()
    }

    /// `imageConfig.aspectRatio`, such as `"21:9"`, in place of the one the
    /// width and height pick.
    pub fn aspect_ratio(mut self, ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(ratio.into());
        self
    }

    /// `imageConfig.imageSize`.
    pub fn image_size(mut self, size: ImageSize) -> Self {
        self.image_size = Some(size);
        self
    }

    /// Ask for text parts beside the images, which [`parts_of`] reads.
    pub fn with_text(mut self) -> Self {
        self.text = true;
        self
    }

    /// The options as `additional_params`: a `generationConfig` the wire
    /// merges over its own.
    pub fn params(self) -> Value {
        let modalities = self.text.then(|| json!(["TEXT", "IMAGE"]));
        let mut config = json!({ "imageConfig": self });
        if let (Some(modalities), Some(config)) = (modalities, config.as_object_mut()) {
            config.insert("responseModalities".to_owned(), modalities);
        }
        json!({ "generationConfig": config })
    }
}

/// One part of an image reply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImagePart {
    /// Text the model wrote.
    Text(String),
    /// An image, decoded from base64.
    Image {
        /// Its media type, such as `image/png`.
        mime_type: String,
        /// Its bytes.
        data: Vec<u8>,
    },
}

/// Every text and image part of the first candidate in `reply`, in reply
/// order. Thoughts, interim images included, and images without data are
/// left out.
///
/// # Errors
///
/// When an image's data is not base64.
pub fn parts_of(reply: &Value) -> Result<Vec<ImagePart>, ProviderError> {
    let parts = reply
        .arr("candidates")
        .first()
        .and_then(|candidate| candidate.get("content"))
        .map(|content| content.arr("parts"))
        .unwrap_or_default();
    let mut out = Vec::new();
    for part in parts
        .iter()
        .filter(|part| part.bool("thought") != Some(true))
    {
        if let Some(text) = part.str("text") {
            out.push(ImagePart::Text(text.to_owned()));
        }
        let Some(blob) = part.get("inlineData") else {
            continue;
        };
        let mime_type = blob.str("mimeType").unwrap_or_default();
        let Some(data) = blob.str("data").filter(|_| mime_type.starts_with("image/")) else {
            continue;
        };
        let data = BASE64_STANDARD.decode(data).map_err(|err| {
            ProviderError::Response(format!("Gemini image data was not valid base64: {err}"))
        })?;
        out.push(ImagePart::Image {
            mime_type: mime_type.to_owned(),
            data,
        });
    }
    Ok(out)
}

/// The first image in `reply`, with its usage and identity.
///
/// # Errors
///
/// When `reply` holds no image, or data that is not base64.
pub fn image_of(reply: &Value) -> Result<image_generation::ImageGenerationResponse, ProviderError> {
    let image = parts_of(reply)?
        .into_iter()
        .find_map(|part| match part {
            ImagePart::Image { data, .. } => Some(data),
            ImagePart::Text(_) => None,
        })
        .ok_or_else(|| {
            ProviderError::Response(
                "Gemini image generation response did not include image data".into(),
            )
        })?;
    Ok(image_generation::ImageGenerationResponse {
        model: reply.str("modelVersion").map(str::to_owned),
        response_id: Some(reply.str("responseId").unwrap_or_default().to_owned()),
        usage: reply.get("usageMetadata").map(usage_of).unwrap_or_default(),
        ..image_generation::ImageGenerationResponse::new(image)
    })
}

fn create_request_body(request: ImageGenerationRequest) -> Result<Value, EncodeError> {
    let mut parts = vec![json!({ "text": request.prompt })];
    for image in request.images {
        parts.push(reference(image)?);
    }
    let mut image_config = serde_json::Map::new();
    if let Some(ratio) = aspect_ratio(request.width, request.height) {
        image_config.insert("aspectRatio".to_owned(), json!(ratio));
    }
    let mut body = json!({
        "contents": [{ "role": "user", "parts": parts }],
        "toolConfig": null,
        "generationConfig": { "responseModalities": ["IMAGE"], "imageConfig": image_config },
        "safetySettings": null,
        "systemInstruction": null,
    });
    match (&mut body, request.additional_params) {
        (Value::Object(body), Some(Value::Object(params))) => {
            crate::completion::options::deep_merge(body, params);
        }
        (body, Some(params)) => *body = params,
        (_, None) => {}
    }
    Ok(body)
}

/// A reference image as a part: a URL by reference, and data inline.
fn reference(image: Image) -> Result<Value, EncodeError> {
    let source = match image.data {
        DocumentSourceKind::Raw(bytes) => DocumentSourceKind::Base64(BASE64_STANDARD.encode(bytes)),
        source => source,
    };
    super::completion::media(super::completion::mime(image.media_type), &source, false)
}

/// The aspect ratio the API takes that `width` by `height` has; `None` for
/// a zero side or a ratio it does not take, which leaves the choice to the
/// model.
fn aspect_ratio(width: u32, height: u32) -> Option<String> {
    let (width, height) = (u64::from(width), u64::from(height));
    ASPECT_RATIOS
        .iter()
        .find(|&&(w, h)| width > 0 && width * u64::from(h) == height * u64::from(w))
        .map(|(w, h)| format!("{w}:{h}"))
}

/// The image generation wire: `POST /v1beta/models/{model}:generateContent`.
///
/// Both [`Mode`]s request image output and decode a whole response document.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Images {
    /// The provider this wire speaks to.
    pub provider: super::GeminiConfig,
    /// Name of the model, for example [`GEMINI_3_1_FLASH_IMAGE`].
    pub model: String,
}

impl Images {
    /// The image generation wire for `model`.
    pub fn new(provider: super::GeminiConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

impl Wire for Images {
    type Op = ImageGeneration;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Event = Value;
    type Decoder<'id> = DocumentDecoder<ImageGeneration, Value>;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME).model(self.model.as_str())
    }

    fn encode(&self, request: ImageGenerationRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let body = serde_json::to_vec(&create_request_body(request)?)?;
        let request = http::Request::post(format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.provider.base_url,
            self.model,
            self.provider.api_key.expose()
        ))
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::Bytes(body))?;
        // Gemini reports no transport request-id header.
        Ok(Encoded::new(request, Framing::Whole))
    }

    /// The first non-thought image; missing image data and invalid base64
    /// are response errors.
    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        DocumentDecoder::new(super::completion::classify_reply, |reply| image_of(&reply))
    }
}

impl super::GeminiConfig {
    /// The image generation wire.
    pub(crate) fn image_generation(&self, model: impl Into<String>) -> Images {
        Images::new(self.clone(), model)
    }
}

#[cfg(test)]
mod tests;
