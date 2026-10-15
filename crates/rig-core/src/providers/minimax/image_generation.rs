//! MiniMax's image-generation models and reply.
//!
//! For `image-01`, the request's `width` and `height` must be 512 to 2048
//! pixels in multiples of 8, or an `aspect_ratio` in `additional_params`
//! replaces them. `image-01-live` reads no size and needs a
//! `subject_reference` there. Rig asks for base64 and returns the first image.
//!
//! ```no_run
//! use rig_core::image_generation::ImageGenerationRequestBuilder;
//! use rig_core::providers::minimax;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = minimax::from_env()?.image_generation(minimax::IMAGE_01);
//! let request = ImageGenerationRequestBuilder::new("A lighthouse at dawn")
//!     .width(1024)
//!     .height(768)
//!     .build();
//! let image = model.call(request).await?.image;
//! # let _ = image;
//! # Ok(())
//! # }
//! ```

use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::error::{EncodeError, ProviderError};
use crate::image_generation::{ImageGenerationRequest, ImageGenerationResponse};

/// `image-01`: text-to-image, and image-to-image with a `subject_reference`.
pub const IMAGE_01: &str = "image-01";
/// `image-01-live`: needs a `subject_reference` image in `additional_params`.
pub const IMAGE_01_LIVE: &str = "image-01-live";

/// The `POST /v1/image_generation` body for `request`, before its
/// `additional_params`. Only `image-01` reads a size, and an `aspect_ratio`
/// in the parameters replaces it, so the size is sent only then. A size
/// MiniMax would reject, after the parameters' own `width` and `height`, is
/// refused here rather than at the provider.
pub(crate) fn request_body(
    model: &str,
    request: &ImageGenerationRequest,
) -> Result<serde_json::Value, EncodeError> {
    let mut body = serde_json::json!({
        "model": model,
        "prompt": request.prompt,
        // The only form the image wire decodes.
        "response_format": "base64",
    });
    let param = |key: &str| {
        request
            .additional_params
            .as_ref()
            .and_then(|params| params.get(key))
    };
    if model != IMAGE_01 || param("aspect_ratio").is_some() {
        return Ok(body);
    }
    let side =
        |key: &str, set: u32| param(key).map_or(Some(u64::from(set)), |value| value.as_u64());
    let accepted = |side: Option<u64>| {
        side.is_some_and(|side| (512..=2048).contains(&side) && side.is_multiple_of(8))
    };
    if !accepted(side("width", request.width)) || !accepted(side("height", request.height)) {
        return Err(EncodeError::request(format!(
            "MiniMax images are 512 to 2048 pixels wide and high in multiples of 8, not {}x{}; \
             set the width and height, or an `aspect_ratio` in `additional_params`",
            request.width, request.height
        )));
    }
    if let Some(object) = body.as_object_mut() {
        object.insert("width".to_owned(), request.width.into());
        object.insert("height".to_owned(), request.height.into());
    }
    Ok(body)
}

/// MiniMax's `POST /v1/image_generation` reply.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ImageGenerationReply {
    /// MiniMax's trace id.
    #[serde(default)]
    pub id: Option<String>,
    /// The generated images. MiniMax may send `null` here.
    #[serde(default)]
    pub data: Option<ImageGenerationData>,
}

/// The images of an [`ImageGenerationReply`], in the requested format.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ImageGenerationData {
    /// Base64-encoded images, when `response_format` is `base64`.
    #[serde(default)]
    pub image_base64: Vec<String>,
    /// Image URLs, valid for 24 hours, when `response_format` is `url`.
    #[serde(default)]
    pub image_urls: Vec<String>,
}

impl ImageGenerationReply {
    /// The first image, decoded, with the trace id as the response id.
    /// A reply without a base64 image, such as one asked for URLs or one
    /// whose every image the content filter blocked, is an error.
    pub fn into_response(self) -> Result<ImageGenerationResponse, ProviderError> {
        let Some(encoded) = self
            .data
            .and_then(|data| data.image_base64.into_iter().next())
            .filter(|encoded| !encoded.is_empty())
        else {
            return Err(ProviderError::Response(
                "MiniMax returned no base64 image; it sends one only for \
                 `response_format: base64`, and none for an image its content filter blocked"
                    .to_owned(),
            ));
        };
        let image = base64::prelude::BASE64_STANDARD
            .decode(encoded)
            .map_err(|error| ProviderError::Response(error.to_string()))?;
        Ok(ImageGenerationResponse {
            response_id: self.id.filter(|id| !id.is_empty()),
            ..ImageGenerationResponse::new(image)
        })
    }
}
