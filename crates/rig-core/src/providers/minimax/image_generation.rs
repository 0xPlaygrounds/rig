//! MiniMax's image-generation models and reply.
//!
//! `image-01` takes `width` and `height` from 512 to 2048 pixels in multiples
//! of 8, so set both on the request. An `aspect_ratio` in `additional_params`
//! takes priority over them, and `image-01-live` also needs a
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

use crate::error::ProviderError;
use crate::image_generation::ImageGenerationResponse;

/// `image-01`: text-to-image, and image-to-image with a `subject_reference`.
pub const IMAGE_01: &str = "image-01";
/// `image-01-live`: needs a `subject_reference` image in `additional_params`.
pub const IMAGE_01_LIVE: &str = "image-01-live";

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
    /// A reply without a base64 image, such as one whose every image the
    /// content filter blocked, is an error.
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
