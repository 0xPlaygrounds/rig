use crate::image_generation::{
    ImageGenerationError, ImageGenerationRequest, ImageGenerationResponse,
};

use super::{Operation, TakeOne};

/// The image generation operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImageGeneration;

impl Operation for ImageGeneration {
    type Request = ImageGenerationRequest;
    type Event = ImageGenerationResponse;
    type Response = ImageGenerationResponse;
    type Error = ImageGenerationError;
    type Capabilities = ();
    type Fold = TakeOne<ImageGenerationResponse, ImageGenerationError>;
}
