use crate::audio_generation::{
    AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};

use super::{Operation, TakeOne};

/// The audio generation operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AudioGeneration;

impl Operation for AudioGeneration {
    type Request = AudioGenerationRequest;
    type Event = AudioGenerationResponse;
    type Response = AudioGenerationResponse;
    type Error = AudioGenerationError;
    type Capabilities = ();
    type Fold = TakeOne<AudioGenerationResponse, AudioGenerationError>;
}
