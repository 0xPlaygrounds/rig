use crate::transcription::{TranscriptionError, TranscriptionRequest, TranscriptionResponse};

use super::{Operation, TakeOne};

/// The audio transcription operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Transcription;

impl Operation for Transcription {
    type Request = TranscriptionRequest;
    type Event = TranscriptionResponse;
    type Response = TranscriptionResponse;
    type Error = TranscriptionError;
    type Capabilities = ();
    type Fold = TakeOne<TranscriptionResponse, TranscriptionError>;
}
