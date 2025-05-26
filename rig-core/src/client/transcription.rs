use crate::client::{AsTranscription, ProviderClient};
use crate::transcription::{
    TranscriptionError, TranscriptionModel, TranscriptionModelDyn, TranscriptionRequest,
    TranscriptionResponse,
};
use std::sync::Arc;

/// A provider client with transcription capabilities.
/// Clone is required for conversions between client types.
pub trait TranscriptionClient: ProviderClient + Clone {
    /// The type of TranscriptionModel used by the Client
    type TranscriptionModel: TranscriptionModel;

    /// Create a transcription model with the given name.
    ///
    /// # Example with OpenAI
    /// ```
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let whisper = openai.transcription_model(openai::WHISPER_1);
    /// ```
    fn transcription_model(&self, model: &str) -> Self::TranscriptionModel;
}

pub trait TranscriptionClientDyn: ProviderClient {
    /// Create a transcription model with the given name.
    fn transcription_model<'a>(&self, model: &str) -> Box<dyn TranscriptionModelDyn + 'a>;
}

impl<T: TranscriptionClient<TranscriptionModel = M>, M: TranscriptionModel + 'static>
    TranscriptionClientDyn for T
{
    fn transcription_model<'a>(&self, model: &str) -> Box<dyn TranscriptionModelDyn + 'a> {
        Box::new(self.transcription_model(model))
    }
}

impl<T: TranscriptionClientDyn + Clone + 'static> AsTranscription for T {
    fn as_transcription(&self) -> Option<Box<dyn TranscriptionClientDyn>> {
        Some(Box::new(self.clone()))
    }
}

/// Wraps a TranscriptionModel in a dyn-compatible way for TranscriptionRequestBuilder.
#[derive(Clone)]
pub struct TranscriptionModelHandle<'a> {
    pub inner: Arc<dyn TranscriptionModelDyn + 'a>,
}

impl TranscriptionModel for TranscriptionModelHandle<'_> {
    type Response = ();

    async fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse<Self::Response>, TranscriptionError> {
        self.inner.transcription(request).await
    }
}
