#[allow(deprecated)]
use crate::transcription::TranscriptionModelDyn;
use crate::transcription::{
    TranscriptionError, TranscriptionModel, TranscriptionRequest, TranscriptionResponse,
};
use std::sync::Arc;

/// A provider client with transcription capabilities.
/// Clone is required for conversions between client types.
pub trait TranscriptionClient {
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
    fn transcription_model(&self, model: impl Into<String>) -> Self::TranscriptionModel;
}

#[allow(deprecated)]
#[deprecated(
    since = "0.25.0",
    note = "`DynClientBuilder` and related features have been deprecated and will be removed in a future release. In this case, use `TranscriptionClient` instead."
)]
pub trait TranscriptionClientDyn {
    /// Create a transcription model with the given name.
    fn transcription_model<'a>(&self, model: &str) -> Box<dyn TranscriptionModelDyn + 'a>;
}

#[allow(deprecated)]
impl<M, T> TranscriptionClientDyn for T
where
    T: TranscriptionClient<TranscriptionModel = M>,
    M: TranscriptionModel + 'static,
{
    fn transcription_model<'a>(&self, model: &str) -> Box<dyn TranscriptionModelDyn + 'a> {
        Box::new(self.transcription_model(model))
    }
}

#[allow(deprecated)]
#[deprecated(
    since = "0.25.0",
    note = "`DynClientBuilder` and related features have been deprecated and will be removed in a future release."
)]
/// Wraps a TranscriptionModel in a dyn-compatible way for TranscriptionRequestBuilder.
#[derive(Clone)]
pub struct TranscriptionModelHandle<'a> {
    pub inner: Arc<dyn TranscriptionModelDyn + 'a>,
}

#[allow(deprecated)]
impl TranscriptionModel for TranscriptionModelHandle<'_> {
    type Response = ();
    type Client = ();

    /// **PANICS**: We are deprecating DynClientBuilder and related functionality, during this
    /// transition some methods will be invalid, like this one
    fn make(_: &Self::Client, _: impl Into<String>) -> Self {
        panic!(
            "Invalid method: Cannot make a TranscriptionModelHandle from a client + model identifier"
        )
    }

    async fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse<Self::Response>, TranscriptionError> {
        self.inner.transcription(request).await
    }
}
