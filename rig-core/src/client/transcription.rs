use crate::{
    client::Nothing,
    transcription::{
        TranscriptionError, TranscriptionModel, TranscriptionModelDyn, TranscriptionRequest,
        TranscriptionResponse,
    },
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
    fn transcription_model(
        &self,
        model: <Self::TranscriptionModel as TranscriptionModel>::Models,
    ) -> Self::TranscriptionModel;
}

pub trait TranscriptionClientDyn {
    /// Create a transcription model with the given name.
    fn transcription_model<'a>(&self, model: &str) -> Box<dyn TranscriptionModelDyn + 'a>;
}

impl<M, T> TranscriptionClientDyn for T
where
    T: TranscriptionClient<TranscriptionModel = M>,
    M: TranscriptionModel + 'static,
{
    fn transcription_model<'a>(&self, model: &str) -> Box<dyn TranscriptionModelDyn + 'a> {
        let model = match model.to_string().try_into() {
            Ok(model) => model,
            Err(_) => panic!("Invalid model '{model}'"),
        };

        Box::new(self.transcription_model(model))
    }
}

/// Wraps a TranscriptionModel in a dyn-compatible way for TranscriptionRequestBuilder.
#[derive(Clone)]
pub struct TranscriptionModelHandle<'a> {
    pub inner: Arc<dyn TranscriptionModelDyn + 'a>,
}

impl TranscriptionModel for TranscriptionModelHandle<'_> {
    type Response = ();
    type Client = ();
    type Models = Nothing;

    fn make(_: &Self::Client, _: Self::Models) -> Self {
        panic!("TranscriptionModelHandle::TranscriptionModel::make should not be called")
    }

    async fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse<Self::Response>, TranscriptionError> {
        self.inner.transcription(request).await
    }
}
