use crate::client::{AsTranscription, ProviderClient};
use crate::transcription::{
    TranscriptionError, TranscriptionModel, TranscriptionModelDyn, TranscriptionRequest,
    TranscriptionResponse,
};
use std::sync::Arc;

pub trait TranscriptionClient: ProviderClient {
    type TranscriptionModel: TranscriptionModel;
    fn transcription_model(&self, model: &str) -> Self::TranscriptionModel;
}

pub trait TranscriptionClientDyn: ProviderClient {
    fn transcription_model<'a>(&'a self, model: &'a str) -> Box<dyn TranscriptionModelDyn + 'a>;
}

impl<T: TranscriptionClient> TranscriptionClientDyn for T {
    fn transcription_model<'a>(&'a self, model: &'a str) -> Box<dyn TranscriptionModelDyn + 'a> {
        Box::new(self.transcription_model(model))
    }
}

impl<T: TranscriptionClientDyn> AsTranscription for T {
    fn as_transcription(&self) -> Option<Box<&dyn TranscriptionClientDyn>> {
        Some(Box::new(self))
    }
}

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
