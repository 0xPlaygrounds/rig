use crate::completion::CompletionModel;
use google_cloud_aiplatform_v1 as vertexai;
use rig::client::{CompletionClient, ProviderValue};
use rig::impl_conversion_traits;
use rig::prelude::*;
use std::sync::Arc;
use tokio::sync::OnceCell;

#[derive(Clone, Debug)]
pub struct Client {
    pub(crate) vertex_client: Arc<OnceCell<vertexai::client::PredictionService>>,
}

impl Client {
    pub fn new() -> Self {
        Self {
            vertex_client: Arc::new(OnceCell::new()),
        }
    }

    pub async fn get_inner(&self) -> &vertexai::client::PredictionService {
        self.vertex_client
            .get_or_init(|| async {
                vertexai::client::PredictionService::builder()
                    .build()
                    .await
                    .expect("Failed to build Vertex AI client. Make sure you have Google Cloud credentials configured (e.g., via 'gcloud auth application-default login')")
            })
            .await
    }
}

impl ProviderClient for Client {
    fn from_env() -> Self
    where
        Self: Sized,
    {
        Client::new()
    }

    fn from_val(_: ProviderValue) -> Self
    where
        Self: Sized,
    {
        panic!(
            "Please use `Client::from_env()` instead. Vertex AI uses Application Default Credentials."
        );
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: &str) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl VerifyClient for Client {
    async fn verify(&self) -> Result<(), VerifyError> {
        // No API endpoint to verify credentials - they're validated on first use
        Ok(())
    }
}

impl_conversion_traits!(
    AsTranscription,
    AsEmbeddings,
    AsImageGeneration,
    AsAudioGeneration for Client
);
