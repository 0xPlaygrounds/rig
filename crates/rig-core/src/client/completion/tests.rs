use super::*;
use crate::completion::{CompletionError, CompletionRequest, CompletionResponse};
use crate::streaming::StreamingCompletionResponse;

/// A model implemented entirely outside rig's provider machinery: no
/// response associated types, no client associated type, and no
/// construction hook.
#[derive(Clone)]
struct ExternalModel {
    name: String,
}

impl CompletionModel for ExternalModel {
    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        Err(CompletionError::ResponseError(format!(
            "{} is a compile-coverage model",
            self.name
        )))
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        Err(CompletionError::ResponseError(format!(
            "{} is a compile-coverage model",
            self.name
        )))
    }
}

struct ExternalClient;

impl CompletionClient for ExternalClient {
    type CompletionModel = ExternalModel;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        ExternalModel { name: model.into() }
    }
}

#[test]
fn external_model_needs_no_client_or_response_associated_types() {
    let model = ExternalClient.completion_model("external-model");
    assert_eq!(model.name, "external-model");
}

#[test]
fn external_model_is_usable_without_a_client() {
    // A bare model with no client at all still satisfies `CompletionModel`.
    fn assert_completion_model<M: CompletionModel>(_: &M) {}

    assert_completion_model(&ExternalModel {
        name: "standalone".to_owned(),
    });
}

/// Compile coverage for an out-of-tree provider extension built on the
/// generic [`crate::client::Client`]: implementing the public
/// [`ConstructCompletionModel`] hook is all it takes for the blanket
/// [`CompletionClient`] implementation to apply. Everything here uses only
/// public API, mirroring what a downstream crate can write.
mod external_generic_extension {
    use super::*;
    use crate::client::{
        BearerAuth, Capabilities, Capable, Client, ClientBuilder, DebugExt, Nothing, Provider,
        ProviderBuilder,
    };
    use crate::http_client::{self, HttpClientExt};

    #[derive(Debug, Default, Clone, Copy)]
    struct ExternalExt;
    #[derive(Debug, Default, Clone, Copy)]
    struct ExternalExtBuilder;

    impl Provider for ExternalExt {
        type Builder = ExternalExtBuilder;
        const VERIFY_PATH: &'static str = "/";
    }

    impl ProviderBuilder for ExternalExtBuilder {
        type Extension<H>
            = ExternalExt
        where
            H: HttpClientExt;
        type ApiKey = BearerAuth;

        const BASE_URL: &'static str = "https://external.invalid";

        fn build<H>(
            _builder: &ClientBuilder<Self, Self::ApiKey, H>,
        ) -> http_client::Result<Self::Extension<H>>
        where
            H: HttpClientExt,
        {
            Ok(ExternalExt)
        }
    }

    impl<H> Capabilities<H> for ExternalExt {
        type Completion = Capable<ExternalGenericModel<H>>;
        type Embeddings = Nothing;
        type Transcription = Nothing;
        type ModelListing = Nothing;
        #[cfg(feature = "image")]
        type ImageGeneration = Nothing;
        #[cfg(feature = "audio")]
        type AudioGeneration = Nothing;
        type Rerank = Nothing;
    }

    impl DebugExt for ExternalExt {}

    #[derive(Clone)]
    struct ExternalGenericModel<H> {
        _client: Client<ExternalExt, H>,
        model: String,
    }

    impl<H> CompletionModel for ExternalGenericModel<H>
    where
        H: Clone + Send + Sync + std::fmt::Debug,
    {
        async fn completion(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, CompletionError> {
            Err(CompletionError::ResponseError(format!(
                "{} is a compile-coverage model",
                self.model
            )))
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<StreamingCompletionResponse, CompletionError> {
            Err(CompletionError::ResponseError(format!(
                "{} is a compile-coverage model",
                self.model
            )))
        }
    }

    impl<H> ConstructCompletionModel<Client<ExternalExt, H>> for ExternalGenericModel<H>
    where
        H: Clone + Send + Sync + std::fmt::Debug,
    {
        fn construct(client: &Client<ExternalExt, H>, model: String) -> Self {
            Self {
                _client: client.clone(),
                model,
            }
        }
    }

    #[test]
    fn external_extension_reaches_the_blanket_completion_client_impl() {
        fn assert_completion_client<C: CompletionClient>() {}

        assert_completion_client::<Client<ExternalExt, crate::test_utils::RecordingHttpClient>>();
    }
}
