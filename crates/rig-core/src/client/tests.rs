use super::*;
use crate::completion::{CompletionError, CompletionModel, CompletionRequest, CompletionResponse};
use crate::markers::Missing;
use crate::providers::anthropic;
use crate::streaming::StreamingCompletionResponse;
use crate::test_utils::RecordingHttpClient;

/// Type-level test that `Client::builder()` methods do not require annotation to determine
/// backig HTTP client
#[test]
fn ensures_client_builder_no_annotation() {
    let http_client = RecordingHttpClient::new("");
    let _ = anthropic::Client::builder()
        .http_client(http_client)
        .api_key("Foo")
        .build()
        .unwrap();
}

// An out-of-tree provider, written against public API only: the `Provider`
// impl plus one `HasCompletion` impl is the whole plumbing between a wire
// implementation and `client.completion_model(..)`.
#[derive(Debug, Clone)]
struct External;

impl Provider for External {
    const NAME: &'static str = "external";
    const BASE_URL: &'static str = "https://external.invalid";
    const VERIFY_PATH: &'static str = "/";
    type ApiKey = BearerAuth;
    type Config = ();
    type EnvInput = String;

    fn build(_: (), _: &BearerAuth) -> http_client::Result<Self> {
        Ok(External)
    }
    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<Self, H>> {
        Client::from_env_api_key("EXTERNAL_API_KEY", None, http)
    }
    fn from_val<H: HttpClientExt>(key: String, http: H) -> ProviderClientResult<Client<Self, H>> {
        Client::new_with(key, http)
    }
}

struct ExternalModel<H> {
    client: Client<External, H>,
    model: String,
}

impl<H: ModelTransport> CompletionModel for ExternalModel<H> {
    async fn completion(
        &self,
        _: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        Err(CompletionError::ProviderError(self.model.clone()))
    }
    async fn stream(
        &self,
        _: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        Err(CompletionError::ProviderError(self.model.clone()))
    }
}

impl HasCompletion for External {
    type Model<H>
        = ExternalModel<H>
    where
        H: ModelTransport;
    fn completion_model<H: ModelTransport>(
        client: &Client<Self, H>,
        model: String,
    ) -> ExternalModel<H> {
        ExternalModel {
            client: client.clone(),
            model,
        }
    }
}

/// The blanket `CompletionClient` impl names the provider's concrete model
/// type — never an erased handle — which is what adapters wrapping
/// `client.completion_model(..)` rely on.
const _: () = {
    const fn same<T>(_: std::marker::PhantomData<T>, _: std::marker::PhantomData<T>) {}
    same(
        std::marker::PhantomData::<
            <Client<External, RecordingHttpClient> as CompletionClient>::CompletionModel,
        >,
        std::marker::PhantomData::<ExternalModel<RecordingHttpClient>>,
    );
};

#[test]
fn out_of_tree_provider_builds_a_client_and_a_model() {
    let client = Client::<External, Missing>::builder()
        .api_key("k")
        .http_client(RecordingHttpClient::new(""))
        .build()
        .expect("client should build");
    let model = client.completion_model("m");
    assert_eq!(model.model, "m");
    assert_eq!(model.client.base_url(), External::BASE_URL);
    assert!(
        model
            .client
            .headers()
            .contains_key(http::header::AUTHORIZATION)
    );
}

#[test]
fn missing_api_key_is_a_build_error() {
    let result = Client::<External, Missing>::builder()
        .http_client(RecordingHttpClient::new(""))
        .build();
    assert!(matches!(
        result,
        Err(ProviderClientError::MissingApiKey("external"))
    ));
}

#[test]
fn nothing_keyed_provider_builds_without_a_key() {
    let client = crate::providers::ollama::Client::builder()
        .http_client(RecordingHttpClient::new(""))
        .build()
        .expect("ollama needs no key");
    assert!(!client.headers().contains_key(http::header::AUTHORIZATION));
}
