use crate::completion::CompletionModel;

/// A provider client with completion capabilities.
///
/// Clients remain `Clone` for conversions between client types; the models
/// they construct no longer need to be.
pub trait CompletionClient {
    /// The type of CompletionModel used by the client.
    type CompletionModel: CompletionModel;

    /// Create a completion model with the given model.
    ///
    /// Construction lives here rather than on [`CompletionModel`] so a model
    /// type can be implemented — and used — without any client type at all.
    /// Implement this by calling the model's own inherent constructor.
    ///
    /// # Example with OpenAI
    /// ```no_run
    /// use rig_core::prelude::*;
    /// use rig_core::providers::openai::{Client, self};
    ///
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key")?;
    ///
    /// let gpt = openai.completion_model(openai::GPT_5_2);
    /// # Ok(())
    /// # }
    /// ```
    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel;
}

/// Construction hook for the blanket [`CompletionClient`] implementation over
/// [`crate::client::Client`].
///
/// That blanket implementation is generic over whichever model type a provider
/// extension declares, so it needs some way to build that model. Coherence
/// rules out one blanket implementation per provider family — they would all
/// overlap on `Client<Ext, H>` — and the alternative of a public bound such as
/// `From<(Client<Ext, H>, String)>` would push a synthetic conversion into
/// every provider model's public API.
///
/// This trait is public because it is the extension point for out-of-tree
/// provider extensions built on the generic [`crate::client::Client`]: such a
/// crate cannot implement [`CompletionClient`] for rig's foreign
/// `Client<Ext, H>` type directly (orphan rule), so it implements this trait
/// on its own model type instead, and the blanket implementation supplies
/// `completion_model` for it. Providers with their own client type simply
/// implement [`CompletionClient`] directly and never need this trait.
pub trait ConstructCompletionModel<C>: Sized {
    /// Build this model from its provider client and a model identifier.
    fn construct(client: &C, model: String) -> Self;
}

#[cfg(test)]
mod tests {
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
            H: Clone + Send + Sync + std::fmt::Debug + 'static,
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
            H: Clone + Send + Sync + std::fmt::Debug + 'static,
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

            assert_completion_client::<Client<ExternalExt, reqwest::Client>>();
        }
    }
}
