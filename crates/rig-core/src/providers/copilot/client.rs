//! The Copilot client: a [`CopilotConfig`] on a transport, and the models it
//! builds.

use crate::driver::Model;
use crate::error::ProviderError;
use crate::model::ModelList;
use crate::providers::internal::client::http_client;

use super::auth::{AuthError, Authenticator};
use super::wire::{CopilotConfig, CopilotWire, Embeddings};

http_client!(
    /// GitHub Copilot: its [`CopilotConfig`] on a transport. Every model it
    /// builds sends through that transport.
    Copilot,
    CopilotConfig
);

impl Copilot {
    /// Copilot with an exchanged session token, on the shared reqwest
    /// client. See [`CopilotConfig::new`] for how the endpoint is chosen.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn new(api_key: impl Into<crate::wire::Secret>) -> Self {
        CopilotConfig::new(api_key).client()
    }

    /// Copilot from the environment ([`CopilotConfig::from_env`]), on the
    /// shared reqwest client. This does not exchange a token.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn from_env() -> Result<Self, crate::client::env::EnvError> {
        Ok(CopilotConfig::from_env()?.client())
    }

    /// The completion model for `model`, on whichever route answers it.
    pub fn completion(&self, model: impl Into<String>) -> Model<CopilotWire> {
        self.model(self.config.completion(model))
    }

    /// The embedding model for `model`, `ndims` wide when set.
    pub fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Model<Embeddings> {
        self.model(self.config.embedding(model, ndims))
    }

    /// The models this session can use.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        self.model(self.config.models()).call(()).await
    }

    /// A client on this client's transport, configured with the session
    /// `authenticator` resolves: the device login, the GitHub token
    /// exchange and the refresh all send through this client's transport.
    /// The configuration is [`CopilotConfig::from_auth`]'s.
    ///
    /// ```no_run
    /// use rig_core::providers::copilot::{CopilotConfig, auth::{AuthSource, Authenticator, DeviceCodeHandler}};
    ///
    /// # async fn run(http: rig_core::http_client::DynHttpClient) -> Result<(), Box<dyn std::error::Error>> {
    /// let authenticator = Authenticator::new(AuthSource::OAuth, None, None, DeviceCodeHandler::default(), true);
    /// let copilot = CopilotConfig::new("").connect(http).authenticate(&authenticator).await?;
    /// # let _ = copilot;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn authenticate(self, authenticator: &Authenticator) -> Result<Self, AuthError> {
        let context = authenticator.auth_context(&self.http).await?;
        Ok(Self {
            config: CopilotConfig::from_auth(&context),
            http: self.http,
        })
    }
}
