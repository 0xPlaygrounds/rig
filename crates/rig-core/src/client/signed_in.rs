//! A catalog model served with a signed-in credential that each request
//! reads afresh, so a long session outlives its access token. ChatGPT's and
//! Copilot's authenticators both sign in this way.

use std::future::Future;
use std::sync::{Arc, Mutex};

use crate::catalog::{Catalog, ModelFacts, ModelSpec};
use crate::completion::UnsupportedOption;
use crate::effect::{EffectKind, HandlerDescriptor, family};
use crate::error::{ErrorKind, ErrorReport, ProviderError};
use crate::http_client::DynHttpClient;
use crate::operation::Completion;
use crate::providers::internal::auth::{AuthContext, AuthError};
use crate::providers::registry::{ConnectError, ConnectOptions, ProviderConfig};
use crate::providers::{chatgpt, copilot};
use crate::serve::adapters::ModelAdapter;
use crate::serve::{Dispatch, Reply, Serve};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Where a [`SignedInModel`] reads its credential on every request.
pub trait CredentialSource: WasmCompatSend + WasmCompatSync + 'static {
    /// The credential to send with now, refreshed through `http` when it
    /// has expired.
    fn auth_context(
        &self,
        http: &DynHttpClient,
    ) -> impl Future<Output = Result<AuthContext, AuthError>> + WasmCompatSend;
}

impl CredentialSource for chatgpt::auth::Authenticator {
    fn auth_context(
        &self,
        http: &DynHttpClient,
    ) -> impl Future<Output = Result<AuthContext, AuthError>> + WasmCompatSend {
        chatgpt::auth::Authenticator::auth_context(self, http)
    }
}

impl CredentialSource for copilot::auth::Authenticator {
    fn auth_context(
        &self,
        http: &DynHttpClient,
    ) -> impl Future<Output = Result<AuthContext, AuthError>> + WasmCompatSend {
        copilot::auth::Authenticator::auth_context(self, http)
    }
}

/// A catalog model as an effect handler whose every request carries the
/// credential its [`CredentialSource`] resolves at that moment, refreshed
/// when it has expired: the key, the account it belongs to, and the API root the
/// sign-in names. The model connected with a credential serves every
/// request until the credential changes, so its memos last across turns.
/// ChatGPT also reads its dialect's endpoint and caller identity overrides.
/// Described as [`ModelAdapter`] describes the model, under its
/// [`reference`](ModelSpec::reference).
///
/// ```no_run
/// use rig_core::catalog::Catalog;
/// use rig_core::providers::chatgpt::auth::{AuthSource, Authenticator, DeviceCodeHandler, FileTokenStore};
/// use rig_core::client::signed_in::SignedInModel;
/// use rig_core::serve::ErasedHandler;
///
/// # fn run(http: rig_core::http_client::DynHttpClient) -> Result<(), Box<dyn std::error::Error>> {
/// let spec = Catalog::builtin().resolve("chatgpt/gpt-6.1-sol")?.spec.clone();
/// let auth = Authenticator::new(AuthSource::OAuth, FileTokenStore::new("/path/to/credentials"), DeviceCodeHandler::default(), false);
/// let handler = ErasedHandler::new(SignedInModel::new(spec, auth, http)?);
/// # let _ = handler;
/// # Ok(())
/// # }
/// ```
pub struct SignedInModel<A> {
    spec: ModelSpec,
    credentials: A,
    http: DynHttpClient,
    /// The model connected without a credential: what it is and what it
    /// would leave out, neither of which sends anything.
    unsigned: ModelAdapter<Completion>,
    /// The last credential and the model connected with it.
    connected: Mutex<Option<(AuthContext, Arc<ModelAdapter<Completion>>)>>,
}

impl<A: CredentialSource> SignedInModel<A> {
    /// `spec` served with what `credentials` resolves, through `http`.
    ///
    /// # Errors
    ///
    /// What [`Catalog::connect_with`] reports for `spec`.
    pub fn new(spec: ModelSpec, credentials: A, http: DynHttpClient) -> Result<Self, ConnectError> {
        let unsigned = Catalog::builtin()
            .connect_with(&spec, ConnectOptions::new().api_key("").http(http.clone()))?;
        Ok(Self {
            unsigned: ModelAdapter::new(spec.reference(), unsigned),
            spec,
            credentials,
            http,
            connected: Mutex::default(),
        })
    }

    /// The model connected with the current credential: the one already
    /// connected while the credential is the same.
    async fn model(&self) -> Result<Arc<ModelAdapter<Completion>>, ErrorReport> {
        let context = self
            .credentials
            .auth_context(&self.http)
            .await
            .map_err(failed_sign_in)?;
        if let Ok(connected) = self.connected.lock()
            && let Some((held, model)) = connected.as_ref()
            && *held == context
        {
            return Ok(Arc::clone(model));
        }
        let model = if self.spec.provider.vendor() == chatgpt::PROVIDER_NAME {
            chatgpt::signed_in_config(&context)
                .map(|config| {
                    ProviderConfig::OpenAi(config).completion_model(
                        &self.spec.id,
                        self.http.clone(),
                        ModelFacts::new(self.spec.clone()).with_catalog(Catalog::builtin()),
                    )
                })
                .map_err(ConnectError::from)
        } else {
            let mut options = ConnectOptions::new()
                .api_key(context.api_key.clone())
                .http(self.http.clone());
            if let Some(account_id) = &context.account_id {
                options = options.account_id(account_id);
            }
            if let Some(api_base) = &context.api_base {
                options = options.base_url(api_base);
            }
            Catalog::builtin().connect_with(&self.spec, options)
        }
        .map_err(|error| ErrorReport::new(ErrorKind::Internal, error.to_string()))?;
        let model = Arc::new(ModelAdapter::new(self.spec.reference(), model));
        if let Ok(mut connected) = self.connected.lock() {
            *connected = Some((context, Arc::clone(&model)));
        }
        Ok(model)
    }
}

/// The report of a call whose sign-in gave no credential. An HTTP failure
/// underneath classifies it as it would any provider call, so a dropped
/// connection or a 503 is retried and its `Retry-After` honoured; any other
/// failure is final and asks to sign in again.
pub fn failed_sign_in(error: AuthError) -> ErrorReport {
    let message = format!("the sign-in gave no credential ({error})");
    let (AuthError::Http(source) | AuthError::Reply { source, .. }) = error else {
        return ErrorReport::new(ErrorKind::Provider, format!("{message}; sign in again"));
    };
    let report = ProviderError::from_transport_error(source).report();
    let message = if report.retryable {
        message
    } else {
        format!("{message}; sign in again")
    };
    ErrorReport { message, ..report }
}

impl<A: CredentialSource> Serve for SignedInModel<A> {
    type Family = family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        self.unsigned.descriptor()
    }

    /// What the model leaves out, from the unsigned model: a check is a
    /// dry run, which needs no credential.
    fn check(&self, kind: &EffectKind) -> Vec<UnsupportedOption> {
        self.unsigned.check(kind)
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        match self.model().await {
            Ok(model) => model.serve(kind, dispatch).await,
            Err(report) => Reply::Outcome(Err(report)),
        }
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;
