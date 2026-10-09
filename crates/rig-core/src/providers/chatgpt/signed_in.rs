//! A ChatGPT plan model served with a signed-in credential that each
//! request reads afresh, so a long session outlives its access token.

use crate::catalog::{Catalog, ModelSpec};
use crate::driver::DynModel;
use crate::effect::{EffectKind, HandlerDescriptor, family};
use crate::error::{ErrorKind, ErrorReport};
use crate::http_client::DynHttpClient;
use crate::operation::Completion;
use crate::providers::registry::{ConnectError, ConnectOptions};
use crate::serve::adapters::ModelAdapter;
use crate::serve::{Dispatch, Reply, Serve};

use super::auth::Authenticator;

/// A catalog model of the ChatGPT plan as an effect handler whose every
/// request carries the credential its [`Authenticator`] resolves at that
/// moment, refreshed when it has expired. Described as
/// [`ModelAdapter`] describes the model, under its
/// [`reference`](ModelSpec::reference).
///
/// ```no_run
/// use rig_core::catalog::Catalog;
/// use rig_core::providers::chatgpt::{SignedInModel, auth::{AuthSource, Authenticator, DeviceCodeHandler}};
/// use rig_core::serve::ErasedHandler;
///
/// # fn run(http: rig_core::http_client::DynHttpClient) -> Result<(), Box<dyn std::error::Error>> {
/// let spec = Catalog::builtin().resolve("chatgpt/gpt-5.5")?.spec.clone();
/// let auth = Authenticator::new(AuthSource::OAuth, None, DeviceCodeHandler::default(), false);
/// let handler = ErasedHandler::new(SignedInModel::new(spec, auth, http)?);
/// # let _ = handler;
/// # Ok(())
/// # }
/// ```
pub struct SignedInModel {
    spec: ModelSpec,
    authenticator: Authenticator,
    http: DynHttpClient,
    descriptor: HandlerDescriptor,
}

impl SignedInModel {
    /// `spec` served with what `authenticator` resolves, through `http`.
    ///
    /// # Errors
    ///
    /// What [`Catalog::connect_with`] reports for `spec`.
    pub fn new(
        spec: ModelSpec,
        authenticator: Authenticator,
        http: DynHttpClient,
    ) -> Result<Self, ConnectError> {
        // Connected without a credential only for the description.
        let unsigned = Catalog::builtin()
            .connect_with(&spec, ConnectOptions::new().api_key("").http(http.clone()))?;
        let descriptor =
            Serve::descriptor(&ModelAdapter::<Completion>::new(spec.reference(), unsigned));
        Ok(Self {
            spec,
            authenticator,
            http,
            descriptor,
        })
    }

    /// The model, connected with the current credential.
    async fn connect(&self) -> Result<DynModel<Completion>, ErrorReport> {
        let context = self
            .authenticator
            .auth_context(&self.http)
            .await
            .map_err(|error| {
                ErrorReport::new(
                    ErrorKind::Provider,
                    format!("the ChatGPT sign-in gave no credential ({error}); sign in again"),
                )
                .with_retryable(false)
            })?;
        let mut options = ConnectOptions::new()
            .api_key(context.access_token)
            .http(self.http.clone());
        if let Some(account_id) = context.account_id {
            options = options.account_id(account_id);
        }
        Catalog::builtin()
            .connect_with(&self.spec, options)
            .map_err(|error| ErrorReport::new(ErrorKind::Internal, error.to_string()))
    }
}

impl Serve for SignedInModel {
    type Family = family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        self.descriptor.clone()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        match self.connect().await {
            Ok(model) => {
                ModelAdapter::<Completion>::new(self.spec.reference(), model)
                    .serve(kind, dispatch)
                    .await
            }
            Err(report) => Reply::Outcome(Err(report)),
        }
    }
}
