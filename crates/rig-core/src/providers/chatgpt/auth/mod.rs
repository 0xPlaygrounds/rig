//! Shared ChatGPT authentication types and target-specific dispatch.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

#[cfg(not(target_family = "wasm"))]
mod native;
#[cfg(target_family = "wasm")]
mod wasm;

#[cfg(not(target_family = "wasm"))]
use native as platform;
#[cfg(target_family = "wasm")]
use wasm as platform;

#[derive(Debug, Clone)]
pub struct DeviceCodePrompt {
    pub verification_uri: String,
    pub user_code: String,
}

/// Where a device-code prompt is delivered.
///
/// This is plain data, not a callback: the auth flow matches on it and the
/// host chooses how the code reaches a human. [`Channel`](Self::Channel)
/// hands the prompt back as an owned event, which is the inversion of the
/// `Arc<dyn Fn(DeviceCodePrompt)>` this replaced — the caller receives the
/// prompt while the flow keeps polling.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum DeviceCodePrompter {
    /// Print sign-in instructions to stdout. The default, matching the
    /// behaviour of the callback-less handler this replaced.
    #[default]
    Stdout,
    /// Emit nothing. For unattended services that surface the prompt some
    /// other way, or that never expect the device flow to trigger.
    Silent,
    /// Send the prompt to the host as an owned event.
    ///
    /// The flow does not wait on the receiver: a full channel or a dropped
    /// receiver is ignored, exactly as a panicking callback used to be the
    /// host's problem rather than the flow's.
    Channel(tokio::sync::mpsc::UnboundedSender<DeviceCodePrompt>),
}

impl DeviceCodePrompter {
    /// Deliver `prompt` according to this prompter.
    pub fn emit(&self, prompt: DeviceCodePrompt) {
        match self {
            Self::Stdout => println!(
                "Sign in with ChatGPT:\n1) Visit {}\n2) Enter code: {}\nDo not share this device code.",
                prompt.verification_uri, prompt.user_code
            ),
            Self::Silent => {}
            Self::Channel(tx) => {
                let _ = tx.send(prompt);
            }
        }
    }
}

#[derive(Clone)]
pub enum AuthSource {
    AccessToken {
        access_token: String,
        account_id: Option<String>,
    },
    OAuth,
}

impl fmt::Debug for AuthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AccessToken { .. } => f.write_str("AccessToken(<redacted>)"),
            Self::OAuth => f.write_str("OAuth"),
        }
    }
}

#[derive(Clone)]
pub struct Authenticator {
    source: AuthSource,
    platform: platform::PlatformAuthenticator,
    state_lock: Arc<Mutex<()>>,
}

impl fmt::Debug for Authenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Authenticator")
            .field("source", &self.source)
            .field("platform", &self.platform)
            .finish()
    }
}

pub use crate::providers::internal::auth::AuthError;

#[derive(Debug, Clone)]
pub struct AuthContext {
    pub access_token: String,
    pub account_id: Option<String>,
}

impl Authenticator {
    pub fn new(
        source: AuthSource,
        auth_file: Option<PathBuf>,
        device_code_prompter: DeviceCodePrompter,
        allow_device_flow: bool,
    ) -> Self {
        Self {
            source,
            platform: platform::PlatformAuthenticator::new(
                auth_file,
                device_code_prompter,
                allow_device_flow,
            ),
            state_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Resolve the credential without any interactive or network step.
    ///
    /// Explicit access tokens resolve directly; OAuth resolves only from an
    /// unexpired cached token file. Returns `None` whenever producing a
    /// usable token would require a refresh or a device-code flow — use
    /// [`Authenticator::auth_context`] for the full flow.
    pub fn cached_auth_context(&self) -> Option<AuthContext> {
        match &self.source {
            AuthSource::AccessToken {
                access_token,
                account_id,
            } => Some(AuthContext {
                access_token: access_token.clone(),
                account_id: account_id.clone(),
            }),
            AuthSource::OAuth => self.platform.cached_auth_context(),
        }
    }

    pub async fn auth_context(&self) -> Result<AuthContext, AuthError> {
        match &self.source {
            AuthSource::AccessToken {
                access_token,
                account_id,
            } => Ok(AuthContext {
                access_token: access_token.clone(),
                account_id: account_id.clone(),
            }),
            AuthSource::OAuth => {
                let _guard = self.state_lock.lock().await;
                self.platform.auth_context_oauth().await
            }
        }
    }
}
