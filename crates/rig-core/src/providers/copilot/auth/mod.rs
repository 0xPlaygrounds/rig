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
                "Sign in with GitHub Copilot:\n1) Visit {}\n2) Enter code: {}",
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
    ApiKey(String),
    GitHubAccessToken(String),
    OAuth,
}

impl fmt::Debug for AuthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApiKey(_) => f.write_str("ApiKey(<redacted>)"),
            Self::GitHubAccessToken(_) => f.write_str("GitHubAccessToken(<redacted>)"),
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
    pub api_key: String,
    pub api_base: Option<String>,
}

impl Authenticator {
    pub fn new(
        source: AuthSource,
        access_token_file: Option<PathBuf>,
        api_key_file: Option<PathBuf>,
        device_code_prompter: DeviceCodePrompter,
        allow_device_flow: bool,
    ) -> Self {
        Self {
            source,
            platform: platform::PlatformAuthenticator::new(
                access_token_file,
                api_key_file,
                device_code_prompter,
                allow_device_flow,
            ),
            state_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Resolve the credential without any interactive or network step.
    ///
    /// Explicit API keys resolve directly; GitHub access tokens and OAuth
    /// resolve only from an unexpired cached Copilot key file (for access
    /// tokens, one bound to that token). Returns `None` whenever producing a
    /// usable key would require an exchange, refresh, or device-code flow —
    /// use [`Authenticator::auth_context`] for the full flow.
    pub fn cached_auth_context(&self) -> Option<AuthContext> {
        match &self.source {
            AuthSource::ApiKey(api_key) => Some(AuthContext {
                api_key: api_key.clone(),
                api_base: None,
            }),
            AuthSource::GitHubAccessToken(access_token) => {
                self.platform.cached_auth_context(Some(access_token))
            }
            AuthSource::OAuth => self.platform.cached_auth_context(None),
        }
    }

    pub async fn auth_context(&self) -> Result<AuthContext, AuthError> {
        match &self.source {
            AuthSource::ApiKey(api_key) => Ok(AuthContext {
                api_key: api_key.clone(),
                api_base: None,
            }),
            AuthSource::GitHubAccessToken(access_token) => {
                let _guard = self.state_lock.lock().await;
                self.platform
                    .auth_context_with_github_access_token(access_token)
                    .await
            }
            AuthSource::OAuth => {
                let _guard = self.state_lock.lock().await;
                self.platform.auth_context_oauth().await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DeviceCodePrompt, DeviceCodePrompter};

    #[test]
    fn channel_prompter_hands_the_prompt_back_as_data() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let prompter = DeviceCodePrompter::Channel(tx);

        prompter.emit(DeviceCodePrompt {
            verification_uri: "https://github.com/login/device".to_string(),
            user_code: "ABCD-1234".to_string(),
        });

        let received = rx.try_recv().expect("prompt delivered");
        assert_eq!(received.verification_uri, "https://github.com/login/device");
        assert_eq!(received.user_code, "ABCD-1234");
    }

    #[test]
    fn a_dropped_receiver_does_not_break_the_flow() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        drop(rx);
        DeviceCodePrompter::Channel(tx).emit(DeviceCodePrompt {
            verification_uri: "https://github.com/login/device".to_string(),
            user_code: "ABCD-1234".to_string(),
        });
    }

    #[test]
    fn silent_prompter_emits_nothing() {
        DeviceCodePrompter::Silent.emit(DeviceCodePrompt {
            verification_uri: "https://github.com/login/device".to_string(),
            user_code: "ABCD-1234".to_string(),
        });
    }

    #[test]
    fn the_default_prompter_is_stdout() {
        assert!(matches!(
            DeviceCodePrompter::default(),
            DeviceCodePrompter::Stdout
        ));
    }
}
