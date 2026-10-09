//! ChatGPT access-token configuration and native OAuth authentication.
//!
//! ```no_run
//! use rig_core::providers::chatgpt::auth::{AuthSource, Authenticator, DeviceCodeHandler};
//!
//! let auth = Authenticator::new(AuthSource::OAuth, None, DeviceCodeHandler::default(), true);
//! ```

use crate::http_client::HttpClientExt;
use crate::wire::Secret;
use futures::lock::Mutex;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

pub use crate::providers::internal::auth::{DeviceCodeHandler, DeviceCodePrompt};

#[cfg(not(target_family = "wasm"))]
mod native;

/// What a browser sign-in shows the user while it waits.
#[derive(Debug, Clone)]
pub struct BrowserSignInPrompt {
    /// The sign-in page, for the user to open when the browser did not.
    pub authorize_url: String,
    /// Whether the desktop's opener started; the page may still not show.
    pub browser_launched: bool,
}

/// How a sign-in asks the user to authorize: in the browser, or with a
/// code entered on another device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignInMethod {
    /// A page opened in the browser, returning to a local listener.
    Browser,
    /// A code the user enters at a URL, on any device.
    DeviceCode,
}

/// What [`Authenticator::sign_in`] asks the user to do, besides entering a
/// device code, which goes to the authenticator's [`DeviceCodeHandler`].
#[derive(Debug, Clone)]
pub enum SignInPrompt {
    /// Sign in on the page the browser was asked to open.
    Browser(BrowserSignInPrompt),
    /// The browser sign-in could not listen for its callback, for this
    /// reason; a device code follows.
    BrowserUnavailable(String),
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

#[derive(Clone, Debug)]
#[cfg_attr(target_family = "wasm", allow(dead_code))]
pub struct Authenticator {
    source: AuthSource,
    auth_file: Option<PathBuf>,
    device_code_handler: DeviceCodeHandler,
    allow_device_flow: bool,
    /// Held across a cache refresh to prevent concurrent updates.
    refresh_lock: Arc<Mutex<()>>,
}

pub use crate::providers::internal::auth::AuthError;

#[derive(Debug, Clone)]
pub struct AuthContext {
    /// Resolved credential. Use [`Secret::expose`] only when raw bytes are required.
    pub access_token: Secret,
    pub account_id: Option<String>,
}

impl Authenticator {
    pub fn new(
        source: AuthSource,
        auth_file: Option<PathBuf>,
        device_code_handler: DeviceCodeHandler,
        allow_device_flow: bool,
    ) -> Self {
        Self {
            source,
            auth_file,
            device_code_handler,
            allow_device_flow,
            refresh_lock: Arc::default(),
        }
    }

    /// Resolve the access token and account id, refreshing through `http` as needed.
    /// Return cache, transport, or authorization errors. OAuth is unsupported
    /// on WASM; explicit access tokens remain available.
    #[cfg_attr(target_family = "wasm", allow(unused_variables))]
    pub async fn auth_context<H>(&self, http: &H) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        match &self.source {
            AuthSource::AccessToken {
                access_token,
                account_id,
            } => Ok(AuthContext {
                access_token: access_token.clone().into(),
                account_id: account_id.clone(),
            }),
            #[cfg(not(target_family = "wasm"))]
            AuthSource::OAuth => self.auth_context_oauth(http).await,
            #[cfg(target_family = "wasm")]
            AuthSource::OAuth => Err(AuthError::Message(
                "ChatGPT OAuth is not supported on wasm targets".into(),
            )),
        }
    }
}
