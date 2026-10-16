//! ChatGPT access-token configuration and native OAuth authentication.
//!
//! ```no_run
//! use rig_core::providers::chatgpt::auth::{AuthSource, Authenticator, DeviceCodeHandler, FileTokenStore};
//!
//! let store = FileTokenStore::new("/path/to/credentials");
//! let auth = Authenticator::new(AuthSource::OAuth, store, DeviceCodeHandler::default(), true);
//! ```

use crate::http_client::HttpClientExt;
use std::fmt;

#[cfg(not(target_family = "wasm"))]
pub use crate::providers::internal::auth::FileTokenStore;
pub use crate::providers::internal::auth::{
    DeviceCodeHandler, DeviceCodePrompt, MemoryTokenStore, StoreLock, TokenStore,
};

#[cfg(not(target_family = "wasm"))]
mod native;

/// The key the sign-in is kept under in the store.
#[cfg(not(target_family = "wasm"))]
const STORE_KEY: &str = super::PROVIDER_NAME;

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

/// What [`Authenticator::sign_out`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignedOut {
    /// No sign-in was kept.
    NotSignedIn,
    /// The sign-in was deleted from the store and forgotten. `revoked` says
    /// whether the provider revoked its tokens too, or why not.
    Forgotten {
        /// The revocation, or why it failed.
        revoked: Result<(), String>,
    },
}

/// Where the ChatGPT credential comes from: a given access token, or a stored OAuth sign-in.
#[derive(Clone)]
pub enum AuthSource {
    /// A token the caller already holds.
    AccessToken {
        /// The token.
        access_token: String,
        /// The ChatGPT account it belongs to, when known.
        account_id: Option<String>,
    },
    /// The OAuth sign-in kept in the token store, refreshed as needed.
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

/// Signs in to ChatGPT and keeps the stored credential fresh. Clones share
/// the store and the refresh lock.
#[derive(Clone)]
pub struct Authenticator(Shell<AuthSource>);

impl fmt::Debug for Authenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

use crate::providers::internal::auth::Shell;
pub use crate::providers::internal::auth::{AuthContext, AuthError};

impl Authenticator {
    /// An authenticator over `source`, keeping an OAuth sign-in in `store`
    /// under [`PROVIDER_NAME`](super::PROVIDER_NAME), `chatgpt`;
    /// `device_code_handler` shows a device code when the device flow is
    /// allowed.
    pub fn new(
        source: AuthSource,
        store: impl TokenStore + 'static,
        device_code_handler: DeviceCodeHandler,
        allow_device_flow: bool,
    ) -> Self {
        Self(Shell::new(
            source,
            store,
            device_code_handler,
            allow_device_flow,
        ))
    }

    /// Forget the credential held in memory, so the next request reads the
    /// store: after the store's record is deleted, as a sign-out does.
    pub fn forget(&self) {
        self.0.remembered.forget();
    }

    /// This authenticator showing device codes with `handler`, still
    /// sharing its store and refresh lock with the original.
    #[must_use]
    pub fn with_device_code_handler(mut self, handler: DeviceCodeHandler) -> Self {
        self.0.device_code_handler = handler;
        self
    }

    /// Resolve the access token and account id, refreshing through `http` as needed.
    /// Return cache, transport, or authorization errors. OAuth is unsupported
    /// on WASM; explicit access tokens remain available.
    #[cfg_attr(target_family = "wasm", allow(unused_variables))]
    pub async fn auth_context<H>(&self, http: &H) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        match &self.0.source {
            AuthSource::AccessToken {
                access_token,
                account_id,
            } => Ok(AuthContext {
                api_key: access_token.clone().into(),
                account_id: account_id.clone(),
                api_base: None,
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
