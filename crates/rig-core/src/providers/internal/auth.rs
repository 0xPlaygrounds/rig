//! Shared OAuth errors, device-code callbacks, token stores, and
//! authentication transport helpers.
//!
//! ```
//! use rig_core::providers::chatgpt::auth::DeviceCodeHandler;
//! let handler = DeviceCodeHandler::new(|prompt| {
//!     println!("Visit {} and enter {}", prompt.verification_uri, prompt.user_code);
//! });
//! ```

use crate::http_client::{self, HttpClientExt};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::sync::Arc;

/// Device authorization details surfaced to a provider callback.
#[derive(Debug, Clone)]
pub struct DeviceCodePrompt {
    /// URL where the user authorizes the device.
    pub verification_uri: String,
    /// Short code the user enters at the verification URL.
    pub user_code: String,
}

/// Device-code callback with thread-safety bounds outside browser WASM.
/// Browser WASM authenticators do not invoke it.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) type DeviceCodeCallback = dyn Fn(DeviceCodePrompt) + Send + Sync;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub(crate) type DeviceCodeCallback = dyn Fn(DeviceCodePrompt);

/// Optional callback invoked when an OAuth device flow needs user action.
#[derive(Clone, Default)]
pub struct DeviceCodeHandler(pub(crate) Option<Arc<DeviceCodeCallback>>);

impl DeviceCodeHandler {
    /// Wraps a device-code callback.
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(DeviceCodePrompt) + WasmCompatSend + WasmCompatSync + 'static,
    {
        Self(Some(Arc::new(handler)))
    }
}

impl DeviceCodeHandler {
    /// Show `prompt` with the callback, or print `fallback_message` to
    /// stdout when there is none.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn show(&self, prompt: DeviceCodePrompt, fallback_message: &str) {
        match &self.0 {
            Some(callback) => callback(prompt),
            None => println!("{fallback_message}"),
        }
    }
}

impl std::fmt::Debug for DeviceCodeHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_some() {
            f.write_str("DeviceCodeHandler(<callback>)")
        } else {
            f.write_str("DeviceCodeHandler(None)")
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// The HTTP transport failed. Non-success responses arrive as the
    /// status-bearing [`http_client::Error`] variants (so the status is still
    /// inspectable); response-less failures as
    /// [`http_client::Error::Transient`] or [`http_client::Error::Permanent`].
    #[error(transparent)]
    Http(#[from] http_client::Error),
    /// The sign-in server answered with a failure: `message` says what it
    /// said, and `source`, its reply, says whether asking again may work.
    #[error("{message}")]
    Reply {
        /// What the failure means for the sign-in.
        message: String,
        /// The non-success reply.
        #[source]
        source: http_client::Error,
    },
    /// The [`TokenStore`] could not load or save a credential.
    #[error("token store failed: {0}")]
    Store(#[source] crate::error::BoxError),
}

/// The credential a signed-in request is sent with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthContext {
    /// The resolved key. Use [`Secret::expose`](crate::wire::Secret::expose)
    /// only when raw bytes are required.
    pub api_key: crate::wire::Secret,
    /// The account the key belongs to, when the provider names one
    /// (ChatGPT's `ChatGPT-Account-Id`).
    pub account_id: Option<String>,
    /// The API root the sign-in names, when it names one (Copilot's
    /// session endpoint).
    pub api_base: Option<String>,
}

/// What every provider's authenticator holds: where its credential comes
/// from, the store that keeps it, how a device code is shown, whether a
/// device flow may start, the lock a refresh holds and the credential last
/// resolved. Clones share the store, the lock and the remembered credential.
#[derive(Clone)]
#[cfg_attr(target_family = "wasm", allow(dead_code))]
pub(crate) struct Shell<S> {
    pub(crate) source: S,
    pub(crate) store: Arc<dyn TokenStore>,
    pub(crate) device_code_handler: DeviceCodeHandler,
    pub(crate) allow_device_flow: bool,
    /// Held across a cache refresh to prevent concurrent updates.
    pub(crate) refresh_lock: Arc<futures::lock::Mutex<()>>,
    pub(crate) remembered: Remembered,
}

impl<S> Shell<S> {
    pub(crate) fn new(
        source: S,
        store: impl TokenStore + 'static,
        device_code_handler: DeviceCodeHandler,
        allow_device_flow: bool,
    ) -> Self {
        Self {
            source,
            store: Arc::new(store),
            device_code_handler,
            allow_device_flow,
            refresh_lock: Arc::default(),
            remembered: Remembered::default(),
        }
    }
}

impl<S> Shell<S> {
    /// Holds the record under `key` for a refresh, a sign-in or a sign-out:
    /// the refresh lock its clones share, then the store's lock on `key`
    /// ([`TokenStore::lock`]), so one runs at a time across clones and
    /// processes, and each reads what the last one saved.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) async fn hold(&self, key: &str) -> Result<Held<'_>, AuthError> {
        let refresh = self.refresh_lock.lock().await;
        let store = self.store.lock(key).await.map_err(AuthError::Store)?;
        Ok(Held {
            _refresh: refresh,
            _store: store,
        })
    }
}

/// What [`Shell::hold`] holds, let go when dropped: the store's lock first.
#[cfg_attr(target_family = "wasm", allow(dead_code))]
pub(crate) struct Held<'a> {
    _store: StoreLock,
    _refresh: futures::lock::MutexGuard<'a, ()>,
}

impl<S: std::fmt::Debug> std::fmt::Debug for Shell<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Authenticator")
            .field("source", &self.source)
            .field("device_code_handler", &self.device_code_handler)
            .field("allow_device_flow", &self.allow_device_flow)
            .finish_non_exhaustive()
    }
}

/// The last credential an authenticator resolved and its expiry, shared by
/// its clones: a live one is served without the store, its lock or a parse.
#[derive(Clone, Default)]
pub(crate) struct Remembered(Arc<std::sync::Mutex<Option<(AuthContext, i64)>>>);

impl Remembered {
    /// The credential while it is live, `skew_seconds` before its expiry.
    pub(crate) fn live(&self, skew_seconds: i64) -> Option<AuthContext> {
        let held = self.0.lock().ok()?;
        let (context, expires_at) = held.as_ref()?;
        (!token_expired(Some(*expires_at), skew_seconds)).then(|| context.clone())
    }

    /// Remember `context` until `expires_at`; one with no expiry is not
    /// remembered.
    pub(crate) fn keep(&self, context: &AuthContext, expires_at: Option<i64>) {
        if let Ok(mut held) = self.0.lock() {
            *held = expires_at.map(|expires_at| (context.clone(), expires_at));
        }
    }

    /// Forget the remembered credential.
    pub(crate) fn forget(&self) {
        if let Ok(mut held) = self.0.lock() {
            *held = None;
        }
    }
}

/// Build a request to an absolute authentication URL, independent of the API base.
pub(crate) fn request(method: http::Method, url: &str) -> http::request::Builder {
    http::Request::builder().method(method).uri(url)
}

/// Send `req` and decode its JSON response.
/// Request and transport failures return [`AuthError::Http`], preserving HTTP
/// status when available. Invalid JSON returns [`AuthError::Json`].
pub(crate) async fn send_json<H, T>(
    http: &H,
    req: http::Result<http::Request<bytes::Bytes>>,
) -> Result<T, AuthError>
where
    H: HttpClientExt,
    T: serde::de::DeserializeOwned,
{
    let bytes = send_bytes(http, req).await?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Send `req` through the transport and return the raw success body.
pub(crate) async fn send_bytes<H>(
    http: &H,
    req: http::Result<http::Request<bytes::Bytes>>,
) -> Result<bytes::Bytes, AuthError>
where
    H: HttpClientExt,
{
    let req = req.map_err(http_client::Error::Protocol)?;
    let response = http.send::<_, bytes::Bytes>(req).await?;
    Ok(response.into_body().await?)
}

/// Returns true when the token is expired (or has no expiry), treating the
/// token as expired `skew_seconds` before its actual `expires_at`.
pub(crate) fn token_expired(expires_at: Option<i64>, skew_seconds: i64) -> bool {
    let now = web_time::SystemTime::now()
        .duration_since(web_time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default();

    expires_at.is_none_or(|exp| now >= exp - skew_seconds)
}

mod store;

#[cfg(not(target_family = "wasm"))]
pub use store::FileTokenStore;
pub use store::{MemoryTokenStore, StoreLock, TokenStore};
pub(crate) use store::{load_record, save_record};
