//! Shared OAuth errors, device-code callbacks, and authentication transport helpers.
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
    /// inspectable); response-less failures as [`http_client::Error::Instance`].
    #[error(transparent)]
    Http(#[from] http_client::Error),
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

/// Platform config directory used for on-disk OAuth/token caches
/// (`APPDATA` on Windows; `XDG_CONFIG_HOME` falling back to `~/.config`
/// elsewhere).
pub(crate) fn config_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;

    #[cfg(target_os = "windows")]
    {
        std::env::var_os("APPDATA").map(PathBuf::from)
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
    }
}
