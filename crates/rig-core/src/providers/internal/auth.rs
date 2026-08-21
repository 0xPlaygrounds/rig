//! Authentication error shared by the OAuth-capable providers (ChatGPT,
//! Copilot). Re-exported from each provider's `auth` module as `AuthError`.

use crate::http_client::{self, HttpClientExt};
use std::sync::Arc;

/// Device authorization details surfaced to a provider callback.
#[derive(Debug, Clone)]
pub struct DeviceCodePrompt {
    /// URL where the user authorizes the device.
    pub verification_uri: String,
    /// Short code the user enters at the verification URL.
    pub user_code: String,
}

/// Optional callback invoked when an OAuth device flow needs user action.
#[derive(Clone, Default)]
pub struct DeviceCodeHandler(pub(crate) Option<Arc<dyn Fn(DeviceCodePrompt) + Send + Sync>>);

impl DeviceCodeHandler {
    /// Wraps a device-code callback.
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(DeviceCodePrompt) + Send + Sync + 'static,
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

/// Build a request to an auth endpoint. Auth flows talk to fixed, absolute
/// URLs (GitHub / OpenAI auth hosts), not the provider's API base, so they
/// drive the transport directly instead of going through the provider client.
pub(crate) fn request(method: http::Method, url: &str) -> http::request::Builder {
    http::Request::builder().method(method).uri(url)
}

/// Send `req` through the transport and decode a JSON body.
///
/// A non-success status surfaces as `AuthError::Http` carrying the
/// transport's status-bearing error (the equivalent of reqwest's
/// `error_for_status`), so callers that need to branch on a status — device
/// flows polling for authorization — read it off the error.
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
