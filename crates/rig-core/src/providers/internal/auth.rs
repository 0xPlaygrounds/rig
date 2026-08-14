//! Authentication error shared by the OAuth-capable providers (ChatGPT,
//! Copilot). Re-exported from each provider's `auth` module as `AuthError`.

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
    #[error(transparent)]
    Http(#[from] reqwest::Error),
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
