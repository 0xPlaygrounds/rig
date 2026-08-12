//! Authentication error shared by the OAuth-capable providers (ChatGPT,
//! Copilot). Re-exported from each provider's `auth` module as `AuthError`.

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
