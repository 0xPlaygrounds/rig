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
