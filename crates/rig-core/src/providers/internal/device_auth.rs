//! Device-flow OAuth helpers shared by the native (non-wasm) ChatGPT and
//! Copilot authenticators: on-disk JSON record caching, token expiry checks,
//! and the device-code prompt fallback.

use super::auth::AuthError;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::path::Path;
use std::sync::Arc;

/// Invokes the provider's device-code callback when one is registered,
/// otherwise prints `fallback_message` to stdout.
pub(crate) fn emit_device_code_prompt<P>(
    callback: Option<&Arc<dyn Fn(P) + Send + Sync>>,
    prompt: P,
    fallback_message: &str,
) {
    if let Some(callback) = callback {
        callback(prompt);
    } else {
        println!("{fallback_message}");
    }
}

pub(crate) fn ensure_parent_dir(path: &Path) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

/// Returns true when the token is expired (or has no expiry), treating the
/// token as expired `skew_seconds` before its actual `expires_at`.
pub(crate) fn token_expired(expires_at: Option<i64>, skew_seconds: i64) -> bool {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default();

    match expires_at {
        Some(exp) => now >= exp - skew_seconds,
        None => true,
    }
}

/// Reads a JSON record from `path`, returning `T::default()` when no path is
/// configured or the file does not exist.
pub(crate) fn read_json_record<T: Default + DeserializeOwned>(
    path: Option<&Path>,
) -> Result<T, AuthError> {
    let Some(path) = path else {
        return Ok(T::default());
    };

    match std::fs::read(path) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(T::default()),
        Err(err) => Err(err.into()),
    }
}

/// Writes a JSON record to `path` (creating parent directories), a no-op when
/// no path is configured.
pub(crate) fn write_json_record<T: Serialize>(
    path: Option<&Path>,
    record: &T,
) -> Result<(), AuthError> {
    let Some(path) = path else {
        return Ok(());
    };

    ensure_parent_dir(path)?;
    std::fs::write(path, serde_json::to_vec_pretty(record)?)?;
    Ok(())
}
