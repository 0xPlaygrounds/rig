//! Device-flow OAuth helpers shared by the native (non-wasm) ChatGPT and
//! Copilot authenticators: on-disk JSON record caching, token expiry checks,
//! and the device-code prompt fallback.

use super::AuthError;
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

/// Creates the directories above `path`; on Unix the ones it creates are
/// readable by the owner alone (0700), since they hold credentials.
fn ensure_parent_dir(path: &Path) -> Result<(), std::io::Error> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(parent)
}

/// Writes a credential file (creating parent directories as
/// [`ensure_parent_dir`] does). On Unix the file is created 0600, and an
/// existing file is narrowed to 0600 before anything is written to it.
pub(crate) fn write_private(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    use std::io::Write;

    ensure_parent_dir(path)?;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options.open(path)?;
    #[cfg(unix)]
    file.set_permissions(std::os::unix::fs::PermissionsExt::from_mode(0o600))?;
    file.write_all(bytes)
}

/// Returns true when the token is expired (or has no expiry), treating the
/// token as expired `skew_seconds` before its actual `expires_at`.
pub(crate) fn token_expired(expires_at: Option<i64>, skew_seconds: i64) -> bool {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default();

    expires_at.is_none_or(|exp| now >= exp - skew_seconds)
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

/// Writes a JSON record to `path` with [`write_private`], a no-op when no
/// path is configured.
pub(crate) fn write_json_record<T: Serialize>(
    path: Option<&Path>,
    record: &T,
) -> Result<(), AuthError> {
    let Some(path) = path else {
        return Ok(());
    };

    write_private(path, &serde_json::to_vec_pretty(record)?)?;
    Ok(())
}
