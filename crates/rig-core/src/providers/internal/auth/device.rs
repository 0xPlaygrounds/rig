//! Device-flow OAuth helpers shared by the native (non-wasm) ChatGPT and
//! Copilot authenticators: token expiry checks and the device-code prompt
//! fallback.

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

/// Returns true when the token is expired (or has no expiry), treating the
/// token as expired `skew_seconds` before its actual `expires_at`.
pub(crate) fn token_expired(expires_at: Option<i64>, skew_seconds: i64) -> bool {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default();

    expires_at.is_none_or(|exp| now >= exp - skew_seconds)
}
