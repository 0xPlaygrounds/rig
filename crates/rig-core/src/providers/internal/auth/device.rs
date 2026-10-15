//! The device-code prompt fallback shared by the native (non-wasm) ChatGPT
//! and Copilot authenticators.

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
