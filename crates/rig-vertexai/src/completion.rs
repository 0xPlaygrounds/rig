//! Model identifiers and completion error mapping.
//!
//! All supported models: <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini>
//!
//! Completions are driven through [`crate::functions`]; pass one of the
//! constants below as the `model` argument.

use rig_core::completion::CompletionError;

/// `gemini-1.5-pro`
pub const GEMINI_1_5_PRO: &str = "gemini-1.5-pro";
/// `gemini-1.5-flash`
pub const GEMINI_1_5_FLASH: &str = "gemini-1.5-flash";
/// `gemini-1.5-pro-latest`
pub const GEMINI_1_5_PRO_LATEST: &str = "gemini-1.5-pro-latest";
/// `gemini-1.5-flash-latest`
pub const GEMINI_1_5_FLASH_LATEST: &str = "gemini-1.5-flash-latest";
/// `gemini-2.0-flash-exp`
pub const GEMINI_2_0_FLASH_EXP: &str = "gemini-2.0-flash-exp";
/// `gemini-2.5-flash-lite`
pub const GEMINI_2_5_FLASH_LITE: &str = "gemini-2.5-flash-lite";
/// `gemini-2.5-flash`
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.5-pro`
pub const GEMINI_2_5_PRO: &str = "gemini-2.5-pro";

/// Map a failed `send()` RPC into a [`CompletionError`] that preserves the
/// provider's gRPC error text verbatim.
///
/// Vertex AI uses a non-HTTP (gRPC/SDK) transport, so there is no
/// [`http::StatusCode`] to attach; the error body is preserved via
/// [`CompletionError::from_provider_body`] (`status: None`) rather than a
/// Rig-prefixed [`CompletionError::ProviderError`] diagnostic. (The
/// `get_inner()` client-init failure stays a `ProviderError` because it is a
/// Rig-side setup failure, not a provider response.)
///
/// Note: the SDK does not distinguish a server-returned gRPC error from a
/// transport/connection failure, so a pure connection error is also preserved
/// here (`status: None`) rather than gated out as a Rig diagnostic the way
/// Bedrock's typed service errors are.
pub(crate) fn rpc_error(error: impl std::fmt::Display) -> CompletionError {
    CompletionError::from_provider_body(error.to_string())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    // The `send()` RPC error type comes from the `google-cloud-aiplatform-v1`
    // SDK and is not trivially constructible, so `rpc_error` is generic over
    // `impl Display` and we pin it here with a representative error string of
    // its parameter type. This guards against a revert to `ProviderError`,
    // which would surface the body as `None`.
    #[test]
    fn rpc_error_preserves_raw_text_without_http_status() {
        let raw = "status: Unavailable, message: \"the service is currently unavailable\"";

        let err = rpc_error(raw);

        assert_eq!(err.provider_response_body(), Some(raw));
        assert_eq!(err.provider_response_status(), None);
    }
}
