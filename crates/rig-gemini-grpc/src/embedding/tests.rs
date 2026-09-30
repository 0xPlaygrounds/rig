use crate::completion::rpc_error;

#[test]
fn rpc_error_preserves_status_text_without_http_status() {
    let status = tonic::Status::unavailable("boom");
    let expected = status.to_string();

    let err = rpc_error(&status);

    // The raw provider error text is preserved verbatim, and there is no
    // HTTP status because gRPC is a non-HTTP transport.
    assert_eq!(err.provider_response_body(), Some(expected.as_str()));
    assert_eq!(err.provider_response_status(), None);
}

/// The embedding wire classifies its RPC failures by the same rule as the
/// completion wire: the code is kept, and only the transient codes retry.
#[test]
fn embedding_rpc_errors_carry_the_code_and_its_verdict() {
    let busy = rpc_error(&tonic::Status::resource_exhausted("quota"));
    assert!(busy.is_retryable());
    assert_eq!(
        rig_core::error::ErrorReport::from(&busy).code.as_deref(),
        Some("RESOURCE_EXHAUSTED")
    );
    let bad = rpc_error(&tonic::Status::invalid_argument("dims"));
    assert!(!bad.is_retryable());
    assert_eq!(
        rig_core::error::ErrorReport::from(&bad).code.as_deref(),
        Some("INVALID_ARGUMENT")
    );
    assert_eq!(bad.provider_response_status(), None);
}

/// An undeclared width leaves `output_dimensionality` to the server, so
/// `gemini-embedding-001` keeps its native 3072 instead of a guessed 768.
#[test]
fn only_a_declared_width_is_sent() -> anyhow::Result<()> {
    use super::{EMBEDDING_004, Embeddings};
    use rig_core::embeddings::EmbeddingWidth;
    use rig_core::wire::{Capabilities, Mode, Wire};

    let sent = |wire: &Embeddings| -> anyhow::Result<Vec<Option<i32>>> {
        Ok(wire
            .encode(vec!["text".to_owned()], Mode::Unary)?
            .into_iter()
            .map(|(_, request)| request.output_dimensionality)
            .collect())
    };

    let native = Embeddings::new("gemini-embedding-001");
    anyhow::ensure!(sent(&native)? == vec![None]);
    anyhow::ensure!(native.describe().capabilities == Capabilities::embedding(100, 3072));

    let declared = Embeddings::new(EMBEDDING_004).with_ndims(256);
    anyhow::ensure!(sent(&declared)? == vec![Some(256)]);
    anyhow::ensure!(
        declared.describe().capabilities == Capabilities::embedding(100, 256).declaring(Some(256))
    );
    Ok(())
}

/// A width the proto's `i32` cannot hold is a request error, not a wrapped
/// negative width on the wire.
#[test]
fn a_width_beyond_i32_is_refused() {
    use super::Embeddings;
    use rig_core::embeddings::EmbeddingWidth;
    use rig_core::wire::{Mode, Wire};

    let too_wide = i32::MAX as usize + 1;
    let result = Embeddings::new("gemini-embedding-001")
        .with_ndims(too_wide)
        .encode(vec!["text".to_owned()], Mode::Unary);
    assert!(
        result.is_err(),
        "a width of {too_wide} must not reach the wire"
    );
}
