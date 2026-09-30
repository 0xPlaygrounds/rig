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

/// The `output_dimensionality` of each request the wire encodes.
fn output_dimensionalities(wire: &super::Embeddings) -> Vec<Option<i32>> {
    use rig_core::wire::{Mode, Wire};

    wire.encode(vec!["text".to_owned()], Mode::Unary)
        .map(|requests| {
            requests
                .into_iter()
                .map(|(_, request)| request.output_dimensionality)
                .collect()
        })
        .unwrap_or_default()
}

/// Without a width the request names none, so the model answers at its own
/// default; a known model still reports its width.
#[test]
fn an_unset_width_is_not_sent() {
    use rig_core::wire::Wire;

    let wire = super::Embeddings::new(super::EMBEDDING_004);

    assert_eq!(output_dimensionalities(&wire), vec![None]);
    assert_eq!(wire.describe().capabilities.ndims, 768);
    assert_eq!(wire.describe().capabilities.declared, None);
    assert_eq!(
        super::Embeddings::new("gemini-embedding-001")
            .describe()
            .capabilities
            .ndims,
        0,
        "an unknown model reports the unknown width rather than a guess"
    );
}

#[test]
fn with_ndims_is_sent_and_declared() {
    use rig_core::wire::Wire;

    let wire = super::Embeddings::new(super::EMBEDDING_004).with_ndims(256);

    assert_eq!(output_dimensionalities(&wire), vec![Some(256)]);
    assert_eq!(wire.describe().capabilities.ndims, 256);
    assert_eq!(wire.describe().capabilities.declared, Some(256));
}
