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

/// The `output_dimensionality` each call for `wire` asks for.
fn sent_widths(wire: &super::Embeddings) -> Vec<Option<i32>> {
    use rig_core::wire::{Mode, Wire};

    wire.encode(vec!["a".to_owned(), "b".to_owned()], Mode::Unary)
        .map(|calls| {
            calls
                .into_iter()
                .map(|(_, request)| request.output_dimensionality)
                .collect()
        })
        .unwrap_or_default()
}

/// A model this crate does not know is left at its own width, and the
/// caller's width is both sent and declared, so a reply of another width
/// fails instead of reaching a vector store.
#[test]
fn only_a_named_or_known_width_is_sent() {
    use rig_core::wire::{Capabilities, Wire};

    let unknown = super::Embeddings::new("gemini-embedding-2", None);
    assert_eq!(sent_widths(&unknown), vec![None, None]);
    assert_eq!(
        unknown.describe().capabilities,
        Capabilities::embedding(100, 0)
    );

    let known = super::Embeddings::new(super::EMBEDDING_004, None);
    assert_eq!(sent_widths(&known), vec![Some(768), Some(768)]);
    assert_eq!(
        known.describe().capabilities,
        Capabilities::embedding(100, 768)
    );

    let named = super::Embeddings::new("gemini-embedding-2", Some(1536));
    assert_eq!(sent_widths(&named), vec![Some(1536), Some(1536)]);
    assert_eq!(
        named.describe().capabilities,
        Capabilities::embedding(100, 1536).declaring(Some(1536))
    );
}
