use super::*;

#[test]
fn per_kind_constructors_set_default_retryability() {
    for (error, retryable) in [
        (ToolExecutionError::timeout("t"), Some(true)),
        (ToolExecutionError::rate_limited("r"), Some(true)),
        (ToolExecutionError::network("n"), Some(true)),
        (ToolExecutionError::not_found("nf"), Some(false)),
        (ToolExecutionError::permission_denied("p"), Some(false)),
        (ToolExecutionError::invalid_args("i"), Some(false)),
        (ToolExecutionError::cancelled("c"), Some(false)),
        (ToolExecutionError::provider("p"), None),
        (ToolExecutionError::other("o"), None),
    ] {
        assert_eq!(error.retryable(), retryable);
    }
}

#[test]
fn error_builder_preserves_policy_fields_and_feedback() {
    let error = ToolExecutionError::rate_limited("operator")
        .with_model_feedback("slow down")
        .with_retryable(false)
        .with_code("RATE_42")
        .with_http_status(429);
    assert_eq!(error.kind(), ToolErrorKind::RateLimited);
    assert_eq!(error.message(), "operator");
    assert_eq!(error.model_feedback(), Some("slow down"));
    assert_eq!(error.retryable(), Some(false));
    assert_eq!(error.code(), Some("RATE_42"));
    assert_eq!(error.http_status(), Some(429));
    let result = ToolResult::failed(error);
    assert_eq!(result.output().as_text(), Some("slow down"));
    assert!(result.is_error_kind(ToolErrorKind::RateLimited));
}

#[test]
fn success_preserves_multiline_output_verbatim() {
    let result = ToolResult::success(ToolOutput::text("hello\nworld"));
    assert!(result.is_success());
    assert_eq!(result.output().as_text(), Some("hello\nworld"));
    assert!(result.error().is_none());
}

#[test]
fn result_states_are_mutually_distinguishable() {
    let success = ToolResult::success(ToolOutput::text("ok"));
    let failure = ToolResult::failed(ToolExecutionError::not_found("missing"));
    let skipped = ToolResult::skipped("policy");
    let refused = ToolResult::failed(ToolExecutionError::refused("denied"));
    assert!(success.is_success());
    assert!(failure.is_error());
    assert!(skipped.is_skipped());
    assert!(refused.is_refused());
    assert!(!refused.is_error());
    assert!(!skipped.is_refused());
    assert!(!refused.is_skipped());
    assert_eq!(success.status_name(), "success");
    assert_eq!(failure.status_name(), "error");
    assert_eq!(skipped.status_name(), "skipped");
    assert_eq!(refused.status_name(), "denied");
}

#[test]
fn from_error_keeps_existing_envelope_and_wraps_other_sources() {
    #[derive(Debug, thiserror::Error)]
    #[error("boom")]
    struct Boom;
    let existing = ToolExecutionError::timeout("slow").with_code("T");
    let kept = ToolExecutionError::from_error(existing);
    assert_eq!(kept.kind(), ToolErrorKind::Timeout);
    assert_eq!(kept.code(), Some("T"));
    let wrapped = ToolExecutionError::from_error(Boom);
    assert_eq!(wrapped.kind(), ToolErrorKind::Other);
    assert!(wrapped.is::<Boom>());
    assert_eq!(wrapped.message(), "boom");
    assert_eq!(wrapped.model_feedback(), Some("the tool failed"));
}

#[test]
fn from_error_preserves_refusal_disposition() {
    let refused =
        ToolExecutionError::from_error(ToolExecutionError::refused("declined").with_code("POLICY"));
    assert!(refused.is_refusal());
    assert_eq!(refused.kind(), ToolErrorKind::PermissionDenied);
    assert_eq!(refused.code(), Some("POLICY"));
}
