use super::*;

#[derive(Debug, thiserror::Error)]
#[error("secret detail")]
struct Concrete;

#[test]
fn envelope_is_classified_cloneable_downcastable_and_redacted() {
    let error = ToolExecutionError::provider("operator message")
        .with_model_feedback("safe feedback")
        .with_http_status(503)
        .with_source(Concrete);
    let cloned = error.clone();
    assert_eq!(error.kind(), ToolErrorKind::Provider);
    assert_eq!(error.model_feedback(), Some("safe feedback"));
    assert_eq!(error.http_status(), Some(503));
    assert!(cloned.is::<Concrete>());
    assert!(!format!("{error:?}").contains("secret detail"));
}

#[test]
fn converting_an_existing_envelope_preserves_classification() {
    let error = ToolExecutionError::from_error(ToolExecutionError::timeout("slow"));
    assert_eq!(error.kind(), ToolErrorKind::Timeout);
    assert_eq!(error.retryable(), Some(true));
}

#[test]
fn detailed_diagnostics_are_model_visible_by_default() {
    let error = ToolExecutionError::provider("upstream rejected field `region`");
    let result = ToolResult::failed(error.clone());

    assert_eq!(error.message(), "upstream rejected field `region`");
    assert_eq!(
        error.model_feedback(),
        Some("upstream rejected field `region`")
    );
    assert_eq!(
        result.output().as_text(),
        Some("upstream rejected field `region`")
    );
}

#[test]
fn sensitive_diagnostics_can_be_explicitly_redacted() {
    let error = ToolExecutionError::provider("authorization header Bearer secret-token")
        .redact_model_feedback();
    let result = ToolResult::failed(error.clone());

    assert_eq!(error.message(), "authorization header Bearer secret-token");
    assert_eq!(error.model_feedback(), Some("the tool provider failed"));
    assert_eq!(result.output().as_text(), Some("the tool provider failed"));
    assert!(!result.output().render().contains("secret-token"));
}

#[test]
fn errors_can_expose_structured_model_output() {
    let output = ToolOutput::json(serde_json::json!({
        "error": "invalid region",
        "allowed": ["us", "eu"]
    }));
    let result = ToolResult::failed(
        ToolExecutionError::invalid_args("region was invalid").with_model_output(output.clone()),
    );

    assert_eq!(result.output(), &output);
    assert_eq!(result.error().unwrap().model_output(), &output);
    assert_eq!(result.error().unwrap().model_feedback(), None);
}

#[test]
fn skip_refusal_and_permission_failure_are_distinct() {
    let skipped = ToolResult::skipped("policy");
    let refused = ToolResult::failed(ToolExecutionError::refused("tool refused"));
    let permission_failure = ToolResult::failed(ToolExecutionError::permission_denied(
        "authorization failed",
    ));
    assert!(skipped.is_skipped());
    assert!(!skipped.is_refused());
    assert!(refused.is_refused());
    assert!(!refused.is_skipped());
    assert!(!refused.is_error());
    assert!(refused.error().is_none());
    assert!(refused.refusal().is_some_and(|error| error.is_refusal()));
    assert!(permission_failure.is_error());
    assert!(!permission_failure.is_refused());
    assert!(permission_failure.refusal().is_none());
    assert!(permission_failure.is_error_kind(ToolErrorKind::PermissionDenied));
    assert!(!refused.is_error_kind(ToolErrorKind::PermissionDenied));
    assert_eq!(refused.status_name(), "denied");
    assert_eq!(permission_failure.status_name(), "error");
}

#[test]
fn execution_error_debug_redacts_operator_and_model_payloads() {
    let error = ToolExecutionError::provider("Bearer secret-operator-message")
        .with_model_output(ToolOutput::json(serde_json::json!({
            "credential": "secret-model-output"
        })))
        .with_source(Concrete);

    let debug = format!("{error:?}");
    assert!(debug.contains("kind: Provider"));
    assert!(debug.contains("model_output: \"<redacted>\""));
    assert!(debug.contains("source_configured: true"));
    for secret in [
        "secret-operator-message",
        "secret-model-output",
        "secret detail",
    ] {
        assert!(!debug.contains(secret));
    }
}

#[test]
fn debug_redacts_every_tool_result_disposition() {
    let success = ToolResult::success(ToolOutput::text("secret-success"));
    let failure = ToolResult::failed(
        ToolExecutionError::provider("secret-operator").with_model_feedback("secret-model"),
    );
    let skipped = ToolResult::skipped("secret-skip");
    let refused = ToolResult::failed(ToolExecutionError::refused("secret-refusal"));

    for (result, expected_status) in [
        (success, "success"),
        (failure, "error"),
        (skipped, "skipped"),
        (refused, "denied"),
    ] {
        let debug = format!("{result:?}");
        assert!(debug.contains(expected_status));
        for secret in [
            "secret-success",
            "secret-operator",
            "secret-model",
            "secret-skip",
            "secret-refusal",
        ] {
            assert!(!debug.contains(secret));
        }
    }
}
