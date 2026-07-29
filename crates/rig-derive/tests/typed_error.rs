#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]

use rig_agent::tool::{PortableTool, ToolErrorKind};
use rig_derive::rig_tool;

#[derive(Debug)]
struct DomainError;

impl std::fmt::Display for DomainError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("typed domain failure")
    }
}

impl std::error::Error for DomainError {}

#[rig_tool(description = "Return a typed domain error")]
fn typed_failure() -> Result<(), DomainError> {
    Err(DomainError)
}

#[rig_tool(description = "Return a typed domain error asynchronously")]
async fn async_typed_failure() -> Result<(), DomainError> {
    Err(DomainError)
}

#[tokio::test]
async fn derive_preserves_typed_errors_until_dispatch() {
    let direct = TypedFailure.call(TypedFailureParameters {}).await;
    assert!(direct.is_err());
    if let Err(error) = direct {
        let _: &DomainError = &error;
        assert_eq!(error.to_string(), "typed domain failure");
    }

    // Dispatch through the erased portable record normalizes the typed error
    // exactly like the classic registry dispatch did.
    let error = TypedFailure
        .portable()
        .execute(serde_json::json!({}))
        .await
        .expect_err("the tool always fails");
    assert_eq!(error.kind(), ToolErrorKind::Other);
    assert_eq!(error.message(), "typed domain failure");
    assert_eq!(error.model_feedback(), Some("the tool failed"));
    assert!(error.is::<DomainError>());
}

#[tokio::test]
async fn async_derive_preserves_typed_errors_until_dispatch() {
    let direct = AsyncTypedFailure.call(AsyncTypedFailureParameters {}).await;
    assert!(direct.is_err());
    if let Err(error) = direct {
        let _: &DomainError = &error;
    }

    let error = AsyncTypedFailure
        .portable()
        .execute(serde_json::json!({}))
        .await
        .expect_err("the tool always fails");
    assert!(error.is::<DomainError>());
}
