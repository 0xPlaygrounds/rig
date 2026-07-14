use rig_core::tool::{Tool, ToolContext, ToolErrorKind, ToolSet};
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
    let direct = TypedFailure
        .call(&mut ToolContext::new(), TypedFailureParameters {})
        .await;
    assert!(direct.is_err());
    if let Err(error) = direct {
        let _: &DomainError = &error;
        assert_eq!(error.to_string(), "typed domain failure");
    }

    let mut tools = ToolSet::default();
    tools.add_tool(TypedFailure);
    let result = tools
        .execute(TypedFailure::NAME, "{}", &mut ToolContext::new())
        .await;
    assert!(result.error().is_some());
    if let Some(error) = result.error() {
        assert_eq!(error.kind(), ToolErrorKind::Other);
        assert_eq!(error.message(), "typed domain failure");
        assert_eq!(error.model_feedback(), Some("the tool failed"));
        assert!(error.is::<DomainError>());
    }
}

#[tokio::test]
async fn async_derive_preserves_typed_errors_until_dispatch() {
    let direct = AsyncTypedFailure
        .call(&mut ToolContext::new(), AsyncTypedFailureParameters {})
        .await;
    assert!(direct.is_err());
    if let Err(error) = direct {
        let _: &DomainError = &error;
    }

    let mut tools = ToolSet::default();
    tools.add_tool(AsyncTypedFailure);
    let result = tools
        .execute(AsyncTypedFailure::NAME, "{}", &mut ToolContext::new())
        .await;
    assert!(result.error().is_some());
    if let Some(error) = result.error() {
        assert!(error.is::<DomainError>());
    }
}
