use super::*;
use crate::tool::tool_definition;

#[test]
fn test_think_tool_definition() {
    let tool = ThinkTool;
    let definition = tool_definition(&tool);

    assert_eq!(definition.name, "think");
    assert!(
        definition
            .description
            .contains("Use the tool to think about something")
    );
}

#[tokio::test]
async fn test_think_tool_call() {
    let tool = ThinkTool;
    let args = ThinkArgs {
        thought: "I need to verify the user's identity before proceeding".to_string(),
    };

    let result = tool.call(args).await.unwrap();
    assert_eq!(
        result,
        "I need to verify the user's identity before proceeding"
    );
}

/// A failure reports as a failed tool, as the tool runner reports it.
#[test]
fn a_think_failure_converts_as_the_tool_runner_reports_it() {
    let error = crate::error::RigError::from(ThinkError("no thought".to_owned()));
    assert_eq!(
        error.kind,
        crate::error::ErrorKind::Tool(crate::tool::ToolErrorKind::Other)
    );
    assert!(!error.retryable);
    assert_eq!(error.message, "Think tool error: no thought");
    assert_eq!(error.source_chain, ["Think tool error: no thought"]);
}
