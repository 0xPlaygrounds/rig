use super::*;
use crate::tool::portable_tool_definition;

#[test]
fn test_think_tool_definition() {
    let tool = ThinkTool;
    let definition = portable_tool_definition(&tool);

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
