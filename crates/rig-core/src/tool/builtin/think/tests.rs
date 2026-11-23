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

    let result = tool
        .call(&mut crate::tool::ToolContext::default(), args)
        .await
        .unwrap();
    assert_eq!(
        result,
        "I need to verify the user's identity before proceeding"
    );
}

#[test]
fn the_description_has_no_source_indentation() {
    let description = ThinkTool.description();
    assert!(
        !description.contains('\n') && !description.contains("  "),
        "{description:?}"
    );
}
