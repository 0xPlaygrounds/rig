use rig_core::tool::{Tool, ToolContext, ToolExecutionError};
use rig_derive::rig_tool;

#[rig_tool(description = "A context-free portable addition tool")]
fn portable_add(left: i32, right: i32) -> Result<i32, ToolExecutionError> {
    Ok(left + right)
}

fn assert_tool<T: Tool>() {}

#[tokio::test]
async fn a_context_free_function_is_a_tool() {
    assert_tool::<PortableAdd>();

    let output = PORTABLE_ADD
        .call(
            &mut ToolContext::new(),
            PortableAddParameters { left: 2, right: 3 },
        )
        .await;

    assert_eq!(output.ok(), Some(5));
}
