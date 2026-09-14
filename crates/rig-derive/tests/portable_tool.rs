use rig_core::tool::{PortableTool, ToolExecutionError};
use rig_derive::rig_tool;

#[rig_tool(description = "A context-free portable addition tool")]
fn portable_add(left: i32, right: i32) -> Result<i32, ToolExecutionError> {
    Ok(left + right)
}

fn assert_portable<T: PortableTool>() {}

#[tokio::test]
async fn context_free_macro_targets_the_portable_tool_contract() {
    assert_portable::<PortableAdd>();

    let output = PORTABLE_ADD
        .call(PortableAddParameters { left: 2, right: 3 })
        .await;

    assert_eq!(output.ok(), Some(5));
}
