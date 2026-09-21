use rig_core::tool::{PortableTool, Tool, ToolContext, ToolExecutionError};
use rig_derive::rig_tool;

#[rig_tool]
fn args(value: String) -> Result<String, ToolExecutionError> {
    Ok(value)
}

#[rig_tool]
async fn _context(
    _state: &mut rig_core::tool::ToolContext,
    value: String,
) -> Result<String, ToolExecutionError> {
    Ok(value)
}

#[tokio::test]
async fn function_named_args_is_not_shadowed() -> Result<(), ToolExecutionError> {
    let result = PortableTool::call(
        &Args,
        ArgsParameters {
            value: "portable".into(),
        },
    )
    .await?;
    assert_eq!(result, "portable");
    Ok(())
}

#[tokio::test]
async fn function_named_context_is_not_shadowed() -> Result<(), ToolExecutionError> {
    let result = Tool::call(
        &Context,
        &mut ToolContext::new(),
        ContextParameters {
            value: "contextual".into(),
        },
    )
    .await?;
    assert_eq!(result, "contextual");
    Ok(())
}

#[tokio::test]
async fn nested_free_function_keeps_lexical_resolution() -> Result<(), ToolExecutionError> {
    #[rig_tool]
    fn args(value: u32) -> Result<u32, ToolExecutionError> {
        Ok(value + 1)
    }

    assert_eq!(
        PortableTool::call(&Args, ArgsParameters { value: 41 }).await?,
        42
    );
    Ok(())
}
