use core_runtime::tool::{PortableTool, ToolExecutionError};
use rig_macros::rig_tool;

#[rig_tool]
fn echo(value: String) -> Result<String, ToolExecutionError> {
    Ok(value)
}

fn assert_portable<T: PortableTool>() {}

fn main() {
    assert_portable::<Echo>();
}
