use macros::rig_tool;
use runtime_agent::tool::ToolContext;

#[rig_tool(description = "portable")]
pub fn portable(value: i64) -> Result<i64, std::convert::Infallible> {
    Ok(value)
}

#[rig_tool(description = "contextual")]
pub fn contextual(#[rig(context)] context: &mut ToolContext, value: i64) -> Result<i64, std::convert::Infallible> {
    context.insert_result(value);
    Ok(value)
}

pub fn assert_owners() {
    fn portable_owner<T: portable_core::tool::Tool>() {}
    fn contextual_owner<T: runtime_agent::tool::ContextualTool>() {}
    portable_owner::<Portable>();
    contextual_owner::<Contextual>();
}
