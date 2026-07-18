use facade::tool_macro as rig_tool;
use facade::tool::ToolContext;

#[rig_tool(description = "portable through renamed facade")]
pub fn portable(value: i64) -> Result<i64, std::convert::Infallible> {
    Ok(value)
}

#[rig_tool(description = "contextual through renamed facade")]
pub fn contextual(
    #[rig(context)] context: &mut ToolContext,
    value: i64,
) -> Result<i64, std::convert::Infallible> {
    context.insert_result(value);
    Ok(value)
}

pub fn assert_owners() {
    fn portable_owner<T: facade::tool::Tool>() {}
    fn contextual_owner<T: facade::tool::ContextualTool>() {}
    portable_owner::<Portable>();
    contextual_owner::<Contextual>();
}
