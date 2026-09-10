use super::*;

#[test]
fn context_separates_inbound_and_result_values() {
    let mut context = ToolContext::new();
    context.insert(42_u32);
    context.insert_result("request-1".to_string());
    assert_eq!(context.get::<u32>(), Some(&42));
    assert_eq!(
        context.result::<String>().map(String::as_str),
        Some("request-1")
    );

    let next = context.for_dispatch();
    assert_eq!(next.get::<u32>(), Some(&42));
    assert!(next.result::<String>().is_none());
}

#[test]
fn missing_context_converts_into_a_tool_execution_error() {
    fn require_value(context: &ToolContext) -> Result<u32, ToolExecutionError> {
        Ok(*context.require::<u32>()?)
    }

    let error = require_value(&ToolContext::new()).unwrap_err();
    assert!(error.is::<MissingToolContext>());
    assert_eq!(
        error.model_feedback(),
        Some("required tool context value of type `u32` was not found")
    );
}
