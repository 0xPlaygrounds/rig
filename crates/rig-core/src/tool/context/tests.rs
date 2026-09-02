use super::*;

#[test]
fn context_separates_inbound_and_result_values() {
    let mut context = ToolContext::new();
    context.insert(42_u32).unwrap();
    context.insert_result("request-1".to_string()).unwrap();
    assert_eq!(context.get::<u32>(), Some(42));
    assert_eq!(context.result::<String>().as_deref(), Some("request-1"));

    let next = context.for_dispatch();
    assert_eq!(next.get::<u32>(), Some(42));
    assert!(next.result::<String>().is_none());
}

#[test]
fn missing_context_converts_into_a_tool_execution_error() {
    fn require_value(context: &ToolContext) -> Result<u32, ToolExecutionError> {
        Ok(context.require::<u32>()?)
    }

    let error = require_value(&ToolContext::new()).unwrap_err();
    assert!(error.is::<ToolContextError>());
    assert_eq!(
        error.model_feedback(),
        Some("required tool context value of type `u32` was not found")
    );
}

#[test]
fn context_round_trips_through_serde() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Session {
        id: String,
    }
    let mut context = ToolContext::new();
    context
        .insert(Session {
            id: "abc".to_string(),
        })
        .unwrap();
    context.insert_result(7_u64).unwrap();

    let json = serde_json::to_string(&context).unwrap();
    let back: ToolContext = serde_json::from_str(&json).unwrap();
    assert_eq!(back, context);
    assert_eq!(
        back.get::<Session>(),
        Some(Session {
            id: "abc".to_string()
        })
    );
    assert_eq!(back.result::<u64>(), Some(7));

    let empty: ToolContext = serde_json::from_str("{}").unwrap();
    assert!(empty.is_empty());
    assert_eq!(serde_json::to_string(&ToolContext::new()).unwrap(), "{}");
}

#[test]
fn decode_mismatch_is_reported_by_require_and_absent_from_get() {
    // A slot holds JSON; a type whose name collides but whose shape differs
    // decodes as an error, never as a panic.
    let mut context = ToolContext::new();
    context.inbound.insert(
        slot_key::<u32>().to_string(),
        serde_json::json!("not a number"),
    );
    assert_eq!(context.get::<u32>(), None);
    assert!(matches!(
        context.require::<u32>(),
        Err(ToolContextError::Decode { key, .. }) if key == "u32"
    ));
}
