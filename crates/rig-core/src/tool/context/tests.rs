use super::*;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Counter(u32);
impl ContextValue for Counter {
    const KEY: &'static str = "test.counter";
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct RequestId(String);
impl ContextValue for RequestId {
    const KEY: &'static str = "test.request_id";
}

// Two shapes that deliberately share a key: the fixture for "the slot holds
// something that is not a `T`".
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct A(u32);
impl ContextValue for A {
    const KEY: &'static str = "k";
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct B {
    name: String,
}
impl ContextValue for B {
    const KEY: &'static str = "k";
}

#[test]
fn context_separates_inbound_and_result_values() {
    let mut context = ToolContext::new();
    context.insert(Counter(42)).unwrap();
    context
        .insert_result(RequestId("request-1".to_string()))
        .unwrap();
    assert_eq!(context.get::<Counter>().unwrap(), Some(Counter(42)));
    assert_eq!(
        context.result::<RequestId>().unwrap(),
        Some(RequestId("request-1".to_string()))
    );

    let next = context.for_dispatch();
    assert_eq!(next.get::<Counter>().unwrap(), Some(Counter(42)));
    assert_eq!(next.result::<RequestId>().unwrap(), None);
}

#[test]
fn missing_context_converts_into_a_tool_execution_error() {
    fn require_value(context: &ToolContext) -> Result<Counter, ToolExecutionError> {
        Ok(context.require::<Counter>()?)
    }

    let error = require_value(&ToolContext::new()).unwrap_err();
    assert!(error.is::<ToolContextError>());
    assert_eq!(
        error.model_feedback(),
        Some("required tool context value `test.counter` was not found")
    );
}

#[test]
fn context_round_trips_through_serde() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Session {
        id: String,
    }
    impl ContextValue for Session {
        const KEY: &'static str = "test.session";
    }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Seq(u64);
    impl ContextValue for Seq {
        const KEY: &'static str = "test.seq";
    }
    let mut context = ToolContext::new();
    context
        .insert(Session {
            id: "abc".to_string(),
        })
        .unwrap();
    context.insert_result(Seq(7)).unwrap();

    let json = serde_json::to_string(&context).unwrap();
    let back: ToolContext = serde_json::from_str(&json).unwrap();
    assert_eq!(back, context);
    assert_eq!(
        back.get::<Session>().unwrap(),
        Some(Session {
            id: "abc".to_string()
        })
    );
    assert_eq!(back.result::<Seq>().unwrap(), Some(Seq(7)));

    let empty: ToolContext = serde_json::from_str("{}").unwrap();
    assert!(empty.is_empty());
    assert_eq!(serde_json::to_string(&ToolContext::new()).unwrap(), "{}");
}

#[test]
fn decode_mismatch_is_reported_by_require_and_get() {
    // A slot holds JSON; a type whose key collides but whose shape differs
    // decodes as an error, never as a panic. `get` distinguishes the
    // mismatch from an empty slot.
    let mut context = ToolContext::new();
    context
        .inbound
        .insert(Counter::KEY.to_string(), serde_json::json!("not a number"));
    assert!(matches!(
        context.get::<Counter>(),
        Err(ToolContextError::Decode { key, .. }) if key == Counter::KEY
    ));
    assert!(matches!(
        context.require::<Counter>(),
        Err(ToolContextError::Decode { key, .. }) if key == Counter::KEY
    ));
}

#[test]
fn get_reports_a_decode_failure_for_a_slot_holding_a_different_shape() {
    let mut context = ToolContext::new();
    context.insert(A(1)).unwrap();
    assert!(matches!(
        context.get::<B>(),
        Err(ToolContextError::Decode { key: B::KEY, .. })
    ));
    // The slot itself is intact and still reads back as what was written.
    assert_eq!(context.get::<A>().unwrap(), Some(A(1)));
}

#[test]
fn insert_replaces_an_undecodable_displaced_value_and_returns_none() {
    let mut context = ToolContext::new();
    context.insert(A(1)).unwrap();
    assert_eq!(
        context
            .insert(B {
                name: "b".to_string()
            })
            .unwrap(),
        None
    );
    assert_eq!(
        context.get::<B>().unwrap(),
        Some(B {
            name: "b".to_string()
        })
    );
}

#[test]
fn the_scope_is_not_data() {
    // The driver's scope rides on the context for the call and nowhere
    // else: not on the wire, not in equality, gone once cleared.
    let scope: std::sync::Arc<dyn Any + Send + Sync> = std::sync::Arc::new(7u32);
    let mut scoped = ToolContext::new().with_scope(scope);
    assert_eq!(scoped.scope::<u32>().as_deref(), Some(&7));
    assert_eq!(
        scoped.scope::<String>(),
        None,
        "another type is not the scope"
    );
    assert_eq!(
        scoped,
        ToolContext::new(),
        "the scope is not part of equality"
    );
    let json = serde_json::to_value(&scoped).expect("serializes");
    assert_eq!(json, serde_json::json!({}), "never on the wire");
    let dispatch = scoped.for_dispatch();
    assert_eq!(
        dispatch.scope::<u32>().as_deref(),
        Some(&7),
        "a nested inline call keeps the scope"
    );
    scoped.clear_scope();
    assert_eq!(scoped.scope::<u32>(), None);
    assert_eq!(
        ToolContext::new().scope::<u32>(),
        None,
        "an inline call has none"
    );
}
