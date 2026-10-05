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
fn decode_failure_keeps_the_serde_error_as_its_source() {
    let mut context = ToolContext::new();
    context
        .inbound
        .insert(Counter::KEY.to_string(), serde_json::json!("not a number"));
    let error = context.require::<Counter>().unwrap_err();
    let source = std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<serde_json::Error>())
        .expect("decode failure should expose the serde error");
    assert!(source.is_data());
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
