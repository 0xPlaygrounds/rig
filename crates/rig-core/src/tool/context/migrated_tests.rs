use super::*;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Num(u32);
impl ContextValue for Num {
    const KEY: &'static str = "test.num";
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Greeting(String);
impl ContextValue for Greeting {
    const KEY: &'static str = "test.greeting";
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Bytes(Vec<u8>);
impl ContextValue for Bytes {
    const KEY: &'static str = "test.bytes";
}

#[test]
fn insert_and_get_returns_value() {
    let mut c = ToolContext::new();
    assert_eq!(c.insert(Num(42)).unwrap(), None);
    assert_eq!(c.get::<Num>().unwrap(), Some(Num(42)));
}
#[test]
fn get_missing_type_returns_none() {
    assert_eq!(ToolContext::new().get::<Num>().unwrap(), None);
}
#[test]
fn insert_overwrites_and_returns_previous() {
    let mut c = ToolContext::new();
    c.insert(Num(1)).unwrap();
    assert_eq!(c.insert(Num(2)).unwrap(), Some(Num(1)));
    assert_eq!(c.get::<Num>().unwrap(), Some(Num(2)));
}
#[test]
fn different_types_are_independent() {
    let mut c = ToolContext::new();
    c.insert(Num(42)).unwrap();
    c.insert(Greeting("hello".to_string())).unwrap();
    assert_eq!(c.get::<Num>().unwrap(), Some(Num(42)));
    assert_eq!(
        c.get::<Greeting>().unwrap(),
        Some(Greeting("hello".to_string()))
    );
}
#[test]
fn contains_tracks_types() {
    let mut c = ToolContext::new();
    c.insert(Num(42)).unwrap();
    assert!(c.contains::<Num>());
    assert!(!c.contains::<Greeting>());
}
#[test]
fn clone_produces_independent_copy() {
    let mut c = ToolContext::new();
    c.insert(Num(42)).unwrap();
    let mut clone = c.clone();
    clone.insert(Num(99)).unwrap();
    assert_eq!(c.get::<Num>().unwrap(), Some(Num(42)));
    assert_eq!(clone.get::<Num>().unwrap(), Some(Num(99)));
}
#[test]
fn clone_deep_copies_heap_values() {
    let mut c = ToolContext::new();
    c.insert(Bytes(vec![1u8, 2, 3])).unwrap();
    let mut clone = c.clone();
    let mut bytes = clone.remove::<Bytes>().unwrap().unwrap();
    bytes.0.push(4);
    clone.insert(bytes).unwrap();
    assert_eq!(c.get::<Bytes>().unwrap(), Some(Bytes(vec![1, 2, 3])));
    assert_eq!(clone.get::<Bytes>().unwrap(), Some(Bytes(vec![1, 2, 3, 4])));
}
#[test]
fn empty_context_is_default_and_serializes_empty() {
    let c = ToolContext::default();
    assert!(!c.contains::<Num>());
    assert!(c.is_empty());
    assert_eq!(serde_json::to_value(&c).unwrap(), serde_json::json!({}));
}
#[test]
fn reinsert_replaces_in_place() {
    let mut c = ToolContext::new();
    c.insert(Num(42)).unwrap();
    assert_eq!(c.insert(Num(99)).unwrap(), Some(Num(42)));
    assert_eq!(c.get::<Num>().unwrap(), Some(Num(99)));
}
#[test]
fn remove_returns_value_and_clears_entry() {
    let mut c = ToolContext::new();
    c.insert(Num(42)).unwrap();
    assert_eq!(c.remove::<Num>().unwrap(), Some(Num(42)));
    assert!(!c.contains::<Num>());
}
#[test]
fn remove_missing_type_returns_none() {
    assert_eq!(ToolContext::new().remove::<Num>().unwrap(), None);
}
#[test]
fn require_present_returns_value() {
    let mut c = ToolContext::new();
    c.insert(Num(42)).unwrap();
    assert_eq!(c.require::<Num>(), Ok(Num(42)));
}
#[test]
fn require_missing_names_key() {
    let e = ToolContext::new().require::<Num>().unwrap_err();
    assert_eq!(e, ToolContextError::Missing(Num::KEY));
    assert!(e.to_string().contains("`test.num`"));
}
#[test]
fn result_metadata_round_trips_and_requires() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Id(u32);
    impl ContextValue for Id {
        const KEY: &'static str = "test.id";
    }
    let mut c = ToolContext::new();
    c.insert_result(Id(7)).unwrap();
    assert_eq!(c.result::<Id>().unwrap(), Some(Id(7)));
    assert_eq!(c.require_result::<Id>(), Ok(Id(7)));
    assert_eq!(c.get::<Id>().unwrap(), None);
}
#[test]
fn debug_reports_keys_without_values() {
    #[derive(Serialize, Deserialize)]
    struct Secret(String);
    impl ContextValue for Secret {
        const KEY: &'static str = "test.secret";
    }
    let mut c = ToolContext::new();
    c.insert(Num(42)).unwrap();
    c.insert_result(Secret("do-not-print".to_string())).unwrap();
    let d = format!("{c:?}");
    assert!(d.contains(Num::KEY));
    assert!(d.contains(Secret::KEY));
    assert!(!d.contains("do-not-print"));
    assert_eq!(
        c.result::<Secret>().unwrap().map(|s| s.0).as_deref(),
        Some("do-not-print")
    );
}
#[test]
fn dispatch_snapshot_isolates_inbound_and_publishes_only_result_metadata() {
    let mut c = ToolContext::new();
    c.insert(Num(7)).unwrap();
    c.insert_result(Greeting("old".to_string())).unwrap();
    let mut d = c.for_dispatch();
    assert_eq!(d.get::<Num>().unwrap(), Some(Num(7)));
    assert_eq!(d.result::<Greeting>().unwrap(), None);
    d.insert(Num(8)).unwrap();
    d.insert_result(Greeting("new".to_string())).unwrap();

    c.accept_dispatch_result(d);
    assert_eq!(c.get::<Num>().unwrap(), Some(Num(7)));
    assert_eq!(
        c.result::<Greeting>().unwrap(),
        Some(Greeting("new".to_string()))
    );
}
#[test]
fn many_distinct_types_round_trip() {
    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct A(u8);
    impl ContextValue for A {
        const KEY: &'static str = "test.a";
    }
    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct B(u16);
    impl ContextValue for B {
        const KEY: &'static str = "test.b";
    }
    let mut c = ToolContext::new();
    c.insert(A(1)).unwrap();
    c.insert(B(2)).unwrap();
    c.insert(Num(3)).unwrap();
    c.insert(Greeting("four".to_string())).unwrap();
    assert_eq!(c.get::<A>().unwrap(), Some(A(1)));
    assert_eq!(c.get::<B>().unwrap(), Some(B(2)));
    assert_eq!(c.get::<Num>().unwrap(), Some(Num(3)));
    assert_eq!(
        c.get::<Greeting>().unwrap(),
        Some(Greeting("four".to_string()))
    );
}
#[test]
fn unencodable_value_is_an_error_not_a_panic() {
    // A map with non-string keys has no JSON form.
    #[derive(Serialize, Deserialize, Debug)]
    struct TupleKeyed(std::collections::HashMap<(u8, u8), u8>);
    impl ContextValue for TupleKeyed {
        const KEY: &'static str = "test.tuple_keyed";
    }
    let mut c = ToolContext::new();
    let value = TupleKeyed([((1, 2), 3)].into_iter().collect());
    let err = c.insert(value).unwrap_err();
    assert!(matches!(
        err,
        ToolContextError::Encode {
            key: TupleKeyed::KEY,
            ..
        }
    ));
    assert!(!c.contains::<TupleKeyed>());
}
