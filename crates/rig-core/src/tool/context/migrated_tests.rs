use super::*;

#[test]
fn insert_and_get_returns_value() {
    let mut c = ToolContext::new();
    assert_eq!(c.insert(42u32).unwrap(), None);
    assert_eq!(c.get::<u32>(), Some(42));
}
#[test]
fn get_missing_type_returns_none() {
    assert_eq!(ToolContext::new().get::<u32>(), None);
}
#[test]
fn insert_overwrites_and_returns_previous() {
    let mut c = ToolContext::new();
    c.insert(1u32).unwrap();
    assert_eq!(c.insert(2u32).unwrap(), Some(1));
    assert_eq!(c.get::<u32>(), Some(2));
}
#[test]
fn different_types_are_independent() {
    let mut c = ToolContext::new();
    c.insert(42u32).unwrap();
    c.insert("hello".to_string()).unwrap();
    assert_eq!(c.get::<u32>(), Some(42));
    assert_eq!(c.get::<String>().as_deref(), Some("hello"));
}
#[test]
fn contains_tracks_types() {
    let mut c = ToolContext::new();
    c.insert(42u32).unwrap();
    assert!(c.contains::<u32>());
    assert!(!c.contains::<String>());
}
#[test]
fn clone_produces_independent_copy() {
    let mut c = ToolContext::new();
    c.insert(42u32).unwrap();
    let mut clone = c.clone();
    clone.insert(99u32).unwrap();
    assert_eq!(c.get::<u32>(), Some(42));
    assert_eq!(clone.get::<u32>(), Some(99));
}
#[test]
fn clone_deep_copies_heap_values() {
    let mut c = ToolContext::new();
    c.insert(vec![1u8, 2, 3]).unwrap();
    let mut clone = c.clone();
    let mut bytes = clone.remove::<Vec<u8>>().unwrap();
    bytes.push(4);
    clone.insert(bytes).unwrap();
    assert_eq!(c.get::<Vec<u8>>(), Some(vec![1, 2, 3]));
    assert_eq!(clone.get::<Vec<u8>>(), Some(vec![1, 2, 3, 4]));
}
#[test]
fn empty_context_is_default_and_serializes_empty() {
    let c = ToolContext::default();
    assert!(!c.contains::<u32>());
    assert!(c.is_empty());
    assert_eq!(serde_json::to_value(&c).unwrap(), serde_json::json!({}));
}
#[test]
fn reinsert_replaces_in_place() {
    let mut c = ToolContext::new();
    c.insert(42u32).unwrap();
    assert_eq!(c.insert(99u32).unwrap(), Some(42));
    assert_eq!(c.get::<u32>(), Some(99));
}
#[test]
fn remove_returns_value_and_clears_entry() {
    let mut c = ToolContext::new();
    c.insert(42u32).unwrap();
    assert_eq!(c.remove::<u32>(), Some(42));
    assert!(!c.contains::<u32>());
}
#[test]
fn remove_missing_type_returns_none() {
    assert_eq!(ToolContext::new().remove::<u32>(), None);
}
#[test]
fn require_present_returns_value() {
    let mut c = ToolContext::new();
    c.insert(42u32).unwrap();
    assert_eq!(c.require::<u32>(), Ok(42));
}
#[test]
fn require_missing_names_type() {
    let e = ToolContext::new().require::<u32>().unwrap_err();
    assert!(e.to_string().contains("u32"));
}
#[test]
fn result_metadata_round_trips_and_requires() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Id(u32);
    let mut c = ToolContext::new();
    c.insert_result(Id(7)).unwrap();
    assert_eq!(c.result::<Id>(), Some(Id(7)));
    assert_eq!(c.require_result::<Id>(), Ok(Id(7)));
    assert!(c.get::<Id>().is_none());
}
#[test]
fn debug_reports_types_without_values() {
    #[derive(Serialize, Deserialize)]
    struct Secret(String);
    let mut c = ToolContext::new();
    c.insert(42u32).unwrap();
    c.insert_result(Secret("do-not-print".to_string())).unwrap();
    let d = format!("{c:?}");
    assert!(d.contains("u32"));
    assert!(d.contains("Secret"));
    assert!(!d.contains("do-not-print"));
    assert_eq!(
        c.result::<Secret>().map(|s| s.0).as_deref(),
        Some("do-not-print")
    );
}
#[test]
fn dispatch_snapshot_isolates_inbound_and_publishes_only_result_metadata() {
    let mut c = ToolContext::new();
    c.insert(7u32).unwrap();
    c.insert_result("old".to_string()).unwrap();
    let mut d = c.for_dispatch();
    assert_eq!(d.get::<u32>(), Some(7));
    assert!(d.result::<String>().is_none());
    d.insert(8u32).unwrap();
    d.insert_result("new".to_string()).unwrap();

    c.accept_dispatch_result(d);
    assert_eq!(c.get::<u32>(), Some(7));
    assert_eq!(c.result::<String>().as_deref(), Some("new"));
}
#[test]
fn many_distinct_types_round_trip() {
    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct A(u8);
    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct B(u16);
    let mut c = ToolContext::new();
    c.insert(A(1)).unwrap();
    c.insert(B(2)).unwrap();
    c.insert(3u32).unwrap();
    c.insert("four".to_string()).unwrap();
    assert_eq!(c.get::<A>(), Some(A(1)));
    assert_eq!(c.get::<B>(), Some(B(2)));
    assert_eq!(c.get::<u32>(), Some(3));
    assert_eq!(c.get::<String>().as_deref(), Some("four"));
}
#[test]
fn unencodable_value_is_an_error_not_a_panic() {
    // A map with non-string keys has no JSON form.
    let mut c = ToolContext::new();
    let value: std::collections::HashMap<(u8, u8), u8> = [((1, 2), 3)].into_iter().collect();
    let err = c.insert(value).unwrap_err();
    assert!(matches!(err, ToolContextError::Encode { .. }));
    assert!(!c.contains::<std::collections::HashMap<(u8, u8), u8>>());
}
