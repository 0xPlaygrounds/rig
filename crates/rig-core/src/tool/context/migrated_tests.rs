use super::*;

#[test]
fn insert_and_get_returns_value() {
    let mut c = ToolContext::new();
    assert_eq!(c.insert(42u32), None);
    assert_eq!(c.get::<u32>(), Some(&42));
}
#[test]
fn get_missing_type_returns_none() {
    assert_eq!(ToolContext::new().get::<u32>(), None);
}
#[test]
fn insert_overwrites_and_returns_previous() {
    let mut c = ToolContext::new();
    c.insert(1u32);
    assert_eq!(c.insert(2u32), Some(1));
    assert_eq!(c.get::<u32>(), Some(&2));
}
#[test]
fn different_types_are_independent() {
    let mut c = ToolContext::new();
    c.insert(42u32);
    c.insert("hello".to_string());
    assert_eq!(c.get::<u32>(), Some(&42));
    assert_eq!(c.get::<String>().map(String::as_str), Some("hello"));
}
#[test]
fn contains_tracks_types() {
    let mut c = ToolContext::new();
    c.insert(42u32);
    assert!(c.contains::<u32>());
    assert!(!c.contains::<String>());
}
#[test]
fn clone_produces_independent_copy() {
    let mut c = ToolContext::new();
    c.insert(42u32);
    let mut clone = c.clone();
    clone.insert(99u32);
    assert_eq!(c.get::<u32>(), Some(&42));
    assert_eq!(clone.get::<u32>(), Some(&99));
}
#[test]
fn clone_deep_copies_heap_values() {
    let mut c = ToolContext::new();
    c.insert(vec![1u8, 2, 3]);
    let mut clone = c.clone();
    clone.get_mut::<Vec<u8>>().unwrap().push(4);
    assert_eq!(c.get::<Vec<u8>>(), Some(&vec![1, 2, 3]));
    assert_eq!(clone.get::<Vec<u8>>(), Some(&vec![1, 2, 3, 4]));
}
#[test]
fn clone_preserves_intentionally_shared_value_state() {
    let shared = std::sync::Arc::new(std::sync::Mutex::new(1_u32));
    let mut context = ToolContext::new();
    context.insert(shared.clone());

    let snapshot = context.for_dispatch();
    *snapshot
        .get::<std::sync::Arc<std::sync::Mutex<u32>>>()
        .expect("shared value")
        .lock()
        .expect("shared value lock") = 2;

    assert_eq!(*shared.lock().expect("shared value lock"), 2);
    assert!(context.contains::<std::sync::Arc<std::sync::Mutex<u32>>>());
}
#[test]
fn empty_context_is_default_and_allocation_free() {
    let c = ToolContext::default();
    assert!(!c.contains::<u32>());
    assert!(c.inbound.map.is_none());
    assert!(c.result.map.is_none());
}
#[test]
fn get_mut_modifies_in_place() {
    let mut c = ToolContext::new();
    c.insert(42u32);
    *c.get_mut::<u32>().unwrap() = 99;
    assert_eq!(c.get::<u32>(), Some(&99));
}
#[test]
fn remove_returns_value_and_clears_entry() {
    let mut c = ToolContext::new();
    c.insert(42u32);
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
    c.insert(42u32);
    assert_eq!(c.require::<u32>().copied(), Ok(42));
}
#[test]
fn require_missing_names_type() {
    let e = ToolContext::new().require::<u32>().unwrap_err();
    assert!(e.to_string().contains("u32"));
}
#[test]
fn result_metadata_round_trips_and_requires() {
    #[derive(Clone, Debug, PartialEq)]
    struct Id(u32);
    let mut c = ToolContext::new();
    c.insert_result(Id(7));
    assert_eq!(c.result::<Id>(), Some(&Id(7)));
    assert_eq!(c.require_result::<Id>(), Ok(&Id(7)));
    assert!(c.get::<Id>().is_none());
}
#[test]
fn debug_reports_types_without_values() {
    #[derive(Clone)]
    struct Secret(&'static str);
    let mut c = ToolContext::new();
    c.insert(42u32);
    c.insert_result(Secret("do-not-print"));
    let d = format!("{c:?}");
    assert!(d.contains("u32"));
    assert!(d.contains("Secret"));
    assert!(!d.contains("do-not-print"));
    assert_eq!(c.result::<Secret>().map(|s| s.0), Some("do-not-print"));
}
#[test]
fn dispatch_snapshot_isolates_inbound_and_publishes_only_result_metadata() {
    let mut c = ToolContext::new();
    c.insert(7u32);
    c.insert_result("old".to_string());
    let mut d = c.for_dispatch();
    assert_eq!(d.get::<u32>(), Some(&7));
    assert!(d.result::<String>().is_none());
    *d.get_mut::<u32>().expect("snapshot value") = 8;
    d.insert_result("new".to_string());

    c.accept_dispatch_result(d);
    assert_eq!(c.get::<u32>(), Some(&7));
    assert_eq!(c.result::<String>().map(String::as_str), Some("new"));
}
#[test]
fn many_distinct_types_round_trip_through_type_id_hasher() {
    #[derive(Clone, PartialEq, Debug)]
    struct A(u8);
    #[derive(Clone, PartialEq, Debug)]
    struct B(u16);
    let mut c = ToolContext::new();
    c.insert(A(1));
    c.insert(B(2));
    c.insert(3u32);
    c.insert("four".to_string());
    assert_eq!(c.get::<A>(), Some(&A(1)));
    assert_eq!(c.get::<B>(), Some(&B(2)));
    assert_eq!(c.get::<u32>(), Some(&3));
    assert_eq!(c.get::<String>().map(String::as_str), Some("four"));
}
