use super::*;

#[test]
fn conversation_id_round_trips_and_displays_transparently() {
    let id = ConversationId::from("thread-1");
    assert_eq!(id.as_str(), "thread-1");
    assert_eq!(id.to_string(), "thread-1");
    assert_eq!(
        serde_json::to_string(&id).expect("serialize"),
        "\"thread-1\""
    );
    assert_eq!(
        serde_json::from_str::<ConversationId>("\"thread-1\"").expect("deserialize"),
        id
    );
    assert_eq!(ConversationId::new(String::from("thread-1")), id);
}

#[test]
fn run_ids_are_unique_increasing_and_round_trip() {
    let a = RunId::new();
    let b = RunId::new();
    assert_ne!(a, b);
    assert!(b > a);
    assert_eq!(RunId::from_raw(a.to_raw()), Some(a));
    assert_eq!(RunId::from_raw(0), None);
    let id = RunId::from_raw(42).expect("non-zero");
    assert_eq!(id.to_string(), "42");
    assert_eq!("42".parse::<RunId>(), Ok(id));
    assert!("0".parse::<RunId>().is_err());
    assert_eq!(format!("{id:?}"), "RunId(42)");
    assert_eq!(serde_json::to_string(&id).expect("serialize"), "42");
    assert_eq!(
        serde_json::from_str::<RunId>("42").expect("deserialize"),
        id
    );
}
