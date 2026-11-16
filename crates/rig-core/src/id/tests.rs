use super::*;

#[test]
fn run_ids_are_unique_increasing_and_read_back() {
    let a = RunId::new();
    let b = RunId::new();
    assert_ne!(a, b);
    assert!(b > a);
    let id = RunId(NonZeroU64::MIN.saturating_add(41));
    assert_eq!(id.to_string(), "42");
    assert_eq!(format!("{id:?}"), "RunId(42)");
    assert_eq!(serde_json::to_string(&id).expect("serialize"), "42");
    assert_eq!(
        serde_json::from_str::<RunId>("42").expect("deserialize"),
        id
    );
    assert!(serde_json::from_str::<RunId>("0").is_err());
}
