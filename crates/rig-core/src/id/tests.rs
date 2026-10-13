use super::*;

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
