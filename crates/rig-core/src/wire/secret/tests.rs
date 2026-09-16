use super::Secret;

#[test]
fn a_secret_never_renders_or_serializes_its_value() {
    let secret = Secret::from("sk-live-do-not-print");
    assert_eq!(format!("{secret:?}"), "[redacted]");
    assert_eq!(
        serde_json::to_string(&secret).expect("a string serializes"),
        "\"[redacted]\""
    );
    assert_eq!(secret.expose(), "sk-live-do-not-print");
}

#[test]
fn secrets_compare_by_value() {
    assert_eq!(Secret::from("k"), Secret::from("k"));
    assert_ne!(Secret::from("k"), Secret::from("j"));
}

#[test]
fn a_secret_deserializes_from_a_bare_string() {
    let secret: Secret = serde_json::from_str("\"k\"").expect("a string deserializes");
    assert_eq!(secret.expose(), "k");
}
