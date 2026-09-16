use super::*;

#[test]
fn secret_debug_is_redacted() {
    let secret = Secret::new("sk-super-secret-key");
    assert_eq!(format!("{secret:?}"), "[redacted]");
}

#[test]
fn secret_serialize_is_redacted() {
    let secret = Secret::new("sk-super-secret-key");
    let json = serde_json::to_string(&secret).expect("serialize");
    assert_eq!(json, "\"[redacted]\"");
}

#[test]
fn secret_deserialize_preserves_value() {
    let json = "\"sk-super-secret-key\"";
    let secret: Secret = serde_json::from_str(json).expect("deserialize");
    assert_eq!(secret.expose_secret(), "sk-super-secret-key");
    assert_eq!(&*secret, "sk-super-secret-key");
}

#[test]
fn secret_equality_compares_values() {
    let s1 = Secret::new("key-a");
    let s2 = Secret::new("key-a");
    let s3 = Secret::new("key-b");
    assert_eq!(s1, s2);
    assert_ne!(s1, s3);
}
