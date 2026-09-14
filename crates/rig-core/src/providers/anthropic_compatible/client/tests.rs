use super::*;

#[test]
fn key_debug_redacts_the_credential_but_authentication_preserves_it() {
    let key = AnthropicKey::from("test-secret-not-for-debug");
    assert_eq!(format!("{key:?}"), "AnthropicKey(<redacted>)");
    let (name, value) = key.into_header().unwrap().unwrap();
    assert_eq!(name, "x-api-key");
    assert_eq!(value, "test-secret-not-for-debug");
}
