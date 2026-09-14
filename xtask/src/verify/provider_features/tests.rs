use super::*;

fn diagnostic(message: &str, code: &str) -> Value {
    serde_json::json!({"reason":"compiler-message", "message":{"level":"error", "message":message, "code":{"code":code}}})
}

#[test]
fn unrelated_failure_cannot_pass_a_negative_probe() {
    let disabled = vec!["openai".into()];
    assert!(
        verify_missing_imports(
            &[diagnostic(
                "unresolved import `rig::providers::openai`",
                "E0432"
            )],
            &disabled,
            "rig"
        )
        .is_ok()
    );
    assert!(
        verify_missing_imports(
            &[diagnostic(
                "unresolved import `rig::providers::openai`",
                "E0433"
            )],
            &disabled,
            "rig"
        )
        .is_err()
    );
    assert!(
        verify_missing_imports(
            &[diagnostic(
                "unresolved import `rig::providers::gemini`",
                "E0432"
            )],
            &disabled,
            "rig"
        )
        .is_err()
    );
    assert!(verify_missing_imports(&[], &disabled, "rig").is_err());
    assert!(
        verify_missing_imports(
            &[
                diagnostic("unresolved import `rig::providers::openai`", "E0432"),
                diagnostic("unrelated type failure", "E0308"),
            ],
            &disabled,
            "rig"
        )
        .is_err()
    );
}
