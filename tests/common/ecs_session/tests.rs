//! Typed Native recovery, independent of the recorded provider answer.
use super::parse_native_output;
use serde::Deserialize;
#[derive(Debug, Deserialize, PartialEq)]
struct Answer {
    value: String,
}
#[test]
fn native_json_recovery_preserves_fences_prose_and_invalid_input_behavior() {
    for (input, expected) in [
        (r#"{"value":"plain"}"#, "plain"),
        ("```json\n{\"value\":\"fenced\"}\n```", "fenced"),
        (
            "Here is the answer: {\"value\":\"prose\"} trailing words",
            "prose",
        ),
    ] {
        assert_eq!(
            parse_native_output::<Answer>(input).unwrap().value,
            expected
        );
    }
    assert!(parse_native_output::<Answer>("no json here").is_err());
    assert!(parse_native_output::<Answer>("prefix {invalid} then {\"value\":\"later\"}").is_err());
    assert_eq!(
        parse_native_output::<Vec<u8>>("Array: [1,2] done").unwrap(),
        vec![1, 2]
    );
}
