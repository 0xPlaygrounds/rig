use crate::providers::openai::client::ApiResponse;

#[derive(Debug, serde::Deserialize)]
struct Success {
    #[allow(dead_code)]
    text: String,
}

fn classify(body: &str) -> String {
    match serde_json::from_str::<ApiResponse<Success>>(body).expect("body must decode") {
        ApiResponse::Err(error) => error.message,
        ApiResponse::Ok(_) => panic!("error body must classify as the error envelope"),
    }
}

/// A body carrying BOTH `message` and `error` must still classify as the
/// error envelope (a field-level `alias = "error"` rejected it as a
/// duplicate field), with the canonical `error` object winning.
#[test]
fn dual_message_and_error_keys_classify_as_the_error_envelope() {
    assert_eq!(
        classify(r#"{"message":"quota exceeded","error":{"code":"429"}}"#),
        r#"{"code":"429"}"#
    );
}

#[test]
fn null_error_key_falls_back_to_message() {
    assert_eq!(
        classify(r#"{"error":null,"message":"over capacity"}"#),
        "over capacity"
    );
}
