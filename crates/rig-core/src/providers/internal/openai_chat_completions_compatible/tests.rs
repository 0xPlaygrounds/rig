use super::finish_reason;
use crate::completion::FinishReason;
use crate::providers::openai::wire::{OPENAI, TOGETHER};
use bytes::Bytes;

#[test]
fn sse_error_detector_handles_null_empty_and_object_or_string_errors() {
    use super::provider_error_envelope as detect;

    // An empty `error` (`null` or `""`) with no choices must NOT terminate the
    // stream — some providers send one with the terminal usage event. Each of
    // these should be treated as "not an error chunk".
    assert!(detect(r#"{"error":null}"#).is_none());
    assert!(detect(r#"{"error":null,"usage":{"total_tokens":3}}"#).is_none());
    assert!(detect(r#"{"error":""}"#).is_none());
    // A normal content chunk (no `error` key) is also not an error.
    assert!(detect(r#"{"choices":[{"delta":{"content":"hi"}}]}"#).is_none());
    // A live content chunk that ALSO carries an `error` field must NOT terminate
    // the stream — the `choices` guard wins regardless of the error value.
    assert!(detect(r#"{"error":"metadata","choices":[{"delta":{"content":"hi"}}]}"#).is_none());
    assert!(
        detect(r#"{"error":{"message":"x"},"choices":[{"delta":{"content":"hi"}}]}"#).is_none()
    );

    // A non-empty string `error` IS detected, preserving the raw body.
    let string_body = r#"{"error":"oops"}"#;
    let string_error = detect(string_body).expect("string error should be detected");
    assert_eq!(string_error.provider_response_body(), Some(string_body));
    assert_eq!(string_error.provider_response_status(), None);

    // A real provider error envelope IS detected, preserving the raw body.
    let body = r#"{"error":{"message":"rate limited","type":"rate_limit_error"}}"#;
    let error = detect(body).expect("object error envelope should be detected");
    assert_eq!(error.provider_response_body(), Some(body));
    // It arrives mid-stream with no HTTP status attached.
    assert_eq!(error.provider_response_status(), None);

    // The choices guard is narrowed to a NON-EMPTY array: an error body
    // that also carries `"choices":[]` (or `null`) is still an error —
    // pre-#2258-B6 it classified as a normal chunk, and a following
    // `[DONE]` committed the failed turn as a successful zero-usage
    // completion.
    let masked = r#"{"error":{"message":"rate limited"},"choices":[]}"#;
    let error = detect(masked).expect("an empty choices array must not mask the error");
    assert_eq!(error.provider_response_body(), Some(masked));
    assert!(
        detect(r#"{"error":{"message":"rate limited"},"choices":null}"#).is_some(),
        "a null choices value must not mask the error"
    );
}

/// NEW-4 (chat): Together ends a successful turn with `eos`, in its
/// documented vocabulary; another dialect never named it, so it fails there.
#[test]
fn a_dialect_states_its_own_finish_vocabulary() {
    assert_eq!(finish_reason("eos", &TOGETHER.quirks), FinishReason::Stop);
    assert_eq!(
        finish_reason("eos", &OPENAI.quirks),
        FinishReason::Other("eos".to_owned())
    );
}

pub(crate) fn sse_bytes_from_data_lines<T>(events: impl IntoIterator<Item = T>) -> Bytes
where
    T: AsRef<str>,
{
    Bytes::from(
        events
            .into_iter()
            .map(|event| format!("data: {}\n\n", event.as_ref()))
            .collect::<String>(),
    )
}

pub(crate) fn sse_bytes_from_json_events(events: &[serde_json::Value]) -> Bytes {
    Bytes::from(
        events
            .iter()
            .map(|event| {
                format!(
                    "data: {}\n\n",
                    serde_json::to_string(event).expect("event should serialize")
                )
            })
            .collect::<String>(),
    )
}
