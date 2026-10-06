use super::finish_reason;
use crate::completion::FinishReason;
use crate::providers::openai::wire::{OPENAI, TOGETHER, ZAI};
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

/// A dialect states its own vocabulary on top of the shared one: Z.AI's
/// context-window stop is a length stop there and unknown elsewhere.
#[test]
fn a_dialect_states_its_own_finish_vocabulary() {
    assert_eq!(
        finish_reason("model_context_window_exceeded", &ZAI.quirks),
        FinishReason::Length
    );
    assert_eq!(
        finish_reason("sensitive", &ZAI.quirks),
        FinishReason::ContentFilter
    );
    assert_eq!(
        finish_reason("model_context_window_exceeded", &OPENAI.quirks),
        FinishReason::Other("model_context_window_exceeded".to_owned())
    );
}

/// Every dialect shares the stop spellings compatible servers use for a
/// natural end, Together's `eos` among them.
#[test]
fn compatible_servers_stop_spellings_are_shared() {
    for reason in ["stop", "end", "eos", "end_turn", "stop_sequence"] {
        for quirks in [&OPENAI.quirks, &TOGETHER.quirks, &ZAI.quirks] {
            assert_eq!(
                finish_reason(reason, quirks),
                FinishReason::Stop,
                "{reason}"
            );
        }
    }
    assert_eq!(
        finish_reason("MALFORMED", &OPENAI.quirks),
        FinishReason::Other("MALFORMED".to_owned())
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
