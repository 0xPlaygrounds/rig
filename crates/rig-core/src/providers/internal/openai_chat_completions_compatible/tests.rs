use super::map_openai_finish_reason;
use crate::completion::FinishReason;

#[test]
fn truncated_output_covers_only_the_cut_short_reasons() {
    assert!(FinishReason::Length.truncated_output());
    assert!(FinishReason::ContentFilter.truncated_output());
    assert!(!FinishReason::Stop.truncated_output());
    assert!(!FinishReason::ToolCalls.truncated_output());
    assert!(!FinishReason::Other("whatever".to_owned()).truncated_output());
}

#[test]
fn sse_error_detector_handles_null_empty_and_object_or_string_errors() {
    use super::provider_response_from_compatible_sse_data as detect;

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

/// Mistral truncates at its context ceiling with `model_length`, which is
/// the same truncation class as `length` — only the limit differs.
///
/// Not a cassette test: forcing the state needs a prompt padded to the
/// model's full context window, which would commit a ~145 KB fixture of
/// repeated filler to exercise one mapping arm. The shape below is the
/// live response recorded while confirming the bug against
/// `voxtral-small-latest` (`max_context_length` 32768):
/// `finish_reason: "model_length"` with
/// `usage {prompt_tokens: 32424, completion_tokens: 344, total_tokens: 32768}`
/// — generation stopped dead on the ceiling with 4096 output tokens still
/// budgeted.
#[test]
fn model_length_is_truncation_not_a_natural_stop() {
    assert_eq!(
        map_openai_finish_reason("model_length"),
        FinishReason::Length,
        "a turn cut off by the context window must be distinguishable from one that \
             simply had nothing more to say"
    );

    // The vocabulary it joins, and the fallback that still preserves an
    // unrecognized spelling verbatim.
    assert_eq!(map_openai_finish_reason("length"), FinishReason::Length);
    assert_eq!(map_openai_finish_reason("max_tokens"), FinishReason::Length);
    assert_eq!(map_openai_finish_reason("stop"), FinishReason::Stop);
    assert_eq!(
        map_openai_finish_reason("some_new_reason"),
        FinishReason::Other("some_new_reason".to_owned())
    );
}
