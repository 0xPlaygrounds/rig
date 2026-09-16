use super::*;

/// The two live shapes, and the catch-all that keeps a third from failing
/// the transcription. Recorded turns of the first two are replayed in
/// `transcription_usage_matrix`; this pins the decode itself, including
/// the shapes no live model produces.
#[test]
fn usage_decodes_both_billing_shapes_and_keeps_unknown_ones() {
    fn usage(body: &str) -> Option<TranscriptionUsage> {
        serde_json::from_str::<TranscriptionResponse>(body)
            .expect("response should decode")
            .usage
    }

    assert_eq!(
        usage(r#"{"text":"hi","usage":{"type":"duration","seconds":6}}"#),
        Some(TranscriptionUsage::Duration {
            r#type: DurationTag::Duration,
            seconds: 6.0
        })
    );
    assert_eq!(
        usage(
            r#"{"text":"hi","usage":{"type":"tokens","input_tokens":54,
                   "input_token_details":{"audio_tokens":54,"text_tokens":0},
                   "output_tokens":16,"total_tokens":70}}"#
        ),
        Some(TranscriptionUsage::Tokens {
            r#type: TokensTag::Tokens,
            input_tokens: 54,
            input_token_details: Some(TranscriptionInputTokenDetails {
                audio_tokens: 54,
                text_tokens: 0,
            }),
            output_tokens: 16,
            total_tokens: 70,
        })
    );
    // The breakdown is optional: a provider that omits it still decodes as
    // a token-billed turn rather than falling to the catch-all.
    assert_eq!(
        usage(
            r#"{"text":"hi","usage":{"type":"tokens","input_tokens":54,
                   "output_tokens":16,"total_tokens":70}}"#
        ),
        Some(TranscriptionUsage::Tokens {
            r#type: TokensTag::Tokens,
            input_tokens: 54,
            input_token_details: None,
            output_tokens: 16,
            total_tokens: 70,
        })
    );
    // The tag decides, not which optional keys are present: a token-billed
    // payload that also reported `seconds` must not decode as a duration
    // and drop every token count.
    assert!(matches!(
        usage(
            r#"{"text":"hi","usage":{"type":"tokens","seconds":6,"input_tokens":54,
                   "output_tokens":16,"total_tokens":70}}"#
        ),
        Some(TranscriptionUsage::Tokens {
            total_tokens: 70,
            ..
        })
    ));
    assert!(matches!(
        usage(r#"{"text":"hi","usage":{"type":"credits","spent":3}}"#),
        Some(TranscriptionUsage::Other(_))
    ));
    // A token-shaped payload missing a required total degrades to the
    // catch-all rather than failing the transcription.
    assert!(matches!(
        usage(r#"{"text":"hi","usage":{"type":"tokens","input_tokens":54}}"#),
        Some(TranscriptionUsage::Other(_))
    ));
    assert_eq!(usage(r#"{"text":"hi"}"#), None);
    assert_eq!(usage(r#"{"text":"hi","usage":null}"#), None);
}

/// The duration-billed shape reports no token counts, so normalization must
/// leave the usage empty rather than inventing zeros; the token-billed shape
/// carries its counts across.
#[test]
fn normalize_carries_token_billing_and_leaves_duration_billing_empty() {
    fn normalized(body: &str) -> crate::completion::Usage {
        serde_json::from_str::<TranscriptionResponse>(body)
            .expect("response should decode")
            .normalize("openai")
            .expect("normalization is infallible for this wire")
            .usage
    }

    let tokens = normalized(
        r#"{"text":"hi","usage":{"type":"tokens","input_tokens":54,
               "output_tokens":16,"total_tokens":70}}"#,
    );
    assert_eq!(tokens.input_tokens, Some(54));
    assert_eq!(tokens.output_tokens, Some(16));
    assert_eq!(tokens.total_tokens, Some(70));

    let duration = normalized(r#"{"text":"hi","usage":{"type":"duration","seconds":6}}"#);
    assert_eq!(duration.input_tokens, None);
    assert_eq!(duration.output_tokens, None);
    assert_eq!(duration.total_tokens, None);
}
