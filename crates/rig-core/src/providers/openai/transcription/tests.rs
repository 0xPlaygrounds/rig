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
            .normalize()
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

/// Timings and speakers from each verbose shape normalize into the same
/// segments and words. The bodies are trimmed from live replies: OpenRouter
/// labels speakers by index, OpenAI by letter, and Mistral by name under
/// `speaker_id`, with untimed segments allowed.
#[test]
fn verbose_shapes_normalize_to_segments_words_and_speakers() {
    fn normalize(body: &str) -> transcription::TranscriptionResponse {
        serde_json::from_str::<TranscriptionResponse>(body)
            .expect("response should decode")
            .normalize()
            .expect("response should normalize")
    }
    fn segment(start: f64, end: f64, text: &str, speaker: Option<&str>) -> TranscriptSegment {
        TranscriptSegment {
            start,
            end,
            text: text.to_owned(),
            speaker: speaker.map(str::to_owned),
        }
    }

    let openrouter = normalize(
        r#"{"text":"Hello there! It was great.","usage":{"seconds":7,"cost":0.0002},
            "language":"en","duration":6.55,"confidence":0.93,
            "segments":[{"id":0,"start":0.12,"end":2,"text":"Hello there!","speaker":0},
                        {"id":1,"start":2.88,"end":5.96,"text":"It was great.","speaker":1}],
            "words":[{"word":"Hello","start":0.12,"end":0.32,"speaker":0,"confidence":0.9}]}"#,
    );
    assert_eq!(openrouter.language.as_deref(), Some("en"));
    assert_eq!(openrouter.duration, Some(6.55));
    assert_eq!(openrouter.confidence, Some(0.93));
    assert_eq!(
        openrouter.segments,
        vec![
            segment(0.12, 2.0, "Hello there!", Some("0")),
            segment(2.88, 5.96, "It was great.", Some("1")),
        ]
    );
    assert_eq!(
        openrouter.words,
        vec![TranscriptWord {
            word: "Hello".to_owned(),
            start: 0.12,
            end: 0.32,
            speaker: Some("0".to_owned()),
            confidence: Some(0.9),
        }]
    );

    let openai = normalize(
        r#"{"text":"Hello there!","task":"transcribe","duration":6.55,
            "segments":[{"type":"transcript.text.segment","text":" Hello there!",
                         "speaker":"A","start":0.0,"end":0.5,"id":"seg_0"}],
            "usage":{"type":"tokens","total_tokens":293,"input_tokens":65,"output_tokens":228}}"#,
    );
    assert_eq!(
        openai.segments,
        vec![segment(0.0, 0.5, " Hello there!", Some("A"))]
    );
    assert_eq!(openai.usage.total_tokens, Some(293));

    let mistral = normalize(
        r#"{"model":"voxtral-mini-latest","text":"Hello there. It was great.","language":null,
            "segments":[{"type":"transcription_segment","text":"Hello there.","start":0.0,"end":0.7,"speaker_id":"speaker_1"},
                        {"type":"transcription_segment","text":" It was great.","start":2.9,"end":4.2,"speaker_id":null},
                        {"type":"transcription_segment","text":" untimed","start":null,"end":null}],
            "usage":{"prompt_audio_seconds":6,"prompt_tokens":5,"total_tokens":438,"completion_tokens":58}}"#,
    );
    assert_eq!(mistral.language, None);
    assert_eq!(
        mistral.segments,
        vec![
            segment(0.0, 0.7, "Hello there.", Some("speaker_1")),
            segment(2.9, 4.2, " It was great.", None),
        ]
    );

    let plain = normalize(r#"{"text":"hi","segments":null,"words":null}"#);
    assert!(plain.segments.is_empty() && plain.words.is_empty());
}
