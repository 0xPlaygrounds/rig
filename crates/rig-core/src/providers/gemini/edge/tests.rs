use super::*;

#[test]
fn usage_follows_the_one_gemini_rule() {
    let usage = usage(Counts {
        prompt: Some(1_000),
        tool_use_prompt: Some(200),
        cached: Some(800),
        candidates: Some(40),
        thoughts: Some(60),
    });
    assert_eq!(usage.input_tokens, Some(1_200));
    assert_eq!(usage.output_tokens, Some(100));
    assert_eq!(usage.total_tokens, Some(1_300));
    assert_eq!(usage.cached_input_tokens, Some(800));
    assert_eq!(usage.reasoning_tokens, Some(60));
    let (Some(input), Some(output), Some(cached), Some(reasoning)) = (
        usage.input_tokens,
        usage.output_tokens,
        usage.cached_input_tokens,
        usage.reasoning_tokens,
    ) else {
        panic!("every counter is reported");
    };
    assert!(cached <= input);
    assert!(reasoning <= output);
    assert_eq!(usage.total_tokens, Some(input + output));
}

#[test]
fn unreported_counts_stay_unreported() {
    assert_eq!(usage(Counts::default()), Usage::default());
    let prompt_only = usage(Counts {
        prompt: Some(5),
        ..Counts::default()
    });
    assert_eq!(prompt_only.input_tokens, Some(5));
    assert_eq!(prompt_only.output_tokens, None);
    assert_eq!(prompt_only.total_tokens, Some(5));
}

#[test]
fn finish_reasons_map_to_rigs_vocabulary() {
    assert_eq!(finish_reason("FINISH_REASON_UNSPECIFIED"), None);
    assert_eq!(finish_reason("STOP"), Some(FinishReason::Stop));
    assert_eq!(finish_reason("MAX_TOKENS"), Some(FinishReason::Length));
    assert_eq!(finish_reason("SPII"), Some(FinishReason::ContentFilter));
    assert_eq!(
        finish_reason("RECITATION"),
        Some(FinishReason::Other("RECITATION".into()))
    );
    assert!(tool_protocol_error("MISSING_THOUGHT_SIGNATURE", None).is_some());
    assert!(tool_protocol_error("STOP", None).is_none());
}

#[test]
fn a_tool_call_without_object_arguments_is_refused() {
    let call = ToolCall::from_wire(
        "c1",
        ToolFunction::new(ToolName::new("f").expect("name"), Value::String("x".into())),
    );
    assert!(assistant_units(AssistantContent::ToolCall(call), &ISSUER).is_err());
}

#[test]
fn only_geminis_own_signatures_are_replayed() {
    let text = message::Text {
        signature: Some(Signature::sealed("anthropic", "Zm9yZWlnbg==")),
        ..message::Text::new("hi")
    };
    assert_eq!(
        assistant_units(AssistantContent::Text(text), &ISSUER).expect("units"),
        [Unit::Text {
            text: "hi".into(),
            signature: None
        }]
    );
}

/// The drift test: the same model-turn units, encoded by either dialect and
/// read back, are the same units. A row that fails names the dialect and the
/// unit that drifted.
mod drift {
    use serde_json::json;

    use super::*;
    use crate::providers::gemini::generate_content::GenerateContentDialect;
    use crate::providers::gemini::interactions_api::InteractionsDialect;

    fn text(text: &str) -> Unit {
        Unit::Text {
            text: text.into(),
            signature: None,
        }
    }

    fn thought(text: &str, signature: Option<&str>) -> Unit {
        Unit::Thought {
            text: text.into(),
            signature: signature.map(str::to_owned),
        }
    }

    fn call(id: &str, signature: Option<&str>) -> Unit {
        Unit::Call {
            id: Some(id.into()),
            name: "lookup".into(),
            args: serde_json::Map::from_iter([("q".to_owned(), json!("a"))]),
            signature: signature.map(str::to_owned),
        }
    }

    fn rows() -> Vec<(&'static str, Vec<Unit>)> {
        vec![
            ("text", vec![text("hello")]),
            (
                "thought then call",
                vec![thought("plan", Some("c2ln")), call("c1", None)],
            ),
            (
                "signature-only thought",
                vec![thought("", Some("c2ln")), call("c1", None), text("done")],
            ),
            ("signed call", vec![call("c1", Some("Y2FsbA=="))]),
            (
                "interleaved",
                vec![text("a"), thought("b", None), text("c"), call("c2", None)],
            ),
        ]
    }

    fn through<D: Dialect>(units: &[Unit]) -> Vec<Unit> {
        units
            .iter()
            .cloned()
            .flat_map(|unit| {
                let encoded = D::part(unit).expect("encodes");
                let raw = serde_json::value::to_raw_value(&encoded).expect("serializes");
                D::units(&raw).expect("reads back")
            })
            .collect()
    }

    #[test]
    fn both_dialects_read_back_what_they_write() {
        for (row, units) in rows() {
            assert_eq!(
                through::<GenerateContentDialect>(&units),
                units,
                "GenerateContent drifted on {row}"
            );
            assert_eq!(
                through::<InteractionsDialect>(&units),
                units,
                "Interactions drifted on {row}"
            );
        }
    }

    #[test]
    fn both_dialects_encode_a_result_and_media() {
        let result = Unit::Result {
            id: Some("c1".into()),
            name: "lookup".into(),
            response: Some(serde_json::Map::from_iter([(
                "result".to_owned(),
                json!({"status": "ok"}),
            )])),
            media: Vec::new(),
        };
        let generate =
            serde_json::to_value(GenerateContentDialect::part(result.clone()).expect("encodes"))
                .expect("JSON");
        let interactions =
            serde_json::to_value(InteractionsDialect::part(result).expect("encodes"))
                .expect("JSON");
        assert_eq!(
            generate["functionResponse"]["response"]["result"],
            json!({"status": "ok"})
        );
        assert_eq!(interactions["result"], json!({"status": "ok"}));
        assert_eq!(generate["functionResponse"]["id"], interactions["call_id"]);

        let media = Unit::Media(Media {
            mime_type: Some("image/png".into()),
            source: Source::Inline("aGk=".into()),
            detail: Some(MediaDetail::High),
        });
        let generate =
            serde_json::to_value(GenerateContentDialect::part(media.clone()).expect("encodes"))
                .expect("JSON");
        let interactions =
            serde_json::to_value(InteractionsDialect::part(media).expect("encodes")).expect("JSON");
        assert_eq!(generate["inlineData"]["data"], interactions["data"]);
        assert_eq!(
            generate["mediaResolution"]["level"],
            "MEDIA_RESOLUTION_HIGH"
        );
        assert_eq!(interactions["resolution"], "high");
    }

    #[test]
    fn a_native_part_of_one_dialect_is_refused_by_the_other() {
        let generate = Unit::Native(NativePart::new(
            crate::providers::gemini::api::PART_SCHEMA,
            serde_json::value::RawValue::from_string(r#"{"toolCall":{}}"#.to_owned()).expect("raw"),
        ));
        assert!(InteractionsDialect::part(generate).is_err());
        let step = Unit::Native(NativePart::new(
            crate::providers::gemini::interactions_api::api::STEP_SCHEMA,
            serde_json::value::RawValue::from_string(
                r#"{"type":"code_execution_call"}"#.to_owned(),
            )
            .expect("raw"),
        ));
        assert!(GenerateContentDialect::part(step).is_err());
    }
}
