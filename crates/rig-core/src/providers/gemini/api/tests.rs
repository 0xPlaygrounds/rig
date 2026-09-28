use serde_json::json;

use super::*;

#[test]
fn unmodeled_accepts_a_field_the_mirror_does_not_type() {
    let settings = Unmodeled::<GenerationSettings>::new()
        .with("responseVerbosity", json!("LOW"))
        .expect("an untyped field");
    assert_eq!(settings.get("responseVerbosity"), Some(&json!("LOW")));
}

#[test]
fn unmodeled_refuses_a_typed_field_in_either_spelling() {
    for key in ["thinkingConfig", "thinking_config"] {
        let refused = Unmodeled::<GenerationSettings>::new().with(key, json!({}));
        assert!(
            matches!(
                refused,
                Err(UnmodeledError::Modeled {
                    mirror: "GenerationSettings",
                    ..
                })
            ),
            "{key} reaches a typed field"
        );
    }
}

#[test]
fn unmodeled_refuses_a_field_rig_owns_in_either_spelling() {
    for key in [
        "maxOutputTokens",
        "max_output_tokens",
        "temperature",
        "responseJsonSchema",
    ] {
        assert!(
            Unmodeled::<GenerationSettings>::new()
                .with(key, json!(1))
                .is_err(),
            "{key} is owned by rig"
        );
    }
    for key in [
        "includeServerSideToolInvocations",
        "include_server_side_tool_invocations",
    ] {
        assert!(
            Unmodeled::<ToolConfigSettings>::new()
                .with(key, json!(true))
                .is_err(),
            "{key} is owned by rig"
        );
    }
    for key in [
        "systemInstruction",
        "system_instruction",
        "cachedContent",
        "contents",
    ] {
        assert!(
            Unmodeled::<RequestSettings>::new()
                .with(key, json!({}))
                .is_err(),
            "{key} is owned by rig"
        );
    }
    assert!(
        Unmodeled::<HostedTool>::new()
            .with("function_declarations", json!([]))
            .is_err()
    );
}

#[test]
fn the_error_names_the_field_and_its_mirror() {
    let error = Unmodeled::<GenerationSettings>::new()
        .with("thinkingConfig", json!({}))
        .expect_err("a typed field");
    assert_eq!(
        error.to_string(),
        "`thinkingConfig` is a typed field of GenerationSettings"
    );
}

#[test]
fn a_response_with_unknown_fields_and_values_re_serializes_unchanged() {
    let body = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"text": "hi", "thoughtSignature": "c2ln"},
                    {"toolCall": {"toolType": "GOOGLE_SEARCH_WEB", "id": "a1", "args": {"queries": ["q"]}}},
                    {"futurePart": {"x": 1}}
                ]
            },
            "finishReason": "SOME_NEW_REASON",
            "index": 0
        }],
        "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 5, "newCounter": 1},
        "modelVersion": "gemini-3.8-flash",
        "responseId": "r1"
    });
    let response: GenerateContentResponse =
        serde_json::from_value(body.clone()).expect("the mirror reads it");
    assert_eq!(serde_json::to_value(&response).expect("serialize"), body);
    let mut unmodeled = Vec::new();
    response.unmodeled_fields("$", &mut unmodeled);
    assert_eq!(
        unmodeled,
        [
            "$.candidates[0].content.parts[2].futurePart",
            "$.usageMetadata.newCounter"
        ]
    );
    let candidate = response.candidates.first().expect("a candidate");
    assert_eq!(
        candidate.finish_reason,
        Some(FinishReason::Unknown("SOME_NEW_REASON".into()))
    );
}

#[test]
fn a_native_part_reads_as_a_part_for_its_issuer_only() {
    let json =
        r#"{"executableCode":{"language":"PYTHON","code":"print(1)"},"thoughtSignature":"c2ln"}"#;
    let native = |issuer: &'static str, schema: &'static str| {
        Sealed::new(
            issuer,
            NativePart::new(
                schema,
                serde_json::value::RawValue::from_string(json.to_owned()).expect("raw"),
            ),
        )
    };
    let part = Part::try_from(&native(super::super::PROVIDER_NAME, PART_SCHEMA)).expect("a part");
    assert_eq!(
        part.executable_code.and_then(|code| code.code).as_deref(),
        Some("print(1)")
    );
    assert!(matches!(
        Part::try_from(&native("anthropic", PART_SCHEMA)),
        Err(NativeError::OtherIssuer(_))
    ));
    assert!(matches!(
        Part::try_from(&native(super::super::PROVIDER_NAME, "other.Step")),
        Err(NativeError::OtherSchema(_))
    ));
}

#[test]
fn settings_templates_serialize_only_what_is_set() {
    let settings = RequestSettings {
        generation_config: GenerationSettings {
            media_resolution: Some(MediaResolution::High),
            ..Default::default()
        },
        service_tier: Some(ServiceTier::Flex),
        ..Default::default()
    };
    assert_eq!(
        serde_json::to_value(&settings).expect("serialize"),
        json!({
            "generationConfig": {"mediaResolution": "MEDIA_RESOLUTION_HIGH"},
            "serviceTier": "flex"
        })
    );
}
