use serde::{Deserialize, Serialize};

use super::{AdditionalParams, Message, Reasoning, ReasoningContent, Text, ToolResultContent};

mod vec_content_serde {
    use super::super::{AssistantContent, Message, UserContent};

    #[test]
    fn message_content_still_serializes_as_a_plain_sequence() {
        // The removed container serialized as a bare sequence, which is why
        // this migration changes no persisted history and no recorded
        // provider fixture. Pin the wire shape so that stays true.
        let message = Message::User {
            content: vec![UserContent::text("hi")],
        };
        let json = serde_json::to_value(&message).expect("serialize");
        assert_eq!(
            json,
            serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            })
        );
    }

    #[test]
    fn message_content_round_trips_byte_identically() {
        let message = Message::Assistant {
            id: Some("msg_1".to_owned()),
            content: vec![AssistantContent::text("hello")],
        };
        let encoded = serde_json::to_string(&message).expect("serialize");
        let decoded: Message = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(
            serde_json::to_string(&decoded).expect("re-serialize"),
            encoded
        );
    }

    #[test]
    fn an_empty_content_array_now_deserializes() {
        // The container's `Deserialize` implemented only `visit_seq` and
        // rejected `[]`. That is the single input whose behaviour this
        // migration changes: it was an error, and it is now an empty list.
        let message: Message =
            serde_json::from_value(serde_json::json!({"role": "user", "content": []}))
                .expect("an empty content list is representable now");
        let Message::User { content } = message else {
            panic!("expected a user message");
        };
        assert!(content.is_empty());
    }
}

#[test]
fn reasoning_constructors_and_accessors_work() {
    let single = Reasoning::new("think");
    assert_eq!(single.first_text(), Some("think"));
    assert_eq!(single.first_signature(), None);

    let signed = Reasoning::new_with_signature("signed", Some("sig-1".to_string()));
    assert_eq!(signed.first_text(), Some("signed"));
    assert_eq!(signed.first_signature(), Some("sig-1"));

    let multi = Reasoning::multi(vec!["a".to_string(), "b".to_string()]);
    assert_eq!(multi.display_text(), "a\nb");
    assert_eq!(multi.first_text(), Some("a"));

    let redacted = Reasoning::redacted("redacted-value");
    assert_eq!(redacted.display_text(), "redacted-value");
    assert_eq!(redacted.first_text(), None);

    let encrypted = Reasoning::encrypted("enc");
    assert_eq!(encrypted.encrypted_content(), Some("enc"));
    assert_eq!(encrypted.display_text(), "");

    let summaries = Reasoning::summaries(vec!["s1".to_string(), "s2".to_string()]);
    assert_eq!(summaries.display_text(), "s1\ns2");
    assert_eq!(summaries.encrypted_content(), None);
}

#[test]
fn reasoning_content_serde_roundtrip() {
    let variants = vec![
        ReasoningContent::Text {
            text: "plain".to_string(),
            signature: Some("sig".to_string()),
        },
        ReasoningContent::Encrypted("opaque".to_string()),
        ReasoningContent::Redacted {
            data: "redacted".to_string(),
        },
        ReasoningContent::Summary("summary".to_string()),
    ];

    for variant in variants {
        let json = serde_json::to_string(&variant).expect("serialize");
        let roundtrip: ReasoningContent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(roundtrip, variant);
    }
}

#[test]
fn system_message_constructor_and_serde_roundtrip() {
    let message = Message::system("You are concise.");

    match &message {
        Message::System { content } => assert_eq!(content, "You are concise."),
        _ => panic!("Expected system message"),
    }

    let json = serde_json::to_string(&message).expect("serialize");
    let roundtrip: Message = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(roundtrip, message);
}

#[test]
fn current_schema_tool_call_json_round_trips_without_provider_promotion() {
    // A minted handle with no provider must stay provider-less —
    // nothing in the round trip may invent provider provenance.
    let call = super::ToolCall::new(
        super::ToolCallId::new("minted-handle").expect("non-empty"),
        super::ToolFunction {
            name: "add".to_string(),
            arguments: serde_json::json!({}),
        },
    );

    let json = serde_json::to_value(&call).expect("serialize");
    assert!(json.get("call_id").is_none());
    let roundtrip: super::ToolCall = serde_json::from_value(json).expect("deserialize");
    assert_eq!(roundtrip.provider, None);
    assert_eq!(roundtrip, call);
}

#[test]
fn empty_params_canonicalize_to_none_in_both_serde_directions() {
    // The `AdditionalParams` contract, pinned where it lives. One
    // fixture, every direction: canonicalization, round-trip, tolerance,
    // and rejection.

    // An explicit `{}` or `null` decodes as `None` exactly like an
    // absent field.
    for empty_spelling in [serde_json::json!({}), serde_json::Value::Null] {
        let text: Text = serde_json::from_value(
            serde_json::json!({"text": "x", "additional_params": empty_spelling}),
        )
        .expect("deserialize");
        assert_eq!(text.additional_params, None);
    }

    // Data survives a round trip value-identically, and `Some` params
    // always carry data — `AdditionalParams` has no empty value, so the
    // old uncanonicalized-`Some({})` hazard is unrepresentable rather
    // than tolerated.
    let text: Text = serde_json::from_value(
        serde_json::json!({"text": "x", "additional_params": {"citations": [1]}}),
    )
    .expect("deserialize");
    assert_eq!(
        text.additional_params,
        AdditionalParams::from_entries([("citations", serde_json::json!([1]))])
    );
    assert_eq!(
        text.additional_params
            .as_ref()
            .and_then(|params| params.get("citations")),
        Some(&serde_json::json!([1]))
    );
    let round: Text = serde_json::from_value(serde_json::to_value(&text).expect("serialize"))
        .expect("round trip");
    assert_eq!(round, text);

    // The empty map canonicalizes to `None` at the constructor, so it
    // never reaches serialization at all.
    assert_eq!(AdditionalParams::new(serde_json::Map::new()), None);
    assert_eq!(
        AdditionalParams::try_from_value(serde_json::json!({})).expect("object"),
        None
    );

    // An unknown key on the block itself is tolerated and dropped —
    // never an error, never captured into params — so histories written
    // by a newer rig (or 0.41 flattened extras that were never
    // re-nested) still load; MIGRATING's strict-decode recipe is the
    // opt-in detector for the dropped keys.
    let tolerant: Text = serde_json::from_value(
        serde_json::json!({"text": "x", "citations": ["stray"], "future_field": 1}),
    )
    .expect("unknown keys on a block must not fail the decode");
    assert_eq!(tolerant.text, "x");
    assert_eq!(tolerant.additional_params, None);

    // Extras are a keyed namespace: a non-object carrier (the shape a
    // mis-firing migration script writes) is malformed data and fails
    // loudly instead of loading as a phantom annotation no extractor
    // can read.
    for malformed in [serde_json::json!([]), serde_json::json!("title")] {
        let err = serde_json::from_value::<Text>(
            serde_json::json!({"text": "x", "additional_params": malformed}),
        )
        .expect_err("non-object params must be a decode error");
        assert!(
            err.to_string().contains("must be a JSON object"),
            "unexpected error: {err}"
        );
        assert!(
            AdditionalParams::try_from_value(serde_json::json!([])).is_err(),
            "try_from_value must hand a non-object back, not swallow it"
        );
    }
}

#[test]
fn round_trip_diff_recipe_detects_every_dropped_key() {
    // Pins MIGRATING's opt-in verification recipe: the runtime load
    // path tolerates unknown keys (see the tolerance case in
    // `empty_params_canonicalize_to_none_in_both_serde_directions`),
    // and a migration script detects what tolerance dropped by loading,
    // re-serializing, and asking `keys_lost_in_round_trip` — a
    // serde_ignored-based recipe cannot serve here, because the
    // internally tagged enums buffer their content and hide ignored
    // keys from its callback.
    let migrated = serde_json::json!({
        "role": "assistant",
        "content": [
            {"type": "text", "text": "cited", "citations": ["not re-nested"]},
            {"type": "text", "text": "clean",
             "additional_params": {"citations": ["re-nested"]}},
        ],
    });
    let loaded: Message =
        serde_json::from_value(migrated.clone()).expect("tolerant decode must succeed");
    let reserialized = serde_json::to_value(&loaded).expect("serialize");
    assert_eq!(
        super::keys_lost_in_round_trip(&migrated, &reserialized),
        vec!["content.0.citations".to_string()],
        "every dropped key must be reported by path, and only dropped keys \
             — writer-added defaults are not differences"
    );

    // A fully re-nested history survives whole: the recipe's success
    // condition is an empty list. MIGRATING's blessed
    // `"additional_params": {}` spelling canonicalizes to absence and
    // must not read as a loss.
    let clean = serde_json::json!({
        "role": "assistant",
        "content": [
            {"type": "text", "text": "clean",
             "additional_params": {"citations": ["re-nested"]}},
            {"type": "text", "text": "mechanically migrated",
             "additional_params": {}},
        ],
    });
    let loaded: Message = serde_json::from_value(clean.clone()).expect("decode");
    let reserialized = serde_json::to_value(&loaded).expect("serialize");
    assert_eq!(
        super::keys_lost_in_round_trip(&clean, &reserialized),
        Vec::<String>::new(),
        "clean history must survive the round trip whole"
    );
}

#[test]
fn legacy_call_id_key_is_ignored_not_lifted() {
    // The pre-provider-split lift is deleted: a legacy `call_id` key is
    // an unknown field, so it deserializes with the key ignored — `id`
    // is read as rig's handle and `provider` stays absent. Pinned so a
    // future change (e.g. making the key a hard error) is a decision,
    // not an accident; the hand-migration recipe lives in MIGRATING.
    let legacy = serde_json::json!({
        "id": "fc_123",
        "call_id": "call_abc",
        "function": {"name": "add", "arguments": {"x": 1}},
    });

    let call: super::ToolCall = serde_json::from_value(legacy).expect("deserialize");
    assert_eq!(call.id, "fc_123");
    assert_eq!(call.provider, None);
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ExecutorLikeResponse {
    output: serde_json::Value,
    logs: Vec<String>,
    execution_time_ms: u64,
}

#[test]
fn tool_result_content_decodes_structured_and_legacy_json() {
    let response = ExecutorLikeResponse {
        output: serde_json::json!({"answer": 42}),
        logs: vec!["computed".to_string()],
        execution_time_ms: 7,
    };
    let value = serde_json::to_value(&response).expect("serialize response");

    let structured = ToolResultContent::json(value.clone());
    assert_eq!(structured.as_json(), Some(&value));
    assert_eq!(structured.as_text(), None);
    assert_eq!(
        structured
            .deserialize_json::<ExecutorLikeResponse>()
            .expect("decode structured response"),
        response
    );

    let legacy_json = value.to_string();
    let legacy_text = ToolResultContent::Text(Text::new(legacy_json.clone()));
    assert_eq!(legacy_text.as_text(), Some(legacy_json.as_str()));
    assert_eq!(legacy_text.as_json(), None);
    assert_eq!(
        legacy_text
            .deserialize_json::<ExecutorLikeResponse>()
            .expect("decode legacy response"),
        response
    );

    let image = ToolResultContent::image_url("https://example.com/result.png", None, None);
    let image_error = image.deserialize_json::<ExecutorLikeResponse>();
    assert!(image_error.is_err());
    if let Err(error) = image_error {
        assert_eq!(
            error.to_string(),
            "cannot decode image tool-result content as JSON"
        );
    }
}
