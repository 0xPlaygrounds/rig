use serde_json::json;

use crate::message::{
    AssistantContent, AssistantMessage, CallId, Fingerprint, Message, Opaque, Origin, StopReason,
    Text, ToolCall, ToolFunction, ToolName,
};

fn call(arguments: serde_json::Value) -> AssistantContent {
    let name = ToolName::new("lookup").expect("a tool name");
    AssistantContent::ToolCall(ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(name, arguments),
    ))
}

#[test]
fn fingerprints_survive_a_serde_round_trip() {
    // Object order is canonicalized before hashing.
    let arguments = json!({"zeta": 1, "alpha": {"y": 0.1, "x": [2.718_281_828_459_1, 1e-7]}});
    let block = call(arguments).with_native(json!({"type": "function_call", "id": "fc_1"}));
    let saved = serde_json::to_string(&block).expect("serialize");
    let loaded: AssistantContent = serde_json::from_str(&saved).expect("deserialize");
    assert_eq!(loaded.fingerprint(), block.fingerprint());
    assert!(loaded.native_item().is_some());
}

#[test]
fn native_calls_survive_repeated_float_storage() {
    let floats = [
        0.1,
        std::f64::consts::E,
        51.248_178_375_505_404,
        -93.311_370_376_880_33,
        2.003_039_774_426_776_2e-253,
        7.101_215_824_554_616e260,
        1.769_268_377_902_049e74,
        3.928_753_217_337_331_5e299,
        1e20,
        -0.0,
    ];
    for float in floats {
        let mut block = call(json!({"nested": [{"value": float}]}))
            .with_native(json!({"type": "function_call", "id": "fc_1", "signature": "keep"}));
        let fingerprint = block.fingerprint();
        for cycle in 0..9 {
            block = match cycle % 3 {
                0 => serde_json::from_str(&serde_json::to_string(&block).expect("serialize"))
                    .expect("load JSON"),
                1 => serde_json::from_value(serde_json::to_value(&block).expect("to value"))
                    .expect("load value"),
                _ => {
                    let value = serde_json::to_value(&block).expect("to value");
                    serde_json::from_str(&value.to_string()).expect("load stored value")
                }
            };
            assert_eq!(
                block.native_item().and_then(|item| item.get("signature")),
                Some(&json!("keep")),
                "{float} after storage cycle {cycle}"
            );
            let AssistantContent::ToolCall(tool) = &block else {
                panic!("a call stays a call");
            };
            assert_eq!(
                tool.native.as_ref().map(|native| native.fingerprint),
                Some(fingerprint)
            );
        }
    }
}

#[test]
fn adjacent_float_edits_stay_stale_before_and_after_storage() {
    for original in [
        0.100_000_000_000_000_03_f64,
        f64::from_bits(0x7617_17c1_3ccb_d15f),
        9.999_999_999_999_84e19,
    ] {
        let edited = f64::from_bits(original.to_bits() + 1);
        let mut block = call(json!({"value": original}))
            .with_native(json!({"type": "function_call", "id": "fc_1"}));
        if let AssistantContent::ToolCall(call) = &mut block {
            call.function
                .arguments
                .insert("value".into(), json!(edited));
        }
        assert!(block.native_item().is_none());
        for _ in 0..4 {
            let saved = serde_json::to_string(&block).expect("serialize edited call");
            block = serde_json::from_str(&saved).expect("load edited call");
            assert!(block.native_item().is_none(), "{saved}");
        }
    }
    let original = 0.100_000_000_000_000_03_f64;

    let mut loaded: AssistantContent = serde_json::from_str(
        &serde_json::to_string(
            &call(json!({"value": original})).with_native(json!({"id": "fc_1"})),
        )
        .expect("serialize"),
    )
    .expect("load");
    if let AssistantContent::ToolCall(call) = &mut loaded {
        let current = call
            .function
            .arguments
            .get("value")
            .and_then(serde_json::Value::as_f64)
            .expect("a float");
        call.function
            .arguments
            .insert("value".into(), json!(f64::from_bits(current.to_bits() + 1)));
    }
    assert!(loaded.native_item().is_none());
}

#[test]
fn float_baselines_detect_integer_precision_lost_by_a_store() {
    fn through_f64(value: &mut serde_json::Value) {
        use serde_json::Value;
        match value {
            Value::Number(number) => {
                if let Some(float) = number.as_f64().and_then(serde_json::Number::from_f64) {
                    *number = float;
                }
            }
            Value::Array(values) => values.iter_mut().for_each(through_f64),
            Value::Object(fields) => fields.values_mut().for_each(through_f64),
            _ => {}
        }
    }
    for (big, current) in [(12_345_678_901_234_567_u64, false), (20, true)] {
        let block =
            call(json!({"big": big, "scale": 0.5, "nested": [{"a/b~c": big}], "text": "i:20"}))
                .with_native(json!({"arguments": format!(r#"{{"big":{big},"scale":0.5}}"#)}));
        let mut stored = serde_json::to_value(&block).expect("save call");
        through_f64(&mut stored);
        let loaded: AssistantContent = serde_json::from_value(stored).expect("load call");
        assert_eq!(loaded.native_item().is_some(), current, "integer {big}");
    }
}

#[test]
fn legacy_native_records_keep_exact_fingerprint_validation() {
    let block = call(json!({"limit": 20})).with_native(json!({"id": "fc_1"}));
    let mut stored = serde_json::to_value(&block).expect("serialize");
    if let Some(native) = stored
        .get_mut("native")
        .and_then(serde_json::Value::as_object_mut)
    {
        native.retain(|key, _| key != "baseline");
    }
    let mut loaded: AssistantContent = serde_json::from_value(stored).expect("legacy load");
    assert!(loaded.native_item().is_some());
    if let AssistantContent::ToolCall(call) = &mut loaded {
        call.function.arguments.insert("limit".into(), json!(21));
    }
    assert!(loaded.native_item().is_none());
}

#[test]
fn opaque_items_carry_no_separate_native() {
    let opaque = AssistantContent::Opaque(Opaque {
        item: json!({"type": "web_search_call", "id": "ws_1"}),
        replay: true,
    });
    assert_eq!(opaque.clone().with_native(json!({"ignored": true})), opaque);
    assert_eq!(opaque.native_item(), None);
}

#[test]
fn an_assistant_message_serializes_flat_under_its_role() {
    let turn = AssistantMessage {
        content: vec![AssistantContent::Text(Text::new("hi"))],
        origin: Some(Origin::new("anthropic.messages", "anthropic", "claude")),
        stop: Some(StopReason::Error("refused".into())),
    };
    let value = serde_json::to_value(Message::Assistant(turn.clone())).expect("serialize");
    assert_eq!(
        value,
        json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "origin": {"api": "anthropic.messages", "provider": "anthropic", "model": "claude"},
            "stop": {"error": "refused"},
        })
    );
    assert_eq!(
        serde_json::from_value::<Message>(value).expect("deserialize"),
        Message::Assistant(turn)
    );
}

#[test]
fn a_store_that_writes_whole_numbers_as_integers_keeps_the_item_current() {
    let block = call(json!({"limit": 20.0, "scale": 0.5}))
        .with_native(json!({"type": "function_call", "id": "fc_1"}));
    let stored = serde_json::to_string(&block)
        .expect("a block serializes")
        .replace("20.0", "20");
    let loaded: AssistantContent = serde_json::from_str(&stored).expect("a stored block loads");
    assert!(loaded.native_item().is_some(), "{stored}");
    assert_ne!(
        Fingerprint::of(&json!({"scale": 0.5})),
        Fingerprint::of(&json!({"scale": 1}))
    );
}

#[test]
fn a_stored_item_that_cannot_be_read_loads_as_no_item() {
    for fingerprint in [json!(1234), json!("not hex"), json!(null)] {
        let block: AssistantContent = serde_json::from_value(json!({
            "type": "text",
            "text": "hi",
            "native": {"item": {"id": "msg_1"}, "fingerprint": fingerprint}
        }))
        .expect("the block loads");
        assert_eq!(block, AssistantContent::text("hi"), "{fingerprint}");
    }
    let current = AssistantContent::text("hi").with_native(json!({"id": "msg_1"}));
    let stored = serde_json::to_value(&current).expect("serializes");
    assert_eq!(
        serde_json::from_value::<AssistantContent>(stored).expect("loads"),
        current
    );
}

/// A native item records the first of its wire's slots that holds a
/// blob, finds the blob from that record after a round trip through
/// storage, and records none when every slot is empty.
#[test]
fn a_reasoning_blob_is_read_where_its_wire_put_it() {
    use crate::message::{Blob, Native};
    let slots = ["/signature", "/data"];
    let mut redacted = Native {
        item: json!({"type": "redacted_thinking", "data": "r"}),
        fingerprint: Fingerprint::of(&"x"),
        baseline: None,
        edited: false,
        blob_slot: None,
    };
    redacted.find_blob(&slots);
    let stored = serde_json::to_value(&redacted).expect("serializes");
    let stored: Native = serde_json::from_value(stored).expect("loads");
    let found = stored.reasoning_blob();
    let mut unsigned = Native {
        item: json!({"type": "thinking", "thinking": "t", "signature": ""}),
        ..redacted.clone()
    };
    unsigned.find_blob(&slots);
    assert_eq!(
        found,
        Some(Blob {
            key: "/data",
            value: "r"
        })
    );
    assert_eq!(
        (&unsigned.blob_slot, unsigned.reasoning_blob()),
        (&None, None)
    );
}

/// A slot's `*` step finds the first entry of an array that holds a blob,
/// however late, and records that entry's index; an array with none, or
/// no array, records none.
#[test]
fn a_star_step_records_the_first_entry_holding_a_blob() {
    use crate::message::{Blob, Native};
    let slots = ["/blocks/*/signature", "/blocks/*/data"];
    let found = |item: serde_json::Value| {
        let mut native = Native {
            item,
            fingerprint: Fingerprint::of(&"x"),
            baseline: None,
            edited: false,
            blob_slot: None,
        };
        native.find_blob(&slots);
        let stored = serde_json::to_value(&native).expect("serializes");
        let stored: Native = serde_json::from_value(stored).expect("loads");
        stored
            .reasoning_blob()
            .map(|Blob { key, value }| (key.to_owned(), value.to_owned()))
    };
    let unsigned = json!({"type": "thinking", "thinking": "t", "signature": ""});
    let redacted = json!({"type": "redacted_thinking", "data": "r"});
    let signed = json!({"type": "thinking", "thinking": "t", "signature": "s"});
    assert_eq!(
        found(json!({"blocks": [unsigned, redacted, signed]})),
        Some(("/blocks/2/signature".to_owned(), "s".to_owned()))
    );
    assert_eq!(
        found(json!({"blocks": [unsigned, redacted]})),
        Some(("/blocks/1/data".to_owned(), "r".to_owned()))
    );
    assert_eq!(found(json!({"blocks": [unsigned]})), None);
    assert_eq!(found(json!({"blocks": "s"})), None);
}

/// Stored native items carry these values, so a change to the projection or
/// to how it is hashed would stale every saved item.
#[test]
fn fingerprints_are_pinned() {
    use crate::completion::ToolDefinition;
    use crate::message::{
        DocumentSourceKind, Image, ImageDetail, ImageMediaType, Reasoning, ToolCall, ToolFunction,
    };
    let tool = |name: &str, parameters| {
        ToolDefinition::new(
            ToolName::new(name).expect("a tool name"),
            format!("{name} things"),
            parameters,
        )
    };
    let tools = [
        tool(
            "search",
            json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        ),
        tool(
            "add",
            json!({"properties": {"b": {"maximum": 1.0}}, "type": "object"}),
        ),
    ];
    let arguments = json!({"b": {"z": 1, "a": 20.0}, "a": [1.5, 2.0, -0.0, 1e20]});
    let local = AssistantContent::ToolCall(ToolCall::new(
        CallId::from_wire(""),
        ToolFunction::new(ToolName::new("lookup").expect("a tool name"), json!({})),
    ));
    let image = AssistantContent::Image(Image {
        data: DocumentSourceKind::base64("aGk="),
        media_type: Some(ImageMediaType::PNG),
        detail: Some(ImageDetail::High),
        native: None,
    });
    let opaque = AssistantContent::Opaque(Opaque {
        item: json!({"type": "web_search_call"}),
        replay: true,
    });
    let reasoning = AssistantContent::Reasoning(Reasoning {
        redacted: true,
        ..Reasoning::new("plan")
    });
    let hex = |fingerprint: Fingerprint| format!("{:016x}", fingerprint.0);
    let pinned = [
        hex(AssistantContent::text("hello").fingerprint()),
        hex(reasoning.fingerprint()),
        hex(call(arguments).fingerprint()),
        hex(local.fingerprint()),
        hex(image.fingerprint()),
        hex(opaque.fingerprint()),
        hex(Fingerprint::of(&json!(tools))),
    ];
    assert_eq!(
        pinned,
        [
            "be5cdb8169d835fd", // text
            "c2dc7b0f4e836ea1", // reasoning
            "3df7dcbe7a89ecc8", // call with a provider id
            "7e2d775ebdb398fc", // call with a local id
            "37e9c9a0147c5070", // image
            "9604486c7b29d4c5", // opaque
            "2fc10d4b16d1b11c", // tools, as a request's context hashes them
        ]
    );
}
