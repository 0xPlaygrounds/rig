use super::*;
use serde_json::json;

#[test]
fn the_matching_mode_reads_exact_and_shape_and_refuses_anything_else() {
    let path = Path::new("example.yaml");
    let mode = |value: Option<&str>| BodyMatching::parse(value.map(str::to_owned), path);
    assert_eq!(mode(None).ok(), Some(BodyMatching::Exact));
    assert_eq!(mode(Some("")).ok(), Some(BodyMatching::Exact));
    assert_eq!(mode(Some("Exact")).ok(), Some(BodyMatching::Exact));
    assert_eq!(mode(Some("SHAPE")).ok(), Some(BodyMatching::Shape));
    let refused = mode(Some("fuzzy"));
    assert!(
        matches!(&refused, Err(CassetteError::InvalidMatchingMode { value, .. }) if value == "fuzzy"),
        "{refused:?}"
    );
    assert_eq!(refused.err().as_ref().map(CassetteError::path), Some(path));
}

#[test]
fn a_string_and_a_one_part_text_array_share_a_key() {
    let string = json!({
        "model": "gpt-5",
        "messages": [{ "role": "system", "content": "be brief" }, { "role": "user", "content": "hi" }],
    });
    let parts = json!({
        "model": "gpt-5",
        "messages": [
            { "role": "system", "content": [{ "type": "text", "text": "be terse" }] },
            { "role": "user", "content": "hello" },
        ],
    });
    assert_eq!(shape_key(&string), shape_key(&parts));
}

#[test]
fn values_and_their_types_are_erased_outside_the_kept_keys() {
    let one = json!({ "temperature": 0.2, "stop": ["a", "b"], "user": "x", "input": "text" });
    let other =
        json!({ "temperature": "hot", "stop": "a", "user": null, "input": ["many", "texts"] });
    assert_eq!(shape_key(&one), shape_key(&other));
}

#[test]
fn model_roles_tools_and_reply_mode_stay_in_the_key() {
    let base = json!({
        "model": "gpt-5",
        "stream": false,
        "messages": [{ "role": "user", "content": "hi" }],
        "tools": [{ "type": "function", "function": { "name": "lookup", "parameters": {} } }],
    });
    let key = shape_key(&base);
    for (pointer, value) in [
        ("/model", json!("gpt-4o")),
        ("/stream", json!(true)),
        ("/messages/0/role", json!("developer")),
        ("/tools/0/function/name", json!("verify")),
    ] {
        let mut changed = base.clone();
        if let Some(slot) = changed.pointer_mut(pointer) {
            *slot = value;
        }
        assert_ne!(shape_key(&changed), key, "{pointer} is part of the key");
    }
    let mut longer = base.clone();
    if let Some(Value::Array(messages)) = longer.pointer_mut("/messages") {
        messages.push(json!({ "role": "assistant", "content": "hello" }));
    }
    assert_ne!(
        shape_key(&longer),
        key,
        "the role sequence is part of the key"
    );
    let mut extra = base;
    if let Some(map) = extra.as_object_mut() {
        map.insert("seed".into(), json!(1));
    }
    assert_ne!(shape_key(&extra), key, "the fields are part of the key");
}

#[test]
fn tool_schemas_arguments_and_results_collapse() {
    let one = json!({
        "tools": [{ "name": "lookup", "input_schema": { "type": "object", "properties": { "q": {} } } }],
        "messages": [
            { "role": "assistant", "content": [{ "type": "tool_use", "name": "lookup", "input": { "q": "a" } }] },
            { "role": "user", "content": [{ "type": "tool_result", "content": "found" }] },
        ],
    });
    let other = json!({
        "tools": [{ "name": "lookup", "input_schema": { "type": "object" } }],
        "messages": [
            { "role": "assistant", "content": [{ "type": "text", "text": "ok" }, { "type": "tool_use", "name": "lookup", "input": {} }] },
            { "role": "user", "content": [{ "type": "tool_result", "content": [{ "type": "text", "text": "none" }] }] },
        ],
    });
    assert_eq!(shape_key(&one), shape_key(&other));
    let renamed = json!({
        "tools": [{ "name": "lookup", "input_schema": {} }],
        "messages": [
            { "role": "assistant", "content": [{ "type": "tool_use", "name": "verify", "input": {} }] },
            { "role": "user", "content": "found" },
        ],
    });
    assert_ne!(
        shape_key(&one),
        shape_key(&renamed),
        "a called tool's name is kept"
    );
}

#[test]
fn a_responses_conversation_keeps_its_items() {
    let call = json!({ "input": [
        { "role": "user", "content": "hi" },
        { "type": "function_call", "name": "lookup", "arguments": "{}", "call_id": "c1" },
    ] });
    let answer = json!({ "input": [
        { "role": "user", "content": "hi" },
        { "type": "function_call_output", "output": "found", "call_id": "c1" },
    ] });
    assert_ne!(shape_key(&call), shape_key(&answer));
}

#[test]
fn multipart_binary_text_and_empty_bodies_have_kinds() {
    let part = |name: &str, body: Value| json!({ "headers": { "content-disposition": format!("form-data; name=\"{name}\"") }, "body": body });
    let audio = json!({ "multipart": [part("model", json!("whisper")), part("file", json!({ "bytes": 3, "fnv1a64": "00" }))] });
    let other = json!({ "multipart": [part("file", json!("x")), part("model", json!("gpt"))] });
    assert_eq!(shape_key(&audio), json!("multipart[file,model]"));
    assert_eq!(shape_key(&audio), shape_key(&other));
    assert_eq!(
        shape_key(&json!({ "bytes": 9, "fnv1a64": "ff" })),
        json!("binary")
    );
    assert_eq!(shape_key(&json!("plain text")), json!("text"));
    assert_eq!(shape_key(&Value::Null), Value::Null);
}
