use super::*;

fn reply(content_type: &str, body: &str) -> Exchange {
    Exchange {
        path: "/v1/chat/completions".into(),
        method: "POST".into(),
        status: 200,
        header: vec![NameValue {
            name: "content-type".into(),
            value: content_type.into(),
        }],
        body: Some(body.into()),
        body_encoding: None,
    }
}

fn candidate(body: &str, source: &str) -> Entry {
    entry(
        "openai",
        "POST /v1/chat/completions",
        &reply("application/json", body),
        source.into(),
    )
}

#[test]
fn a_reply_names_the_tools_it_calls_and_how_it_ended() {
    let chat = candidate(
        r#"{"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[{"function":{"name":"add","arguments":"{}"}},{"function":{"name":"add","arguments":"{}"}}]}}]}"#,
        "a.yaml#0",
    );
    assert_eq!(chat.calls, ["add"]);
    assert_eq!(chat.ends, ["tool_calls"]);

    let stream = entry(
        "anthropic",
        "POST /v1/messages",
        &reply(
            "text/event-stream",
            "event: content_block_start\ndata: {\"content_block\":{\"type\":\"tool_use\",\"name\":\"lookup\",\"input\":{}}}\n\n\
             event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
        ),
        "b.yaml#1".into(),
    );
    assert_eq!(stream.calls, ["lookup"]);
    assert_eq!(stream.ends, ["tool_use"]);

    let responses = candidate(
        r#"{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[],"tools":[{"name":"add","parameters":{}}]}"#,
        "c.yaml#0",
    );
    assert!(responses.calls.is_empty(), "a tool definition is no call");
    assert_eq!(responses.ends, ["incomplete", "max_output_tokens"]);
}

#[test]
fn the_bank_keeps_the_smallest_reply_of_a_key_then_the_first_source() {
    let text = |content: &str| {
        format!(r#"{{"choices":[{{"finish_reason":"stop","message":{{"content":"{content}"}}}}]}}"#)
    };
    let bank = build(vec![
        candidate(&text("a longer answer"), "a.yaml#0"),
        candidate(&text("short"), "c.yaml#0"),
        candidate(&text("tiny!"), "b.yaml#3"),
    ]);
    assert_eq!(bank.len(), 1, "one shape, one entry");
    assert_eq!(bank[0].source, "b.yaml#3");
}

#[test]
fn replies_calling_different_tools_are_different_entries() {
    let call = |name: &str| {
        format!(
            r#"{{"choices":[{{"finish_reason":"tool_calls","message":{{"tool_calls":[{{"function":{{"name":"{name}","arguments":"{{}}"}}}}]}}}}]}}"#
        )
    };
    let bank = build(vec![
        candidate(&call("add"), "a.yaml#0"),
        candidate(&call("subtract"), "a.yaml#1"),
    ]);
    assert_eq!(bank.len(), 2);
    assert_eq!(bank[0].shape, bank[1].shape);
}

#[test]
fn a_rendered_bank_reads_back() {
    let bank = build(vec![
        candidate(r#"{"choices":[{"finish_reason":"stop"}]}"#, "a.yaml#0"),
        entry(
            "anthropic",
            "POST /v1/messages",
            &reply("application/json", r#"{"stop_reason":"end_turn"}"#),
            "b.yaml#0".into(),
        ),
    ]);
    let files = render(&bank).expect("renders");
    assert_eq!(
        files.keys().collect::<Vec<_>>(),
        ["anthropic.yaml", "openai.yaml"]
    );
    let read: Vec<Entry> = files
        .values()
        .flat_map(|text| parse(text).expect("parses"))
        .collect();
    assert_eq!(read, bank);
}

#[test]
fn only_completion_encoders_are_banked() {
    assert!(is_completion("POST /v1/chat/completions"));
    assert!(is_completion(
        "POST /v1beta/models/{model}:streamGenerateContent"
    ));
    assert!(is_completion("POST /model/{model}/converse-stream"));
    assert!(!is_completion("POST /v1/embeddings"));
    assert!(!is_completion("GET /v1/responses"));
    assert!(!is_completion("DELETE /v1/responses/{id}"));
}

#[test]
fn scripts_read_back_and_name_unbanked_replies_with_a_dash() {
    let mut scripts = Scripts::new();
    scripts.insert(
        "openai/a.yaml".into(),
        vec![Some("POST /v1/responses;00ff;add".into()), None],
    );
    scripts.insert(
        "openai/b.yaml".into(),
        vec![Some("POST /v1/responses;0aff;".into())],
    );
    let text = render_scripts(&scripts);
    assert!(
        text.contains("openai/a.yaml\tPOST /v1/responses;00ff;add\t-\n"),
        "{text}"
    );
    assert_eq!(parse_scripts(&text).expect("parses"), scripts);
}

#[test]
fn scenario_literals_are_read_off_a_source() {
    let found = string_literals(
        r#"with_openai_cassette("matrix/cell_one", body); let x = "not a scenario"; "a/b-c.d""#,
    );
    assert_eq!(
        found.into_iter().collect::<Vec<_>>(),
        ["a/b-c.d", "matrix/cell_one"]
    );
}

#[test]
fn a_pinned_list_names_fixtures_and_their_reasons() {
    let listed = parse_pinned(
        "# why each is pinned\n\nopenai/a.yaml  the answer must end in DONE\n  deepseek/b.yaml reasoning\n",
    );
    assert_eq!(
        listed.into_iter().collect::<Vec<_>>(),
        ["deepseek/b.yaml", "openai/a.yaml"]
    );
    assert_eq!(fixture_of("openai/a.yaml#12"), "openai/a.yaml");
    assert_eq!(interaction_of("openai/a.yaml#12"), 12);
}

#[test]
fn a_named_function_delta_is_a_call_and_a_server_tool_is_not() {
    let stream = entry(
        "cohere",
        "POST /compatibility/v1/chat/completions",
        &reply(
            "text/event-stream",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"name\":\"subtract\"}}]}}]}\n\n",
        ),
        "c.yaml#0".into(),
    );
    assert_eq!(stream.calls, ["subtract"]);
    let server = entry(
        "anthropic",
        "POST /v1/messages",
        &reply(
            "application/json",
            r#"{"content":[{"type":"server_tool_use","name":"web_search","input":{}}],"stop_reason":"end_turn"}"#,
        ),
        "s.yaml#0".into(),
    );
    assert!(server.calls.is_empty(), "the provider runs its own tools");
}

#[test]
fn the_decode_targets_sweeps_name_the_providers_whose_replies_hold_shapes() {
    let source = "macro_rules! sweeps {\n    ($($provider:ident),* $(,)?) => {};\n}\n\nsweeps!(\n    anthropic, openai,\n    xai,\n);\n";
    assert_eq!(
        swept_providers(source),
        Some(BTreeSet::from(["anthropic", "openai", "xai"]))
    );
    assert_eq!(swept_providers("sweeps!();"), None);
    assert_eq!(swept_providers("fn main() {}\nsweeps!(\n);\n"), None);
}

#[test]
fn the_committed_bank_holds_reply_shapes_only_for_swept_providers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let held = held_shapes(&root).unwrap();
    assert!(held.keys().all(|key| key.kind == Kind::Reply));
    assert!(held.keys().any(|key| key.provider == "openai"));
    assert!(!held.keys().any(|key| key.provider == "bedrock"));
}

/// A key a kept script names stays in the bank after its source fixture is
/// re-recorded with another shape; a key no script names, or one a current
/// recording gives, is not kept from the committed bank.
#[test]
fn a_key_a_script_names_survives_its_source_being_re_recorded() {
    let named = candidate(
        r#"{"choices":[{"finish_reason":"stop","message":{"content":"old"}}]}"#,
        "openai/rerecorded.yaml#0",
    );
    let unnamed = candidate(
        r#"{"choices":[{"finish_reason":"length","message":{"content":"old"}}]}"#,
        "openai/rerecorded.yaml#1",
    );
    let given = candidate(
        r#"{"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[{"function":{"name":"add","arguments":"{}"}}]}}]}"#,
        "openai/rerecorded.yaml#2",
    );
    let committed = vec![named.clone(), unnamed, given.clone()];
    let candidates = vec![given.clone()];
    let scripts = Scripts::from([(
        "openai/pruned.yaml".to_owned(),
        vec![Some(named.script_key()), Some(given.script_key()), None],
    )]);
    let kept: Vec<&str> = stranded(&committed, &candidates, &scripts)
        .map(|entry| entry.source.as_str())
        .collect();
    assert_eq!(kept, ["openai/rerecorded.yaml#0"]);
}
