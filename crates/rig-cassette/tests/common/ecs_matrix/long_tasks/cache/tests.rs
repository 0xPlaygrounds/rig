use super::*;

#[test]
fn anthropic_cache_oracle_checks_actual_encoded_modes() {
    use rig_core::{
        completion::{CompletionRequest, ToolDefinition},
        message::Message,
        providers::anthropic::{completion::CacheTtl, wire::Anthropic},
        wire::{Body, Mode, Wire as _},
    };
    let provider = Anthropic::new("local-test-key");
    let cases = [
        ("repair", provider.messages("model").with_prompt_caching()),
        (
            "repair_streamed",
            provider.messages("model").with_automatic_caching_1h(),
        ),
        (
            "inventory",
            provider
                .messages("model")
                .with_automatic_caching()
                .with_static_prefix_cache_ttl(CacheTtl::OneHour),
        ),
    ];
    for (scenario, model) in cases {
        for extra_turns in 0..3 {
            let mut history = vec![
                Message::system("Stable task contracts"),
                Message::user("Inspect the task"),
            ];
            for _ in 0..extra_turns {
                history.push(Message::assistant("Continue checking"));
                history.push(Message::user("Apply the next verified update"));
            }
            let request = CompletionRequest {
                model: None,
                chat_history: history,
                documents: vec![],
                tools: vec![ToolDefinition {
                    name: "probe".into(),
                    description: "Inspect state".into(),
                    parameters: json!({"type":"object","properties":{}}),
                }],
                temperature: None,
                max_tokens: Some(32),
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            };
            let encoded = model
                .encode(request, Mode::Unary)
                .expect("encode Anthropic request");
            let request = encoded.requests.first().expect("one request");
            let Body::Bytes(bytes) = request.body() else {
                panic!("JSON body")
            };
            let body: Value = serde_json::from_slice(bytes).expect("request JSON");
            super::super::assert_anthropic_request(&body, scenario);
        }
    }
}

#[test]
fn anthropic_accounting_oracle_checks_existing_native_cache_usage() {
    let path = crate::cassettes::cassette_root().parent().expect("fixture root").join("effects/world/anthropic_prompt_caching_conformance_agent_loop_keeps_hitting_across_tool_turns.effects.json");
    let log: EffectLog =
        serde_json::from_slice(&std::fs::read(path).expect("existing native cache log"))
            .expect("effect log");
    assert_usage(ThinkingWire::Anthropic, &log);
}

#[test]
#[should_panic(expected = "raw cache counters must match")]
fn wrong_cache_counter_is_rejected() {
    assert_prompt_usage(
        &Usage {
            input_tokens: Some(100),
            cached_input_tokens: Some(91),
            ..Usage::default()
        },
        (Some(100), Some(90), None),
    );
}

#[test]
#[should_panic(expected = "completion usage counted twice")]
fn duplicate_turn_is_rejected() {
    let mut seen = BTreeSet::new();
    assert_unique(&mut seen, "run/0:4".into());
    assert_unique(&mut seen, "run/0:4".into());
}

#[test]
fn unknown_counters_are_not_zero() {
    let rows = [
        Usage {
            cached_input_tokens: Some(12),
            ..Usage::default()
        },
        Usage::default(),
    ];
    assert_eq!(
        total_counter(&rows, |u| u.cached_input_tokens),
        json!({"reported_sum":12,"missing_turns":1,"complete_total":null})
    );
    assert_eq!(
        total_counter(&rows, |u| u.input_tokens),
        json!({"reported_sum":null,"missing_turns":2,"complete_total":null})
    );
}

#[test]
#[should_panic(expected = "cumulative reported usage")]
fn wrong_cumulative_cache_total_is_rejected() {
    let first = Usage {
        input_tokens: Some(100),
        cached_input_tokens: Some(40),
        ..Usage::default()
    };
    let second = Usage {
        input_tokens: Some(200),
        ..Usage::default()
    };
    let wrong = Usage {
        input_tokens: Some(300),
        cached_input_tokens: Some(80),
        ..Usage::default()
    };
    super::assert_totals(&[first, second], wrong);
}

#[test]
fn changing_the_recorded_system_prefix_is_detected() {
    use rig_test_support::cache_prefix::{canonical_prefix_blocks, compare};
    let interactions =
        crate::cassettes::recorded_interaction_bodies("openai", "long_task_matrix/chat_repair");
    let (first, _) = interactions.first().expect("recorded native request");
    let mut changed: Value = serde_json::from_str(first).expect("request JSON");
    let before = canonical_prefix_blocks("/v1/chat/completions", &changed).expect("modeled wire");
    *changed
        .pointer_mut("/messages/0/content")
        .expect("system content") = json!("Different task instructions");
    let after = canonical_prefix_blocks("/v1/chat/completions", &changed).expect("modeled wire");
    assert!(compare("changed system prefix", 1, &before, &after).is_some());
}

#[test]
fn complete_totals_count_each_reported_turn_once() {
    let rows = [
        Usage {
            input_tokens: Some(40),
            ..Usage::default()
        },
        Usage {
            input_tokens: Some(50),
            ..Usage::default()
        },
    ];
    assert_eq!(
        total_counter(&rows, |u| u.input_tokens),
        json!({"reported_sum":90,"missing_turns":0,"complete_total":90})
    );
}
