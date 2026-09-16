use super::*;
use serde_json::json;

#[test]
fn test_completion_response_deserialization_gemini_flash() {
    // Real response from OpenRouter with google/gemini-2.5-flash
    let json = json!({
        "id": "gen-AAAAAAAAAA-AAAAAAAAAAAAAAAAAAAA",
        "provider": "Google",
        "model": "google/gemini-2.5-flash",
        "object": "chat.completion",
        "created": 1765971703u64,
        "choices": [{
            "logprobs": null,
            "finish_reason": "stop",
            "native_finish_reason": "STOP",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "CONTENT",
                "refusal": null,
                "reasoning": null
            }
        }],
        "usage": {
            "prompt_tokens": 669,
            "completion_tokens": 5,
            "total_tokens": 674
        }
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    assert_eq!(response.id, "gen-AAAAAAAAAA-AAAAAAAAAAAAAAAAAAAA");
    assert_eq!(response.model, "google/gemini-2.5-flash");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.choices[0].finish_reason, Some("stop".to_string()));
    assert_eq!(response.choices[0].logprobs, None);
    let serialized = serde_json::to_value(&response).unwrap();
    assert!(
        serialized["choices"][0].get("logprobs").is_none(),
        "an absent optional native field stays absent when serialized"
    );
}

#[test]
fn raw_completion_choice_retains_logprobs() {
    let logprobs = json!({
        "content": [{
            "token": "cobalt",
            "logprob": -0.01,
            "bytes": [99],
            "top_logprobs": []
        }],
        "refusal": null
    });
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "gen-logprobs",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-4o-mini",
        "system_fingerprint": null,
        "choices": [{
            "index": 0,
            "native_finish_reason": "stop",
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "cobalt"},
            "logprobs": logprobs
        }],
        "usage": null
    }))
    .expect("OpenRouter's documented probability object should decode");

    assert_eq!(response.choices[0].logprobs, Some(logprobs));
}

/// A Gemini route answers with `role: "model"` rather than `"assistant"`,
/// and reports the model it actually served rather than the one requested.
#[test]
fn completion_response_decodes_a_gemini_model_role() {
    let json = json!({
        "id": "gen-BBBBBBBBBB-BBBBBBBBBBBBBBBBBBBB",
        "provider": "Google",
        "model": "google/gemini-2.5-pro-exp-03-25:free",
        "object": "chat.completion",
        "created": 1743780565u64,
        "choices": [{
            "logprobs": null,
            "finish_reason": "stop",
            "native_finish_reason": "STOP",
            "index": 0,
            "message": {
                "role": "model",
                "content": "CONTENT",
                "refusal": null,
                "reasoning": null
            }
        }],
        "usage": {
            "prompt_tokens": 669,
            "completion_tokens": 5,
            "total_tokens": 674
        }
    });
    let response: CompletionResponse = serde_json::from_value(json).unwrap();

    assert_eq!(
        response.model, "google/gemini-2.5-pro-exp-03-25:free",
        "the reported model is the routed one, not the requested one"
    );
    assert_eq!(response.provider.as_deref(), Some("Google"));
    let choice = response.choices.first().expect("one choice");
    assert_eq!(choice.native_finish_reason.as_deref(), Some("STOP"));
    assert_eq!(
        crate::providers::openai::completion::assistant_message_text_response(&choice.message)
            .as_deref(),
        Some("CONTENT")
    );
}

/// The shared choice decoder tolerates the truncated JSON a
/// `max_tokens`-capped turn emits only under `finish_reason: length`, and
/// drops the unusable call rather than losing the turn. Reproduced live on
/// DeepSeek (rig#2354) at 24/32/48/64-token budgets; the same wire type
/// backs OpenRouter, so the same turn shape is pinned here.
#[test]
fn truncated_tool_arguments_do_not_destroy_the_response() {
    let json = json!({
        "id": "gen-truncated",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek/deepseek-chat",
        "choices": [{
            "index": 0,
            "finish_reason": "length",
            "message": {
                "role": "assistant",
                "content": "Acknowledged.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "page", "arguments": "{\"team\":\"platform\"}"}
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "file_report", "arguments": "{\"summary\": "}
                    }
                ]
            }
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 24, "total_tokens": 34}
    });
    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    let choice = response.choices.first().expect("the turn survives");

    match &choice.message {
        Message::Assistant { tool_calls, .. } => {
            let names = tool_calls
                .iter()
                .map(|call| call.function.name.as_str())
                .collect::<Vec<_>>();
            assert_eq!(names, vec!["page"], "only the truncated call is dropped");
        }
        other => panic!("expected an assistant message, got {other:?}"),
    }
    assert_eq!(
        crate::providers::openai::completion::assistant_message_text_response(&choice.message)
            .as_deref(),
        Some("Acknowledged."),
        "the turn's text survives"
    );
    assert_eq!(response.usage.expect("usage").total_tokens, 34);
}

/// OpenRouter reports the upstream's own terminal reason when it has no
/// normalized one, so the truncation tolerance has to read that field too —
/// otherwise an Anthropic route's `max_output_tokens` turn fails to decode
/// at all.
#[test]
fn native_length_authorizes_the_same_tolerance() {
    let json = json!({
        "id": "gen-native-truncated",
        "object": "chat.completion",
        "created": 1,
        "model": "anthropic/claude-haiku-4.5",
        "choices": [{
            "index": 0,
            "finish_reason": null,
            "native_finish_reason": "max_output_tokens",
            "message": {
                "role": "assistant",
                "content": "still useful",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":"}
                }]
            }
        }]
    });
    let response: CompletionResponse = serde_json::from_value(json)
        .expect("the native terminal reason should authorize narrow truncation tolerance");
    let choice = response.choices.first().expect("the turn survives");

    assert_eq!(choice.finish_reason, None);
    assert_eq!(
        choice.native_finish_reason.as_deref(),
        Some("max_output_tokens")
    );
    match &choice.message {
        Message::Assistant { tool_calls, .. } => assert!(
            tool_calls.is_empty(),
            "the only call was truncated, so nothing is handed to a tool"
        ),
        other => panic!("expected an assistant message, got {other:?}"),
    }
}

/// Without a terminal reason that says "truncated", malformed tool output
/// stays loud.
#[test]
fn malformed_tool_arguments_without_a_length_reason_remain_loud() {
    let json = json!({
        "id": "gen-malformed",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-4.1-mini",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":"}
                }]
            }
        }]
    });

    assert!(
        serde_json::from_value::<CompletionResponse>(json).is_err(),
        "ordinary malformed tool output must remain loud"
    );
}

#[test]
fn test_message_assistant_without_reasoning_details() {
    // Verify that missing reasoning_details field doesn't cause deserialization failure
    let json = json!({
        "role": "assistant",
        "content": "Hello world",
        "refusal": null,
        "reasoning": null
    });

    let message: Message = serde_json::from_value(json).unwrap();
    match message {
        Message::Assistant {
            content,
            reasoning_details,
            ..
        } => {
            assert_eq!(content.len(), 1);
            assert!(reasoning_details.is_empty());
        }
        _ => panic!("Expected Assistant message"),
    }
}

#[test]
fn test_data_collection_serialization() {
    assert_eq!(
        serde_json::to_string(&DataCollection::Allow).unwrap(),
        r#""allow""#
    );
    assert_eq!(
        serde_json::to_string(&DataCollection::Deny).unwrap(),
        r#""deny""#
    );
}

#[test]
fn test_data_collection_default() {
    assert_eq!(DataCollection::default(), DataCollection::Allow);
}

#[test]
fn test_quantization_serialization() {
    assert_eq!(
        serde_json::to_string(&Quantization::Int4).unwrap(),
        r#""int4""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Int8).unwrap(),
        r#""int8""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Fp16).unwrap(),
        r#""fp16""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Bf16).unwrap(),
        r#""bf16""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Fp32).unwrap(),
        r#""fp32""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Fp8).unwrap(),
        r#""fp8""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Unknown).unwrap(),
        r#""unknown""#
    );
}

#[test]
fn test_provider_sort_strategy_serialization() {
    assert_eq!(
        serde_json::to_string(&ProviderSortStrategy::Price).unwrap(),
        r#""price""#
    );
    assert_eq!(
        serde_json::to_string(&ProviderSortStrategy::Throughput).unwrap(),
        r#""throughput""#
    );
    assert_eq!(
        serde_json::to_string(&ProviderSortStrategy::Latency).unwrap(),
        r#""latency""#
    );
}

#[test]
fn test_sort_partition_serialization() {
    assert_eq!(
        serde_json::to_string(&SortPartition::Model).unwrap(),
        r#""model""#
    );
    assert_eq!(
        serde_json::to_string(&SortPartition::None).unwrap(),
        r#""none""#
    );
}

#[test]
fn test_provider_sort_simple() {
    let sort = ProviderSort::Simple(ProviderSortStrategy::Latency);
    let json = serde_json::to_value(&sort).unwrap();
    assert_eq!(json, "latency");
}

#[test]
fn test_provider_sort_complex() {
    let sort = ProviderSort::Complex(
        ProviderSortConfig::new(ProviderSortStrategy::Price).partition(SortPartition::None),
    );
    let json = serde_json::to_value(&sort).unwrap();
    assert_eq!(json["by"], "price");
    assert_eq!(json["partition"], "none");
}

#[test]
fn test_provider_sort_complex_without_partition() {
    let sort = ProviderSort::Complex(ProviderSortConfig::new(ProviderSortStrategy::Throughput));
    let json = serde_json::to_value(&sort).unwrap();
    assert_eq!(json["by"], "throughput");
    assert!(json.get("partition").is_none());
}

#[test]
fn test_provider_sort_from_strategy() {
    let sort: ProviderSort = ProviderSortStrategy::Price.into();
    assert_eq!(sort, ProviderSort::Simple(ProviderSortStrategy::Price));
}

#[test]
fn test_provider_sort_from_config() {
    let config = ProviderSortConfig::new(ProviderSortStrategy::Latency);
    let sort: ProviderSort = config.into();
    match sort {
        ProviderSort::Complex(c) => assert_eq!(c.by, ProviderSortStrategy::Latency),
        _ => panic!("Expected Complex variant"),
    }
}

#[test]
fn test_percentile_thresholds_builder() {
    let thresholds = PercentileThresholds::new()
        .p50(10.0)
        .p75(25.0)
        .p90(50.0)
        .p99(100.0);

    assert_eq!(thresholds.p50, Some(10.0));
    assert_eq!(thresholds.p75, Some(25.0));
    assert_eq!(thresholds.p90, Some(50.0));
    assert_eq!(thresholds.p99, Some(100.0));
}

#[test]
fn test_percentile_thresholds_default() {
    let thresholds = PercentileThresholds::default();
    assert_eq!(thresholds.p50, None);
    assert_eq!(thresholds.p75, None);
    assert_eq!(thresholds.p90, None);
    assert_eq!(thresholds.p99, None);
}

#[test]
fn test_throughput_threshold_simple() {
    let threshold = ThroughputThreshold::Simple(50.0);
    let json = serde_json::to_value(&threshold).unwrap();
    assert_eq!(json, 50.0);
}

#[test]
fn test_throughput_threshold_percentile() {
    let threshold = ThroughputThreshold::Percentile(PercentileThresholds::new().p90(50.0));
    let json = serde_json::to_value(&threshold).unwrap();
    assert_eq!(json["p90"], 50.0);
}

#[test]
fn test_latency_threshold_simple() {
    let threshold = LatencyThreshold::Simple(0.5);
    let json = serde_json::to_value(&threshold).unwrap();
    assert_eq!(json, 0.5);
}

#[test]
fn test_latency_threshold_percentile() {
    let threshold = LatencyThreshold::Percentile(PercentileThresholds::new().p50(0.1).p99(1.0));
    let json = serde_json::to_value(&threshold).unwrap();
    assert_eq!(json["p50"], 0.1);
    assert_eq!(json["p99"], 1.0);
}

#[test]
fn test_max_price_builder() {
    let price = MaxPrice::new().prompt(0.001).completion(0.002);

    assert_eq!(price.prompt, Some(0.001));
    assert_eq!(price.completion, Some(0.002));
    assert_eq!(price.request, None);
    assert_eq!(price.image, None);
}

#[test]
fn test_max_price_all_fields() {
    let price = MaxPrice::new()
        .prompt(0.001)
        .completion(0.002)
        .request(0.01)
        .image(0.05);

    let json = serde_json::to_value(&price).unwrap();
    assert_eq!(json["prompt"], 0.001);
    assert_eq!(json["completion"], 0.002);
    assert_eq!(json["request"], 0.01);
    assert_eq!(json["image"], 0.05);
}

#[test]
fn test_max_price_default() {
    let price = MaxPrice::default();
    assert_eq!(price.prompt, None);
    assert_eq!(price.completion, None);
    assert_eq!(price.request, None);
    assert_eq!(price.image, None);
}

#[test]
fn test_provider_preferences_default() {
    let prefs = ProviderPreferences::default();
    assert!(prefs.order.is_none());
    assert!(prefs.only.is_none());
    assert!(prefs.ignore.is_none());
    assert!(prefs.allow_fallbacks.is_none());
    assert!(prefs.require_parameters.is_none());
    assert!(prefs.data_collection.is_none());
    assert!(prefs.zdr.is_none());
    assert!(prefs.sort.is_none());
    assert!(prefs.preferred_min_throughput.is_none());
    assert!(prefs.preferred_max_latency.is_none());
    assert!(prefs.max_price.is_none());
    assert!(prefs.quantizations.is_none());
}

#[test]
fn test_provider_preferences_order_with_fallbacks() {
    let prefs = ProviderPreferences::new()
        .order(["anthropic", "openai"])
        .allow_fallbacks(true);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["order"], json!(["anthropic", "openai"]));
    assert_eq!(provider["allow_fallbacks"], true);
}

#[test]
fn test_provider_preferences_only_allowlist() {
    let prefs = ProviderPreferences::new()
        .only(["azure", "together"])
        .allow_fallbacks(false);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["only"], json!(["azure", "together"]));
    assert_eq!(provider["allow_fallbacks"], false);
}

#[test]
fn test_provider_preferences_ignore() {
    let prefs = ProviderPreferences::new().ignore(["deepinfra"]);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["ignore"], json!(["deepinfra"]));
}

#[test]
fn test_provider_preferences_sort_latency() {
    let prefs = ProviderPreferences::new().sort(ProviderSortStrategy::Latency);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["sort"], "latency");
}

#[test]
fn test_provider_preferences_price_with_throughput() {
    let prefs = ProviderPreferences::new()
        .sort(ProviderSortStrategy::Price)
        .preferred_min_throughput(ThroughputThreshold::Percentile(
            PercentileThresholds::new().p90(50.0),
        ));

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["sort"], "price");
    assert_eq!(provider["preferred_min_throughput"]["p90"], 50.0);
}

#[test]
fn test_provider_preferences_require_parameters() {
    let prefs = ProviderPreferences::new().require_parameters(true);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["require_parameters"], true);
}

#[test]
fn test_provider_preferences_data_policy_and_zdr() {
    let prefs = ProviderPreferences::new()
        .data_collection(DataCollection::Deny)
        .zdr(true);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["data_collection"], "deny");
    assert_eq!(provider["zdr"], true);
}

#[test]
fn test_provider_preferences_quantizations() {
    let prefs = ProviderPreferences::new().quantizations([Quantization::Int8, Quantization::Fp16]);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["quantizations"], json!(["int8", "fp16"]));
}

#[test]
fn test_provider_preferences_serialization_skips_none() {
    let prefs = ProviderPreferences::new().sort(ProviderSortStrategy::Price);

    let json = serde_json::to_value(&prefs).unwrap();

    assert_eq!(json["sort"], "price");
    assert!(json.get("order").is_none());
    assert!(json.get("only").is_none());
    assert!(json.get("ignore").is_none());
    assert!(json.get("zdr").is_none());
}

#[test]
fn test_provider_preferences_deserialization() {
    let json = json!({
        "order": ["anthropic", "openai"],
        "sort": "throughput",
        "data_collection": "deny",
        "zdr": true,
        "quantizations": ["int8", "fp16"]
    });

    let prefs: ProviderPreferences = serde_json::from_value(json).unwrap();

    assert_eq!(
        prefs.order,
        Some(vec!["anthropic".to_string(), "openai".to_string()])
    );
    assert_eq!(
        prefs.sort,
        Some(ProviderSort::Simple(ProviderSortStrategy::Throughput))
    );
    assert_eq!(prefs.data_collection, Some(DataCollection::Deny));
    assert_eq!(prefs.zdr, Some(true));
    assert_eq!(
        prefs.quantizations,
        Some(vec![Quantization::Int8, Quantization::Fp16])
    );
}

#[test]
fn test_provider_preferences_deserialization_complex_sort() {
    let json = json!({
        "sort": {
            "by": "latency",
            "partition": "model"
        }
    });

    let prefs: ProviderPreferences = serde_json::from_value(json).unwrap();

    match prefs.sort {
        Some(ProviderSort::Complex(config)) => {
            assert_eq!(config.by, ProviderSortStrategy::Latency);
            assert_eq!(config.partition, Some(SortPartition::Model));
        }
        _ => panic!("Expected Complex sort variant"),
    }
}

#[test]
fn test_provider_preferences_full_integration() {
    let prefs = ProviderPreferences::new()
        .order(["anthropic", "openai"])
        .only(["anthropic", "openai", "google"])
        .sort(ProviderSortStrategy::Throughput)
        .data_collection(DataCollection::Deny)
        .zdr(true)
        .quantizations([Quantization::Int8])
        .allow_fallbacks(false);

    let json = prefs.to_json();

    assert!(json.get("provider").is_some());
    let provider = &json["provider"];
    assert_eq!(provider["order"], json!(["anthropic", "openai"]));
    assert_eq!(provider["only"], json!(["anthropic", "openai", "google"]));
    assert_eq!(provider["sort"], "throughput");
    assert_eq!(provider["data_collection"], "deny");
    assert_eq!(provider["zdr"], true);
    assert_eq!(provider["quantizations"], json!(["int8"]));
    assert_eq!(provider["allow_fallbacks"], false);
}

#[test]
fn test_provider_preferences_max_price() {
    let prefs =
        ProviderPreferences::new().max_price(MaxPrice::new().prompt(0.001).completion(0.002));

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["max_price"]["prompt"], 0.001);
    assert_eq!(provider["max_price"]["completion"], 0.002);
}

#[test]
fn test_provider_preferences_preferred_max_latency() {
    let prefs = ProviderPreferences::new().preferred_max_latency(LatencyThreshold::Simple(0.5));

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["preferred_max_latency"], 0.5);
}

#[test]
fn test_provider_preferences_empty_arrays() {
    let prefs = ProviderPreferences::new()
        .order(Vec::<String>::new())
        .quantizations(Vec::<Quantization>::new());

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["order"], json!([]));
    assert_eq!(provider["quantizations"], json!([]));
}
