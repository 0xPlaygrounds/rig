use super::*;
use serde_json::json;

#[test]
fn the_default_is_empty_and_serializes_without_stop() {
    let options = GenerationOptions::default();
    assert!(options.is_default());
    assert_eq!(options.on_unsupported, OnUnsupported::Error);
    assert_eq!(
        serde_json::to_value(&options).ok(),
        Some(json!({
            "reasoning": null,
            "cache": null,
            "service_tier": null,
            "verbosity": null,
            "parallel_tool_calls": null,
            "top_p": null,
            "seed": null,
            "on_unsupported": "error",
        }))
    );
}

#[test]
fn builders_set_each_field_and_serde_uses_the_wire_words() {
    let options = GenerationOptions::default()
        .reasoning(Effort::XHigh)
        .cache(CacheRetention::Long)
        .service_tier(ServiceTier::Flex)
        .verbosity(Verbosity::Low)
        .parallel_tool_calls(false)
        .top_p(0.9)
        .seed(7)
        .stop(["END"])
        .on_unsupported(OnUnsupported::Ignore);
    assert!(!options.is_default());
    let value = serde_json::to_value(&options).ok();
    assert_eq!(
        value,
        Some(json!({
            "reasoning": {"effort": "xhigh"},
            "cache": "long",
            "service_tier": "flex",
            "verbosity": "low",
            "parallel_tool_calls": false,
            "top_p": 0.9,
            "seed": 7,
            "stop": ["END"],
            "on_unsupported": "ignore",
        }))
    );
    let back = value.and_then(|value| serde_json::from_value::<GenerationOptions>(value).ok());
    assert_eq!(back, Some(options));
}

#[test]
fn every_field_is_optional_when_read() {
    let options = serde_json::from_value::<GenerationOptions>(json!({
        "reasoning": {"budget": {"tokens": 2048}},
    }))
    .ok();
    assert_eq!(
        options,
        Some(GenerationOptions::default().reasoning(Reasoning::Budget { tokens: 2048 }))
    );
}

#[test]
fn an_unsupported_option_names_the_field_provider_and_model() {
    let refusal = UnsupportedOption::new("cache", "deepseek", "deepseek-v4", "no cache control");
    assert_eq!(
        refusal.to_string(),
        "`cache` is not supported by deepseek model `deepseek-v4`: no cache control"
    );
}
