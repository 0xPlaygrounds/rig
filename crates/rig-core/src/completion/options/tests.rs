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

/// A target whose answer for each option is the function it holds.
#[derive(Debug)]
struct Fake(fn(OptionFields<'_>) -> OptionMap);

impl crate::completion::ReplayTarget for Fake {
    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static("fake.chat")
    }

    fn provider(&self) -> &str {
        "fake"
    }

    fn model(&self) -> &str {
        "fake-1"
    }

    fn accepts(&self, _model: &str) -> crate::completion::Accepts {
        crate::completion::Accepts::ALL
    }

    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: OptionFields<'_>,
    ) -> OptionMap {
        (self.0)(fields)
    }
}

/// `top_p` and `seed` sent, `stop` refused, `cache` placed, everything else
/// omitted.
fn answers(fields: OptionFields<'_>) -> OptionMap {
    let OptionFields {
        reasoning,
        cache,
        service_tier,
        verbosity,
        parallel_tool_calls,
        top_p,
        seed,
        stop,
    } = fields;
    OptionMap {
        reasoning: Mapping::of(reasoning, |_| {
            Mapping::Send(json!({"thinking": {"type": "enabled"}}))
        }),
        cache: Mapping::of(cache, |_| Mapping::Place),
        service_tier: Mapping::of(service_tier, |_| Mapping::Omit("the default")),
        verbosity: Mapping::of(verbosity, |_| Mapping::Omit("the default")),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| Mapping::Omit("the default")),
        top_p: Mapping::of(top_p, |p| Mapping::Send(json!({"sampling": {"top_p": p}}))),
        seed: Mapping::of(seed, |n| Mapping::Send(json!({"sampling": {"seed": n}}))),
        stop: Mapping::of_stop(stop, |_| Mapping::unsupported("no stop sequences")),
    }
}

fn request(options: GenerationOptions) -> crate::completion::CompletionRequest {
    crate::completion::CompletionRequest::new("hi").options(options)
}

fn base(
    input: &mut BaseInput<'_>,
) -> Result<serde_json::Map<String, serde_json::Value>, crate::error::EncodeError> {
    let mut body = serde_json::Map::new();
    body.insert("sampling".into(), json!({"temperature": 0.5, "top_p": 1.0}));
    body.insert("cached".into(), json!(input.cache().is_some()));
    let tools = input.raw_tools()?;
    body.insert("tools".into(), json!([{"name": "rig"}]));
    if let Some(serde_json::Value::Array(all)) = body.get_mut("tools") {
        all.extend(tools);
    }
    Ok(body)
}

#[test]
fn the_merge_ranks_base_then_options_then_additional_params() {
    let target = Fake(answers);
    let mut request = request(GenerationOptions::default().top_p(0.5).seed(7));
    request.additional_params = Some(json!({
        "sampling": {"seed": 9},
        "tools": [{"name": "raw"}],
        "user": null,
    }));
    let body = request_params(&target, &request, base, RawAt::Top, &[])
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        body.get("sampling"),
        Some(&json!({"temperature": 0.5, "top_p": 0.5, "seed": 9}))
    );
    assert_eq!(
        body.get("tools"),
        Some(&json!([{"name": "rig"}, {"name": "raw"}]))
    );
    assert_eq!(body.get("user"), Some(&serde_json::Value::Null));
    assert_eq!(body.pointer("/sampling/seed"), Some(&json!(9)));
}

#[test]
fn a_refused_option_is_an_error_or_a_warning_and_cleared() {
    let target = Fake(answers);
    let error = request_params(
        &target,
        &request(GenerationOptions::default().stop(["END"])),
        base,
        RawAt::Top,
        &[],
    )
    .expect_err("stop is refused");
    let refused = error.unsupported_option().cloned();
    assert_eq!(refused.as_ref().map(|refused| refused.option), Some("stop"));
    assert_eq!(
        refused.as_ref().map(|refused| refused.provider.as_str()),
        Some("fake")
    );
    assert_eq!(
        refused.map(|refused| refused.model),
        Some("fake-1".to_owned())
    );

    let capture = crate::test_utils::TraceCapture::default();
    let mut ignored = request(
        GenerationOptions::default()
            .stop(["END"])
            .seed(3)
            .on_unsupported(OnUnsupported::Ignore),
    );
    tracing::subscriber::with_default(capture.subscriber(), || check(&target, &mut ignored))
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(ignored.options.stop.is_empty());
    assert_eq!(ignored.options.seed, Some(3));
    let warnings = capture.warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("option=stop"), "{warnings:?}");
    assert!(warnings[0].contains("provider=fake"), "{warnings:?}");
}

#[test]
fn a_set_option_answered_with_nothing_fails() {
    fn silent(fields: OptionFields<'_>) -> OptionMap {
        OptionMap {
            seed: Mapping::Nothing,
            ..answers(fields)
        }
    }
    let error = request_params(
        &Fake(silent),
        &request(GenerationOptions::default().seed(1)),
        base,
        RawAt::Top,
        &[],
    )
    .expect_err("a set option is never dropped");
    assert!(error.unsupported_option().is_none());
    assert!(error.to_string().contains("seed"), "{error}");
}

#[test]
fn a_placed_cache_reaches_the_base_builder() {
    let body = request_params(
        &Fake(answers),
        &request(GenerationOptions::default().cache(CacheRetention::Long)),
        base,
        RawAt::Top,
        &[],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(body.get("cached"), Some(&json!(true)));
}

#[test]
fn the_raw_layer_lands_where_the_wire_says() {
    let raw = json!({"keep_alive": "5m", "num_ctx": 4096, "options": {"seed": 1}});
    let mut split = request(GenerationOptions::default());
    split.additional_params = Some(raw.clone());
    let body = request_params(
        &Fake(answers),
        &split,
        |_| {
            Ok(serde_json::Map::from_iter([(
                "options".into(),
                json!({"temperature": 0.1}),
            )]))
        },
        RawAt::Split {
            top: &["keep_alive"],
            rest: "options",
        },
        &[],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(body.get("keep_alive"), Some(&json!("5m")));
    assert_eq!(
        body.get("options"),
        Some(&json!({"temperature": 0.1, "seed": 1, "num_ctx": 4096}))
    );

    let body = request_params(
        &Fake(answers),
        &split,
        |_| Ok(serde_json::Map::new()),
        RawAt::Under("/additionalModelRequestFields"),
        &[],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(body.get("additionalModelRequestFields"), Some(&raw));

    let body = request_params(
        &Fake(answers),
        &split,
        |_| Ok(serde_json::Map::new()),
        RawAt::Ignored("the gateway rejects pass-through parameters"),
        &[],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert!(body.is_empty());
}

#[test]
fn mapped_param_reads_the_options_and_the_raw_layer_above_them() {
    let target = Fake(answers);
    let mut mapped = request(GenerationOptions::default().reasoning(Effort::High));
    assert_eq!(
        mapped_param(&target, &mapped, "thinking"),
        Some(json!({"type": "enabled"}))
    );
    mapped.additional_params = Some(json!({"thinking": {"type": "disabled"}}));
    assert_eq!(
        mapped_param(&target, &mapped, "thinking"),
        Some(json!({"type": "disabled"}))
    );
    assert_eq!(
        param(&target, &mapped, "thinking"),
        Some(&json!({"type": "disabled"}))
    );
}

#[test]
fn overlay_keeps_what_the_top_layer_leaves_unset() {
    let agent = GenerationOptions::default()
        .reasoning(Effort::Low)
        .stop(["A"])
        .on_unsupported(OnUnsupported::Ignore);
    let run = GenerationOptions::default().top_p(0.2);
    let resolved = agent.clone().overlay(&run);
    assert_eq!(resolved.reasoning, Some(Reasoning::Effort(Effort::Low)));
    assert_eq!(resolved.top_p, Some(0.2));
    assert_eq!(resolved.stop, vec!["A".to_owned()]);
    assert_eq!(resolved.on_unsupported, OnUnsupported::Ignore);
    assert_eq!(agent.clone().overlay(&GenerationOptions::default()), agent);
}
