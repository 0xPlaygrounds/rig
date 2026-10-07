use serde_json::{Value, json};

use super::*;
use crate::completion::{GenerationOptions, OnUnsupported, Verbosity};
use crate::error::ProviderError;
use crate::operation::Completion;
use crate::providers::openai::wire::{
    AZURE, Chat, DEEPSEEK, Dialect, GROQ, MISTRAL, MOONSHOT, OLLAMA, OPENAI, OPENROUTER,
    OpenAIConfig, VENICE,
};
use crate::wire::{Body, Mode, Operation, Wire};

fn chat(dialect: &Dialect, model: &str) -> Chat {
    Chat::new(OpenAIConfig::with_key(dialect, "sk-test"), model)
}

/// The body `chat` sends for `options` alone, prepared as the driver
/// prepares it.
fn sent(chat: &Chat, options: GenerationOptions) -> Result<Value, ProviderError> {
    let request = CompletionRequest::new("hi").max_tokens(16).options(options);
    let request = Completion::prepare(request, &chat.describe())?;
    let encoded = chat.encode(request, Mode::Unary)?;
    let Body::Bytes(bytes) = encoded.request.body() else {
        return Err(ProviderError::request("a chat body is JSON"));
    };
    Ok(serde_json::from_slice(bytes)?)
}

fn refused(result: Result<Value, ProviderError>) -> Option<&'static str> {
    match result {
        Err(ProviderError::UnsupportedOption(option)) => match option.option {
            std::borrow::Cow::Borrowed(name) => Some(name),
            std::borrow::Cow::Owned(name) => panic!("a GenerationOptions field, got {name}"),
        },
        _ => None,
    }
}

#[test]
fn openai_model_facts_come_from_the_catalog() {
    assert_eq!(reasons("o3-mini"), Some(true));
    assert_eq!(reasons("openai/gpt-5.6-sol"), Some(true));
    assert_eq!(reasons("gpt-5-2025-08-07"), Some(true), "a dated snapshot");
    assert_eq!(reasons("gpt-4.1"), Some(false));
    assert_eq!(reasons("my-deployment"), None);
    assert!(caches_by_options("gpt-6-sol") && !caches_by_options("gpt-5.5"));
}

/// OpenAI ids the catalog does not list (from OpenAI's own model list,
/// `openai/models/list_models_smoke.yaml`) reason by their name: Chat sends
/// `max_completion_tokens`, and `Off` is refused where the name says the
/// model cannot turn reasoning off.
#[test]
fn an_unlisted_openai_reasoning_id_reasons_by_its_name() {
    let unlisted = [
        "gpt-5-codex",
        "gpt-5.1-codex",
        "gpt-5.1-codex-mini",
        "gpt-5.1-codex-max",
        "gpt-5.2-codex",
        "gpt-5-chat-latest",
        "gpt-5.1-chat-latest",
        "gpt-5-search-api-2025-10-14",
        "o3-deep-research-2025-06-26",
        "o4-mini-deep-research",
    ];
    for model in unlisted {
        assert!(openai_spec(model).is_none(), "{model} is listed now");
        assert_eq!(reasons(model), Some(true), "{model}");
        let body = sent(&chat(&OPENAI, model), GenerationOptions::default()).expect(model);
        assert_eq!(body["max_completion_tokens"], 16, "{model}");
        assert!(body.get("max_tokens").is_none(), "{model}");
    }
    for model in ["gpt-5-codex", "o3-deep-research-2025-06-26"] {
        assert_eq!(
            refused(sent(
                &chat(&OPENAI, model),
                GenerationOptions::default().reasoning(Reasoning::Off),
            )),
            Some("reasoning"),
            "{model}"
        );
    }
    let body = sent(
        &chat(&OPENAI, "gpt-5.1-codex"),
        GenerationOptions::default().reasoning(Reasoning::Off),
    )
    .expect("GPT-5.1 turns reasoning off");
    assert_eq!(body["reasoning_effort"], "none");
}

/// The `max_completion_tokens` rename reads the full id: a vendor-prefixed
/// id (a proxy behind `OPENAI_BASE_URL`) is sent `max_tokens`, as before the
/// catalog.
#[test]
fn a_vendor_prefixed_id_keeps_max_tokens() {
    for model in ["openai/gpt-5", "openai/o3-mini", "litellm/gpt-5.4"] {
        let body = sent(&chat(&OPENAI, model), GenerationOptions::default()).expect(model);
        assert_eq!(body["max_tokens"], 16, "{model}");
        assert!(body.get("max_completion_tokens").is_none(), "{model}");
    }
    let body = sent(&chat(&OPENAI, "gpt-5"), GenerationOptions::default()).expect("gpt-5");
    assert_eq!(body["max_completion_tokens"], 16);
}

/// On xAI's Chat route the encoder refuses a reasoning option exactly when
/// `ModelSpec::validate` does, for every xAI row: Grok 4.3 turns reasoning
/// off with `reasoning_effort: "none"`.
#[test]
fn xai_chat_reasoning_agrees_with_validate() {
    let reasonings = [
        Reasoning::Off,
        Reasoning::Effort(Effort::Minimal),
        Reasoning::Effort(Effort::Low),
        Reasoning::Effort(Effort::Medium),
        Reasoning::Effort(Effort::High),
        Reasoning::Effort(Effort::XHigh),
        Reasoning::Effort(Effort::Max),
        Reasoning::Budget { tokens: 1024 },
    ];
    let mut checked = 0;
    for spec in crate::catalog::Catalog::builtin()
        .iter()
        .filter(|spec| spec.provider.vendor() == "xai")
    {
        for reasoning in &reasonings {
            let options = GenerationOptions::default().reasoning(*reasoning);
            let encoded = sent(
                &chat(&crate::providers::xai::DIALECT, &spec.id),
                options.clone(),
            );
            assert_eq!(
                refused(encoded).is_some(),
                spec.validate(&options).is_err(),
                "{}: {reasoning:?}",
                spec.id
            );
            checked += 1;
        }
    }
    assert!(checked > 100, "{checked}");
    let body = sent(
        &chat(&crate::providers::xai::DIALECT, "grok-4.3"),
        GenerationOptions::default().reasoning(Reasoning::Off),
    )
    .expect("Grok 4.3 turns reasoning off");
    assert_eq!(body["reasoning_effort"], "none");
}

/// Grok 4.3 takes `xhigh` (docs.x.ai/docs/models/grok-4.3).
#[test]
fn grok_4_3_sends_xhigh_effort() {
    let body = sent(
        &chat(&crate::providers::xai::DIALECT, "grok-4.3"),
        GenerationOptions::default().reasoning(Effort::XHigh),
    )
    .expect("Grok 4.3 takes xhigh");
    assert_eq!(body["reasoning_effort"], "xhigh");
}

/// A Grok id the catalog does not list reasons unless its id says
/// `non-reasoning`, so `Off` is sent as nothing there and refused elsewhere.
#[test]
fn an_unlisted_grok_id_reasons_by_its_name() {
    for model in ["grok-4-fast-non-reasoning", "grok-4-1-fast-non-reasoning"] {
        let body = sent(
            &chat(&crate::providers::xai::DIALECT, model),
            GenerationOptions::default().reasoning(Reasoning::Off),
        )
        .expect("a non-reasoning model is already off");
        assert!(body.get("reasoning_effort").is_none(), "{body}");
    }
    assert_eq!(
        refused(sent(
            &chat(&crate::providers::xai::DIALECT, "grok-4-fast-reasoning"),
            GenerationOptions::default().reasoning(Reasoning::Off),
        )),
        Some("reasoning")
    );
}

#[test]
fn openai_chat_cells() {
    let gpt = |model| chat(&OPENAI, model);
    let body = sent(
        &gpt("gpt-5.2"),
        GenerationOptions::default().reasoning(Reasoning::Off),
    )
    .expect("GPT-5.2 turns reasoning off");
    assert_eq!(body["reasoning_effort"], "none");
    assert_eq!(
        refused(sent(
            &gpt("gpt-5"),
            GenerationOptions::default().reasoning(Reasoning::Off)
        )),
        Some("reasoning")
    );
    let body = sent(
        &gpt("gpt-4o"),
        GenerationOptions::default().reasoning(Reasoning::Off),
    )
    .expect("a model that does not reason is already off");
    assert!(body.get("reasoning_effort").is_none(), "{body}");
    assert_eq!(
        refused(sent(
            &gpt("gpt-4o"),
            GenerationOptions::default().reasoning(Effort::High)
        )),
        Some("reasoning")
    );

    let body = sent(
        &gpt("gpt-5.2"),
        GenerationOptions::default().cache(CacheRetention::Short),
    )
    .expect("in-memory retention");
    assert_eq!(body["prompt_cache_retention"], "in_memory");
    let body = sent(
        &gpt("gpt-5.2"),
        GenerationOptions::default().cache(CacheRetention::Long),
    )
    .expect("extended retention");
    assert_eq!(body["prompt_cache_retention"], "24h");
    let body = sent(
        &gpt("gpt-5.6"),
        GenerationOptions::default().cache(CacheRetention::None),
    )
    .expect("explicit mode with no breakpoints");
    assert_eq!(body["prompt_cache_options"], json!({"mode": "explicit"}));
    assert_eq!(
        refused(sent(
            &gpt("gpt-5.6"),
            GenerationOptions::default().cache(CacheRetention::Long)
        )),
        Some("cache")
    );
    assert_eq!(
        refused(sent(
            &gpt("gpt-5.5"),
            GenerationOptions::default().cache(CacheRetention::Short)
        )),
        Some("cache")
    );

    let body = sent(
        &gpt("gpt-5.2"),
        GenerationOptions::default()
            .service_tier(ServiceTier::Flex)
            .verbosity(Verbosity::High)
            .parallel_tool_calls(true)
            .top_p(0.3)
            .seed(4)
            .stop(["a", "b"]),
    )
    .expect("every other option");
    assert_eq!(body["service_tier"], "flex");
    assert_eq!(body["verbosity"], "high");
    assert_eq!(body["parallel_tool_calls"], true);
    assert_eq!(body["top_p"], 0.3);
    assert_eq!(body["seed"], 4);
    assert_eq!(body["stop"], json!(["a", "b"]));
    assert_eq!(
        refused(sent(
            &gpt("gpt-5.2"),
            GenerationOptions::default().stop(["a", "b", "c", "d", "e"])
        )),
        Some("stop")
    );
}

#[test]
fn azure_refuses_what_its_default_api_version_predates() {
    let azure = chat(&AZURE, "prod-deployment");
    assert_eq!(
        refused(sent(
            &azure,
            GenerationOptions::default().reasoning(Effort::High)
        )),
        Some("reasoning")
    );
    assert_eq!(
        refused(sent(
            &azure,
            GenerationOptions::default().service_tier(ServiceTier::Auto)
        )),
        Some("service_tier")
    );
    let mut later = azure.clone();
    later.provider.api_version = Some("2025-04-01-preview".to_owned());
    let body = sent(&later, GenerationOptions::default().reasoning(Effort::High))
        .expect("a later api-version takes reasoning");
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(
        refused(sent(
            &later,
            GenerationOptions::default().reasoning(Effort::Max)
        )),
        Some("reasoning")
    );
}

#[test]
fn dialect_cells() {
    let body = sent(
        &chat(&OPENROUTER, "openai/gpt-5.2"),
        GenerationOptions::default()
            .reasoning(Reasoning::Off)
            .cache(CacheRetention::Short)
            .service_tier(ServiceTier::Auto),
    )
    .expect("OpenRouter options");
    assert_eq!(body["reasoning"], json!({"effort": "none"}));
    assert!(
        body.get("cache_control").is_none(),
        "an automatic upstream: {body}"
    );
    assert!(body.get("service_tier").is_none(), "{body}");
    let body = sent(
        &chat(&OPENROUTER, "anthropic/claude-sonnet-4.5"),
        GenerationOptions::default()
            .reasoning(Reasoning::Budget { tokens: 2048 })
            .cache(CacheRetention::Long),
    )
    .expect("an Anthropic upstream takes a budget and a marker");
    assert_eq!(body["reasoning"], json!({"max_tokens": 2048}));
    assert_eq!(
        body["cache_control"],
        json!({"type": "ephemeral", "ttl": "1h"})
    );

    let body = sent(
        &chat(&DEEPSEEK, "deepseek-v4-flash"),
        GenerationOptions::default().reasoning(Reasoning::Off),
    )
    .expect("DeepSeek turns thinking off");
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
    assert_eq!(
        refused(sent(
            &chat(&DEEPSEEK, "deepseek-v4-flash"),
            GenerationOptions::default().reasoning(Effort::Medium)
        )),
        Some("reasoning")
    );

    let body = sent(
        &chat(&MISTRAL, "mistral-large"),
        GenerationOptions::default().seed(3),
    )
    .expect("Mistral's seed");
    assert_eq!(body["random_seed"], 3);
    assert!(body.get("seed").is_none(), "{body}");

    let body = sent(
        &chat(&GROQ, "qwen/qwen3.8-27b"),
        GenerationOptions::default().service_tier(ServiceTier::Priority),
    )
    .expect("Groq's tiers");
    assert_eq!(body["service_tier"], "performance");
    assert_eq!(
        refused(sent(
            &chat(&GROQ, "openai/gpt-oss-120b"),
            GenerationOptions::default().reasoning(Reasoning::Off)
        )),
        Some("reasoning")
    );

    let body = sent(
        &chat(&VENICE, "venice-uncensored"),
        GenerationOptions::default().cache(CacheRetention::Long),
    )
    .expect("Venice's retention");
    assert_eq!(body["prompt_cache_retention"], "24h");

    let body = sent(
        &chat(&MOONSHOT, "kimi-k2.6"),
        GenerationOptions::default()
            .reasoning(Reasoning::Off)
            .cache(CacheRetention::Short),
    )
    .expect("Kimi K2.6");
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
    assert_eq!(
        body["prompt_cache_options"],
        json!({"mode": "implicit", "ttl": "5m"})
    );

    let body = sent(
        &chat(&OLLAMA, "qwen3:4b"),
        GenerationOptions::default().reasoning(Effort::Low),
    )
    .expect("Ollama's /v1 effort");
    assert_eq!(body["reasoning_effort"], "low");
}

#[test]
fn a_gateway_rig_does_not_know_refuses_every_option() {
    const GATEWAY: Dialect = Dialect::gateway("mygw", "https://gateway.example/v1", "MYGW_KEY");
    let gateway = chat(&GATEWAY, "model");
    assert_eq!(
        refused(sent(&gateway, GenerationOptions::default().top_p(0.5))),
        Some("top_p")
    );
    let body = sent(&gateway, GenerationOptions::default()).expect("no option set");
    assert!(body.get("top_p").is_none(), "{body}");
}

mod responses {
    use serde_json::{Value, json};

    use crate::completion::{
        CacheRetention, CompletionRequest, Effort, GenerationOptions, Reasoning, ServiceTier,
        Verbosity,
    };
    use crate::error::ProviderError;
    use crate::operation::Completion;
    use crate::providers::openai::responses_api::wire::Responses;
    use crate::providers::openai::wire::{Dialect, OPENAI, OPENROUTER, OpenAIConfig};
    use crate::wire::{Body, Mode, Operation, Wire};

    fn responses(dialect: &Dialect, model: &str) -> Responses {
        Responses::new(OpenAIConfig::with_key(dialect, "sk-test"), model)
    }

    fn sent(wire: &Responses, request: CompletionRequest) -> Result<Value, ProviderError> {
        let request = Completion::prepare(request, &wire.describe())?;
        let encoded = wire.encode(request, Mode::Unary)?;
        let Body::Bytes(bytes) = encoded.request.body() else {
            return Err(ProviderError::request("a Responses body is JSON"));
        };
        Ok(serde_json::from_slice(bytes)?)
    }

    fn with(options: GenerationOptions) -> CompletionRequest {
        CompletionRequest::new("hi").options(options)
    }

    fn refused(result: Result<Value, ProviderError>) -> Option<&'static str> {
        match result {
            Err(ProviderError::UnsupportedOption(option)) => match option.option {
                std::borrow::Cow::Borrowed(name) => Some(name),
                std::borrow::Cow::Owned(name) => panic!("a GenerationOptions field, got {name}"),
            },
            _ => None,
        }
    }

    /// A reasoning id the catalog does not list takes `top_p` only at effort
    /// `none`, and `none` only where its name allows it.
    #[test]
    fn an_unlisted_reasoning_id_samples_only_with_reasoning_off() {
        for model in [
            "gpt-5-codex",
            "o3-deep-research-2025-06-26",
            "gpt-5.1-codex",
        ] {
            assert_eq!(
                refused(sent(
                    &responses(&OPENAI, model),
                    with(GenerationOptions::default().top_p(0.5))
                )),
                Some("top_p"),
                "{model}"
            );
        }
        for model in ["gpt-5-codex", "o3-deep-research-2025-06-26"] {
            assert_eq!(
                refused(sent(
                    &responses(&OPENAI, model),
                    with(GenerationOptions::default().reasoning(Reasoning::Off))
                )),
                Some("reasoning"),
                "{model}"
            );
        }
        let body = sent(
            &responses(&OPENAI, "gpt-5.1-codex"),
            with(
                GenerationOptions::default()
                    .reasoning(Reasoning::Off)
                    .top_p(0.5),
            ),
        )
        .expect("GPT-5.1 samples with reasoning off");
        assert_eq!(body["reasoning"], json!({"effort": "none"}));
        assert_eq!(body["top_p"], 0.5);
    }

    /// Ported from #2171 (`prompt_cache_options_serialize_at_request_top_level`):
    /// a long cache on GPT-5.6 is `prompt_cache_options.ttl`, and a raw
    /// `mode` and `prompt_cache_key` join it at the top level.
    #[test]
    fn prompt_cache_options_serialize_at_request_top_level() {
        let body = sent(
            &responses(&OPENAI, "gpt-5.6"),
            with(GenerationOptions::default().cache(CacheRetention::Long)).additional_params(
                json!({
                    "prompt_cache_key": "tenant:acme:v1",
                    "prompt_cache_options": {"mode": "explicit"},
                }),
            ),
        )
        .expect("encodes");
        assert_eq!(body["prompt_cache_key"], "tenant:acme:v1");
        assert_eq!(body["prompt_cache_options"]["mode"], "explicit");
        assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
    }

    /// Ported from #2616 (`prompt_cache_options_reach_the_request_typed_and_through_additional_params`):
    /// the option and the raw key send the same object, and no retention.
    #[test]
    fn prompt_cache_options_reach_the_request_typed_and_through_additional_params() {
        let wire = responses(&OPENAI, crate::providers::openai::GPT_6_SOL);
        let typed = sent(
            &wire,
            with(GenerationOptions::default().cache(CacheRetention::Long)),
        )
        .expect("encodes");
        assert_eq!(typed["prompt_cache_options"], json!({"ttl": "30m"}));
        assert_eq!(typed.get("prompt_cache_retention"), None);
        let raw = sent(
            &wire,
            CompletionRequest::new("hi")
                .additional_params(json!({"prompt_cache_options": {"ttl": "30m"}})),
        )
        .expect("encodes");
        assert_eq!(raw["prompt_cache_options"], typed["prompt_cache_options"]);
    }

    #[test]
    fn openai_responses_cells() {
        let gpt = |model| responses(&OPENAI, model);
        let body = sent(
            &gpt("gpt-5.2"),
            with(GenerationOptions::default().reasoning(Effort::XHigh)),
        )
        .expect("GPT-5.2 takes xhigh");
        assert_eq!(body["reasoning"], json!({"effort": "xhigh"}));
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        for (model, reasoning) in [
            ("gpt-5.1", Reasoning::Effort(Effort::XHigh)),
            ("gpt-5.4", Reasoning::Effort(Effort::Max)),
            ("gpt-5.2", Reasoning::Effort(Effort::Minimal)),
            ("gpt-5", Reasoning::Off),
            ("gpt-6-astra", Reasoning::Off),
            ("gpt-5.2-pro", Reasoning::Off),
            ("gpt-5.2", Reasoning::Budget { tokens: 1024 }),
        ] {
            assert_eq!(
                refused(sent(
                    &gpt(model),
                    with(GenerationOptions::default().reasoning(reasoning))
                )),
                Some("reasoning"),
                "{model}: {reasoning:?}"
            );
        }
        let body = sent(
            &gpt("gpt-5.2"),
            with(GenerationOptions::default().verbosity(Verbosity::High)),
        )
        .expect("verbosity");
        assert_eq!(body["text"], json!({"verbosity": "high"}));

        // `top_p` reaches a reasoning model only at effort `none`.
        assert_eq!(
            refused(sent(
                &gpt("gpt-5.2"),
                with(GenerationOptions::default().top_p(0.5))
            )),
            Some("top_p")
        );
        let body = sent(
            &gpt("gpt-5.2"),
            with(
                GenerationOptions::default()
                    .top_p(0.5)
                    .reasoning(Reasoning::Off),
            ),
        )
        .expect("top_p at effort none");
        assert_eq!(body["top_p"], 0.5);
        let body = sent(
            &gpt("gpt-4.1"),
            with(GenerationOptions::default().top_p(0.5)),
        )
        .expect("a model that does not reason samples");
        assert_eq!(body["top_p"], 0.5);
        for options in [
            GenerationOptions::default().seed(1),
            GenerationOptions::default().stop(["x"]),
        ] {
            assert!(refused(sent(&gpt("gpt-5.2"), with(options))).is_some());
        }
    }

    #[test]
    fn dialect_cells() {
        let xai = responses(&crate::providers::xai::DIALECT, "grok-4.7");
        let body = sent(
            &xai,
            with(GenerationOptions::default().reasoning(Effort::XHigh)),
        )
        .expect("Grok 4.7 takes xhigh");
        assert_eq!(body["reasoning"], json!({"effort": "xhigh"}));
        let body = sent(
            &responses(&crate::providers::xai::DIALECT, "grok-4.3"),
            with(GenerationOptions::default().reasoning(Effort::XHigh)),
        )
        .expect("Grok 4.3 takes xhigh");
        assert_eq!(body["reasoning"], json!({"effort": "xhigh"}));
        assert_eq!(
            refused(sent(
                &xai,
                with(GenerationOptions::default().reasoning(Reasoning::Off))
            )),
            Some("reasoning")
        );
        assert_eq!(
            refused(sent(
                &responses(&crate::providers::xai::DIALECT, "grok-4.5"),
                with(GenerationOptions::default().reasoning(Effort::XHigh))
            )),
            Some("reasoning")
        );

        let openrouter = responses(&OPENROUTER, "anthropic/claude-sonnet-4.5");
        let body = sent(
            &openrouter,
            with(
                GenerationOptions::default()
                    .cache(CacheRetention::Short)
                    .service_tier(ServiceTier::Flex),
            ),
        )
        .expect("OpenRouter Responses");
        assert_eq!(body["cache_control"], json!({"type": "ephemeral"}));
        assert_eq!(body["service_tier"], "flex");

        let copilot = responses(&crate::providers::copilot::wire::DIALECT, "gpt-5.3-codex");
        let body = sent(
            &copilot,
            with(GenerationOptions::default().reasoning(Effort::High)),
        )
        .expect("Copilot's codex models take an effort");
        assert_eq!(body["reasoning"], json!({"effort": "high"}));
        assert_eq!(
            refused(sent(
                &copilot,
                with(GenerationOptions::default().reasoning(Reasoning::Off))
            )),
            Some("reasoning")
        );
    }

    #[test]
    fn the_websocket_body_is_the_http_body_without_its_stream_flags() {
        use crate::providers::openai::responses_api::Delivery;
        let wire = responses(&OPENAI, "gpt-5.2");
        let request = with(GenerationOptions::default().reasoning(Effort::Low))
            .additional_params(json!({"background": true, "stream": true}));
        let http = serde_json::to_value(
            wire.responses_request(&request, Delivery::Http(Mode::Streaming))
                .expect("the HTTP body"),
        )
        .expect("serializes");
        let socket = serde_json::to_value(
            wire.responses_request(&request, Delivery::WebSocket)
                .expect("the session body"),
        )
        .expect("serializes");
        assert_eq!(http["stream"], true);
        assert_eq!(http["background"], true);
        let mut expected = http;
        if let Some(fields) = expected.as_object_mut() {
            fields.shift_remove("stream");
            fields.shift_remove("background");
        }
        assert_eq!(socket, expected);
    }
}

// GPT-6's sampling and Chat Completions tool rules, ported from #2616 and
// expressed as catalog data: the four GPT-6 models' `sampling` is
// `reasoning_off`, they are marked `chat_tools_need_reasoning_off`, and their
// documented default effort is `medium`. Sources: the GPT-6 model pages and
// <https://developers.openai.com/api/docs/guides/latest-model>.

use crate::completion::ToolDefinition;
use crate::message::ToolName;
use crate::providers::openai::completion::{GPT_6_ASTRA, GPT_6_LUNA, GPT_6_SOL};
use crate::providers::openai::responses_api::wire::Responses as ResponsesWire;

fn gpt_6_request(additional_params: Option<Value>) -> CompletionRequest {
    CompletionRequest::new("hi").additional_params(additional_params)
}

fn with_tool(request: CompletionRequest) -> CompletionRequest {
    request.tool(ToolDefinition::new(
        ToolName::new("lookup").expect("a tool name"),
        "Look something up.",
        json!({"type": "object", "properties": {}}),
    ))
}

fn gpt_6_body(
    encoded: Result<crate::wire::Encoded, crate::error::EncodeError>,
) -> Result<Value, String> {
    let encoded = encoded.map_err(|error| ProviderError::from(error).to_string())?;
    let Body::Bytes(bytes) = encoded.request.body() else {
        return Err("a JSON body".to_owned());
    };
    serde_json::from_slice(bytes).map_err(|error| error.to_string())
}

fn on_chat(model: &str, request: CompletionRequest) -> Result<Value, String> {
    gpt_6_body(chat(&OPENAI, model).encode(request, Mode::Unary))
}

/// The body `chat` sends for `request`, prepared as the driver prepares it.
fn on_prepared_chat(model: &str, request: CompletionRequest) -> Result<Value, ProviderError> {
    crate::test_utils::provider_extensions::encoded_body(
        &chat(&OPENAI, model),
        request,
        Mode::Unary,
    )
}

/// The body Responses sends for `request`, prepared as the driver prepares
/// it.
fn on_prepared_responses(model: &str, request: CompletionRequest) -> Result<Value, ProviderError> {
    crate::test_utils::provider_extensions::encoded_body(
        &responses_wire(model),
        request,
        Mode::Unary,
    )
}

fn on_responses(model: &str, request: CompletionRequest) -> Result<Value, String> {
    gpt_6_body(
        ResponsesWire::new(OpenAIConfig::with_key(&OPENAI, "sk-test"), model)
            .encode(request, Mode::Unary),
    )
}

/// `request` with a generation option set, so the catalog's refusals apply
/// under the default `Error` policy. The effort is GPT-6's default.
fn strict(request: CompletionRequest) -> CompletionRequest {
    request.options(GenerationOptions::default().reasoning(Effort::Medium))
}

/// `request` with the `Ignore` policy and no other option.
fn lenient(request: CompletionRequest) -> CompletionRequest {
    request.options(GenerationOptions::default().on_unsupported(OnUnsupported::Ignore))
}

/// The refusal `result` carries: its option name, provider and model.
fn refusal(result: Result<Value, ProviderError>) -> (String, String, String) {
    match result {
        Err(ProviderError::UnsupportedOption(refused)) => {
            (refused.option.into_owned(), refused.provider, refused.model)
        }
        other => panic!("expected an unsupported option, got {other:?}"),
    }
}

/// The body `wire` sends for `request`, prepared as the driver prepares
/// it, and the warnings logged on the way.
fn sent_with_warnings<W>(wire: &W, request: CompletionRequest) -> (Value, Vec<String>)
where
    W: Wire<Op = crate::operation::Completion, Payload = crate::wire::Encoded>,
{
    let capture = crate::test_utils::TraceCapture::default();
    let body = tracing::subscriber::with_default(capture.subscriber(), || {
        crate::test_utils::provider_extensions::encoded_body(wire, request, Mode::Unary)
    })
    .unwrap_or_else(|error| panic!("{error}"));
    (body, capture.warnings())
}

fn responses_wire(model: &str) -> ResponsesWire {
    ResponsesWire::new(OpenAIConfig::with_key(&OPENAI, "sk-test"), model)
}

/// With a generation option set, the default `Error` policy refuses a
/// sampling parameter while GPT-6 reasons, by default or at an explicit
/// effort, with the field's name.
#[test]
fn gpt_6_sampling_parameters_fail_while_the_model_reasons_by_default() {
    for model in [GPT_6_ASTRA, GPT_6_SOL, GPT_6_LUNA] {
        let with_temperature = strict(gpt_6_request(None).temperature(0.2));
        for result in [
            on_prepared_chat(model, with_temperature.clone()),
            on_prepared_responses(model, with_temperature),
        ] {
            assert_eq!(
                refusal(result),
                (
                    "temperature".to_owned(),
                    "openai".to_owned(),
                    model.to_owned()
                )
            );
        }
        let top_p =
            on_prepared_responses(model, strict(gpt_6_request(Some(json!({"top_p": 0.9})))));
        assert_eq!(refusal(top_p).0, "top_p");
        for (field, value) in [
            ("top_p", json!(0.9)),
            ("top_logprobs", json!(2)),
            ("logprobs", json!(true)),
        ] {
            let result =
                on_prepared_chat(model, strict(gpt_6_request(Some(json!({field: value})))));
            assert_eq!(refusal(result).0, field, "{model}");
        }
        let error =
            on_chat(model, strict(gpt_6_request(None).temperature(0.2))).expect_err("refused");
        assert!(
            error.contains(model) && error.contains("`temperature`"),
            "{error}"
        );
    }
}

/// A request that sets no generation option is sent as built, as before the
/// catalog held these rules, and the provider decides: the sampling and
/// tool rules refuse nothing for it.
#[test]
fn gpt_6_without_options_is_sent_and_left_to_the_provider() {
    for model in [GPT_6_ASTRA, GPT_6_SOL, GPT_6_LUNA] {
        let body = on_chat(
            model,
            with_tool(gpt_6_request(Some(json!({"top_p": 0.9})))).temperature(0.2),
        )
        .unwrap_or_else(|error| panic!("{model}: {error}"));
        assert_eq!(body["temperature"], json!(0.2), "{model}");
        assert_eq!(body["top_p"], json!(0.9), "{model}");
        assert_eq!(body["tools"][0]["function"]["name"], "lookup", "{model}");
        let body = on_responses(
            model,
            gpt_6_request(Some(json!({"top_logprobs": 2}))).temperature(0.2),
        )
        .unwrap_or_else(|error| panic!("{model}: {error}"));
        assert_eq!(body["temperature"], json!(0.2), "{model}");
        assert_eq!(body["top_logprobs"], json!(2), "{model}");
    }
}

/// Under `Ignore` the request goes out with one warning per refused field:
/// a field rig wrote from a typed source (the request's `temperature`, the
/// `top_p` option) is left out, a raw key is sent as written.
#[test]
fn gpt_6_sampling_under_ignore_warns_and_sends() {
    let typed = gpt_6_request(None).temperature(0.2).options(
        GenerationOptions::default()
            .top_p(0.9)
            .on_unsupported(OnUnsupported::Ignore),
    );
    let (body, warnings) = sent_with_warnings(&chat(&OPENAI, GPT_6_SOL), typed.clone());
    assert_eq!(body.get("temperature"), None, "{body}");
    assert_eq!(body.get("top_p"), None, "{body}");
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(warnings[0].contains("option=temperature"), "{warnings:?}");
    assert!(warnings[1].contains("option=top_p"), "{warnings:?}");
    let (body, warnings) = sent_with_warnings(&responses_wire(GPT_6_SOL), typed);
    assert_eq!(body.get("temperature"), None, "{body}");
    assert_eq!(body.get("top_p"), None, "{body}");
    assert_eq!(warnings.len(), 2, "{warnings:?}");

    let raw = lenient(gpt_6_request(Some(json!({"top_p": 0.9, "logprobs": true}))));
    let (body, warnings) = sent_with_warnings(&chat(&OPENAI, GPT_6_LUNA), raw);
    assert_eq!(body["top_p"], json!(0.9), "{body}");
    assert_eq!(body["logprobs"], json!(true), "{body}");
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(
        warnings
            .iter()
            .all(|warning| warning.contains("sent as written")),
        "{warnings:?}"
    );

    // A raw key over a typed one is the raw key, so it is sent.
    let both = gpt_6_request(Some(json!({"temperature": 0.7}))).temperature(0.2);
    let (body, _) = sent_with_warnings(&chat(&OPENAI, GPT_6_SOL), lenient(both));
    assert_eq!(body["temperature"], json!(0.7), "{body}");
}

#[test]
fn gpt_6_explicit_effort_other_than_none_still_rejects_sampling() {
    let chat_request = gpt_6_request(Some(json!({"reasoning_effort": "high"}))).temperature(0.2);
    assert!(on_chat(GPT_6_SOL, strict(chat_request)).is_err());
    let responses_request =
        gpt_6_request(Some(json!({"reasoning": {"effort": "low"}}))).temperature(0.2);
    assert!(on_responses(GPT_6_LUNA, strict(responses_request)).is_err());
}

#[test]
fn gpt_6_effort_none_allows_sampling_on_sol_and_luna() {
    for model in [GPT_6_SOL, GPT_6_LUNA] {
        let off = |request: CompletionRequest| {
            request.options(GenerationOptions::default().reasoning(Reasoning::Off))
        };
        let body =
            on_chat(model, off(gpt_6_request(None).temperature(0.2))).expect("effort none samples");
        assert_eq!(body["temperature"], json!(0.2));
        assert_eq!(body["reasoning_effort"], "none");
        let body = on_responses(model, off(gpt_6_request(None).temperature(0.2)))
            .expect("effort none samples");
        assert_eq!(body["temperature"], json!(0.2));
    }
}

#[test]
fn gpt_6_astras_sampling_error_does_not_suggest_effort_none() {
    let error = on_responses(GPT_6_ASTRA, strict(gpt_6_request(None).temperature(0.2)))
        .expect_err("rejected");
    assert!(!error.contains("to `none`"), "{error}");
}

/// Astra and 6.1 Sol never call tools through Chat Completions: with an
/// option set, `Error` refuses them under the name `tools` and `Ignore`
/// sends them as built with a warning. Sol and Luna call them at effort
/// `none`; at their default effort the request reaches OpenAI, whose 400
/// names the fix (the recorded `gpt_6_luna` session pins that reply).
#[test]
fn gpt_6_chat_completions_tools_fail_for_astra_and_reach_the_api_on_sol_and_luna() {
    let result = on_prepared_chat(GPT_6_ASTRA, strict(with_tool(gpt_6_request(None))));
    assert_eq!(
        refusal(result),
        (
            "tools".to_owned(),
            "openai".to_owned(),
            GPT_6_ASTRA.to_owned()
        )
    );
    let error = on_chat(GPT_6_ASTRA, strict(with_tool(gpt_6_request(None)))).expect_err("astra");
    assert!(error.contains("Responses"), "{error}");
    let error = on_chat(
        GPT_6_ASTRA,
        strict(with_tool(gpt_6_request(Some(
            json!({"reasoning_effort": "none"}),
        )))),
    )
    .expect_err("astra never calls tools on chat");
    assert!(error.contains("Responses"), "{error}");
    let (body, warnings) = sent_with_warnings(
        &chat(&OPENAI, GPT_6_ASTRA),
        lenient(with_tool(gpt_6_request(None))),
    );
    assert_eq!(body["tools"][0]["function"]["name"], "lookup");
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("option=tools"), "{warnings:?}");
    for model in [GPT_6_SOL, GPT_6_LUNA] {
        let body =
            on_chat(model, strict(with_tool(gpt_6_request(None)))).expect("sent; OpenAI decides");
        assert_eq!(body["tools"][0]["function"]["name"], "lookup");
        let body = on_chat(
            model,
            with_tool(gpt_6_request(Some(json!({"reasoning_effort": "none"})))),
        )
        .expect("effort none calls tools");
        assert_eq!(body["tools"][0]["function"]["name"], "lookup");
    }
}

#[test]
fn gpt_6_responses_carries_tools_at_any_effort() {
    for model in [GPT_6_ASTRA, GPT_6_SOL, GPT_6_LUNA] {
        let body =
            on_responses(model, strict(with_tool(gpt_6_request(None)))).expect("responses tools");
        assert_eq!(body["tools"][0]["name"], "lookup");
    }
}

#[test]
fn gpt_6_chat_requests_use_the_reasoning_output_cap() {
    for model in [GPT_6_ASTRA, GPT_6_SOL, GPT_6_LUNA] {
        let capped = gpt_6_request(Some(json!({"reasoning_effort": "none"}))).max_tokens(256);
        let body = on_chat(model, capped).expect("encodes");
        assert_eq!(body["max_completion_tokens"], json!(256), "{model}");
        assert_eq!(body.get("max_tokens"), None, "{model}");
    }
}

#[test]
fn gpt_6_rules_leave_other_models_unchecked() {
    for model in ["gpt-6", "gpt-6-sol-mini", "openai/gpt-6-sol"] {
        let sampled = strict(with_tool(gpt_6_request(None)).temperature(0.2));
        assert!(on_chat(model, sampled.clone()).is_ok(), "{model}");
        assert!(on_responses(model, sampled).is_ok(), "{model}");
    }
}

#[test]
fn gpt_6_rules_follow_a_per_request_model_override() {
    let request = strict(gpt_6_request(None).model(GPT_6_SOL).temperature(0.2));
    assert_eq!(
        refusal(on_prepared_chat("gpt-5.6-sol", request.clone())).2,
        GPT_6_SOL
    );
    assert!(on_responses("gpt-5.6-sol", request).is_err());
}

/// GPT-5.1 and later take `temperature`, `top_p` and `logprobs` only at
/// effort `none` (OpenAI's model guidance for GPT-5.2 and GPT-5.4): an effort
/// set to anything else, typed or raw, refuses them on both routes. With no
/// effort set the model is not checked, since its default is `none`.
#[test]
fn gpt_5_1_and_later_refuse_sampling_while_they_reason() {
    for model in ["gpt-5.1", "gpt-5.4", "gpt-5.6-sol"] {
        let high = |request: CompletionRequest| {
            request.options(GenerationOptions::default().reasoning(Effort::High))
        };
        let result = on_prepared_chat(model, high(gpt_6_request(None).temperature(0.2)));
        assert_eq!(refusal(result).0, "temperature", "{model}");
        let result = on_prepared_chat(
            model,
            strict(gpt_6_request(Some(
                json!({"reasoning_effort": "low", "top_p": 0.9}),
            ))),
        );
        assert_eq!(refusal(result).0, "top_p", "{model}");
        let result = on_prepared_responses(model, high(gpt_6_request(None).temperature(0.2)));
        assert_eq!(refusal(result).0, "temperature", "{model}");

        let (body, warnings) = sent_with_warnings(
            &chat(&OPENAI, model),
            gpt_6_request(None).temperature(0.2).options(
                GenerationOptions::default()
                    .reasoning(Effort::High)
                    .on_unsupported(OnUnsupported::Ignore),
            ),
        );
        assert_eq!(body.get("temperature"), None, "{model}: {body}");
        assert_eq!(body["reasoning_effort"], "high", "{model}");
        assert_eq!(warnings.len(), 1, "{model}: {warnings:?}");

        let body = on_chat(
            model,
            gpt_6_request(Some(json!({"reasoning_effort": "high"}))).temperature(0.2),
        )
        .expect("no option set: sent");
        assert_eq!(body["temperature"], json!(0.2), "{model}");
        let body = on_chat(
            model,
            gpt_6_request(None)
                .temperature(0.2)
                .options(GenerationOptions::default().seed(1)),
        )
        .expect("no effort set");
        assert_eq!(body["temperature"], json!(0.2), "{model}");
        let body = on_chat(
            model,
            strict(gpt_6_request(Some(json!({"reasoning_effort": "none"}))).temperature(0.2)),
        )
        .expect("effort none samples");
        assert_eq!(body["temperature"], json!(0.2), "{model}");
    }
}
