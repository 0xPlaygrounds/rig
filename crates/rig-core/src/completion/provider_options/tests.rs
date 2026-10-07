//! The provider-options layer and typed reply extras, through a test-only
//! extension for a fake provider with two routes, `fake.chat` and
//! `fake.responses`.

use super::*;
use crate::completion::options::{
    BaseInput, Mapping, OptionFields, OptionMap, RawAt, check, param, request_params,
};
use crate::completion::{
    CompletionResponse, GenerationOptions, OnUnsupported, Usage, message::Origin,
};
use serde_json::json;

/// The fake provider's marker.
struct Fake;

/// Another provider, whose entry the fake wire never reads.
struct Other;

#[derive(Clone, Debug, Default, Serialize)]
struct Shared {
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sampling: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct Chat {
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct Responses {
    #[serde(skip_serializing_if = "Option::is_none")]
    truncation: Option<String>,
}

/// The fake provider's options: one shared section and one per route. The
/// chat route cannot store a conversation.
#[derive(Clone, Debug, Default, Serialize)]
struct FakeOptions {
    #[serde(rename = "*")]
    shared: Shared,
    #[serde(rename = "fake.chat")]
    chat: Chat,
    #[serde(rename = "fake.responses")]
    responses: Responses,
}

impl ExtensionOptions for FakeOptions {
    fn unsupported(
        &self,
        target: &dyn ReplayTarget,
        _request: &CompletionRequest,
    ) -> Vec<(&'static str, String)> {
        let mut refused = Vec::new();
        if target.api().as_str() == "fake.chat" {
            refused.push(("store", "the chat route stores nothing".to_owned()));
            refused.push(("logit_bias", "the chat model takes no bias".to_owned()));
        }
        refused
    }
}

#[derive(Debug, PartialEq, Deserialize)]
struct FakeExtras {
    route: String,
    cost: Option<f64>,
}

impl ReplyExtras for FakeExtras {
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Ok(Self {
            route: api.as_str().to_owned(),
            cost: Option::<f64>::deserialize(raw.get("cost").unwrap_or(&Value::Null))?,
        })
    }
}

impl ProviderExtension for Fake {
    const PROVIDER: &'static str = "fake";
    type Options = FakeOptions;
    type Extras = FakeExtras;
}

impl ProviderExtension for Other {
    const PROVIDER: &'static str = "other";
    type Options = FakeOptions;
    type Extras = FakeExtras;
}

/// A fake wire on `api`: `top_p` sent under `sampling`, every other option
/// refused.
#[derive(Debug)]
struct Wire(&'static str);

impl ReplayTarget for Wire {
    fn api(&self) -> Api {
        Api::from_static(self.0)
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

    fn map_options(&self, _request: &CompletionRequest, fields: OptionFields<'_>) -> OptionMap {
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
            reasoning: Mapping::of(reasoning, |_| no()),
            cache: Mapping::of(cache, |_| no()),
            service_tier: Mapping::of(service_tier, |_| no()),
            verbosity: Mapping::of(verbosity, |_| no()),
            parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| no()),
            top_p: Mapping::of(top_p, |p| Mapping::Send(json!({"sampling": {"top_p": p}}))),
            seed: Mapping::of(seed, |_| no()),
            stop: Mapping::of_stop(stop, |_| no()),
        }
    }
}

fn no() -> Mapping {
    Mapping::unsupported("not on the fake wire")
}

const CHAT: Wire = Wire("fake.chat");
const RESPONSES: Wire = Wire("fake.responses");

fn base(_: &mut BaseInput<'_>) -> Result<Map<String, Value>, crate::error::EncodeError> {
    Ok(Map::from_iter([(
        "sampling".to_owned(),
        json!({"temperature": 0.5}),
    )]))
}

fn options(top_k: u32) -> FakeOptions {
    FakeOptions {
        shared: Shared {
            top_k: Some(top_k),
            ..Shared::default()
        },
        ..FakeOptions::default()
    }
}

fn provider_options(options: &FakeOptions) -> ProviderOptions {
    ProviderOptions::new()
        .with::<Fake>(options)
        .unwrap_or_else(|error| panic!("{error}"))
}

fn request(options: &FakeOptions) -> CompletionRequest {
    CompletionRequest::new("hi").provider_options(provider_options(options))
}

/// `value`, an object, as a map.
fn sections(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => panic!("an object: {other}"),
    }
}

fn body(target: &Wire, request: &CompletionRequest, raw_at: RawAt) -> Map<String, Value> {
    let body = request_params(target, request, base, raw_at, &[])
        .unwrap_or_else(|error| panic!("{error}"));
    sections(serde_json::to_value(&body).unwrap_or_default())
}

#[test]
fn an_entry_is_keyed_by_the_provider_and_holds_its_sections() {
    let set = provider_options(&options(4));
    assert!(set.contains::<Fake>());
    assert!(!set.contains::<Other>());
    assert_eq!(
        set.get::<Fake>(),
        Some(&sections(json!({"*": {"top_k": 4}})))
    );
    assert_eq!(set.get::<Other>(), None);
    let mut set = set;
    set.remove::<Fake>();
    assert!(set.is_empty());
}

#[test]
fn options_that_write_nothing_leave_no_entry() {
    let mut set = provider_options(&options(4));
    set.insert::<Fake>(&FakeOptions::default())
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(set.is_empty());
    let request = CompletionRequest::new("hi").provider_options(set);
    let value = serde_json::to_value(&request).unwrap_or_default();
    assert!(value.get("provider_options").is_none(), "{value}");
}

#[test]
fn options_must_be_an_object_of_object_sections() {
    #[derive(Clone, Debug, Serialize)]
    struct Flat {
        top_k: u32,
    }
    impl ExtensionOptions for Flat {}
    struct Bad;
    impl ProviderExtension for Bad {
        const PROVIDER: &'static str = "bad";
        type Options = Flat;
        type Extras = FakeExtras;
    }
    let error = ProviderOptions::new().with::<Bad>(&Flat { top_k: 1 });
    assert!(
        matches!(error, Err(OptionsError::NotSections { provider: "bad" })),
        "{error:?}"
    );
    let parsed = serde_json::from_value::<ProviderOptions>(json!({"fake": {"*": 1}}));
    assert!(parsed.is_err());
}

#[test]
fn the_serialized_form_is_the_sections_and_reads_back() {
    let request = request(&options(4));
    let value = serde_json::to_value(&request).unwrap_or_default();
    assert_eq!(
        value.get("provider_options"),
        Some(&json!({"fake": {"*": {"top_k": 4}}}))
    );
    let back = serde_json::from_value::<CompletionRequest>(value)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(back.provider_options, request.provider_options);
}

#[test]
fn the_layer_sits_between_the_mapped_options_and_additional_params() {
    let mut request = request(&FakeOptions {
        shared: Shared {
            sampling: Some(json!({"top_p": 0.7, "top_k": 3, "min_p": 0.1})),
            ..Shared::default()
        },
        ..FakeOptions::default()
    })
    .options(GenerationOptions::default().top_p(0.5));
    request.additional_params = Some(json!({"sampling": {"top_k": 9}}));
    let body = body(&RESPONSES, &request, RawAt::Top);
    assert_eq!(
        body.get("sampling"),
        Some(&json!({"temperature": 0.5, "top_p": 0.7, "top_k": 9, "min_p": 0.1}))
    );
}

#[test]
fn the_layer_is_top_level_wherever_the_raw_layer_goes() {
    let mut request = request(&options(4));
    request.additional_params = Some(json!({"top_k": 9}));
    let body = body(
        &RESPONSES,
        &request,
        RawAt::Under("/additionalModelRequestFields"),
    );
    assert_eq!(body.get("top_k"), Some(&json!(4)));
    assert_eq!(
        body.get("additionalModelRequestFields"),
        Some(&json!({"top_k": 9}))
    );
}

#[test]
fn the_route_section_beats_the_shared_one_and_another_route_is_skipped() {
    let request = request(&FakeOptions {
        shared: Shared {
            top_k: Some(4),
            ..Shared::default()
        },
        chat: Chat {
            top_k: Some(8),
            ..Chat::default()
        },
        responses: Responses {
            truncation: Some("auto".to_owned()),
        },
    });
    let chat = body(&CHAT, &request, RawAt::Top);
    assert_eq!(chat.get("top_k"), Some(&json!(8)));
    assert_eq!(chat.get("truncation"), None);

    let capture = crate::test_utils::TraceCapture::default();
    let responses = tracing::subscriber::with_default(capture.subscriber(), || {
        body(&RESPONSES, &request, RawAt::Top)
    });
    assert_eq!(responses.get("top_k"), Some(&json!(4)));
    assert_eq!(responses.get("truncation"), Some(&json!("auto")));
    let skipped = capture
        .events()
        .into_iter()
        .filter(|event| event.level == tracing::Level::DEBUG)
        .filter(|event| event.fields.get("section") == Some(&json!("fake.chat")))
        .count();
    assert_eq!(skipped, 1);
    assert!(capture.warnings().is_empty(), "{:?}", capture.warnings());
}

#[test]
fn a_wire_reads_only_its_own_providers_entry() {
    let request = CompletionRequest::new("hi").provider_options(
        ProviderOptions::new()
            .with::<Other>(&options(4))
            .unwrap_or_else(|error| panic!("{error}")),
    );
    let body = body(&RESPONSES, &request, RawAt::Top);
    assert_eq!(body.get("top_k"), None);
}

#[test]
fn a_refused_field_is_named_by_provider_section_and_field() {
    let request = request(&FakeOptions {
        shared: Shared {
            store: Some(true),
            ..Shared::default()
        },
        ..FakeOptions::default()
    });
    let error = request_params(&CHAT, &request, base, RawAt::Top, &[])
        .expect_err("the chat route refuses store");
    let refused = error.unsupported_option().cloned();
    assert_eq!(
        refused.as_ref().map(|refused| refused.option.as_ref()),
        Some("fake.*.store")
    );
    assert_eq!(
        refused.as_ref().map(|refused| refused.provider.as_str()),
        Some("fake")
    );
    assert_eq!(
        refused.map(|refused| refused.reason),
        Some("the chat route stores nothing".to_owned())
    );
    // The other route sends it.
    assert_eq!(
        body(&RESPONSES, &request, RawAt::Top).get("store"),
        Some(&json!(true))
    );
    // A refusal of a field that is not set reports nothing.
    assert!(body(&CHAT, &self::request(&options(4)), RawAt::Top).contains_key("top_k"));
}

#[test]
fn a_refused_route_field_names_its_route_section() {
    let request = request(&FakeOptions {
        chat: Chat {
            logit_bias: Some(json!({"42": 1})),
            ..Chat::default()
        },
        ..FakeOptions::default()
    });
    let error = request_params(&CHAT, &request, base, RawAt::Top, &[])
        .expect_err("the chat model refuses logit_bias");
    assert_eq!(
        error
            .unsupported_option()
            .map(|refused| refused.option.to_string()),
        Some("fake.fake.chat.logit_bias".to_owned())
    );
}

#[test]
fn under_ignore_a_refused_field_is_warned_once_and_not_sent() {
    let mut request = request(&FakeOptions {
        shared: Shared {
            store: Some(true),
            top_k: Some(4),
            ..Shared::default()
        },
        ..FakeOptions::default()
    })
    .options(GenerationOptions::default().on_unsupported(OnUnsupported::Ignore));
    let capture = crate::test_utils::TraceCapture::default();
    let body = tracing::subscriber::with_default(capture.subscriber(), || {
        check(&CHAT, &mut request).unwrap_or_else(|error| panic!("{error}"));
        body(&CHAT, &request, RawAt::Top)
    });
    assert_eq!(body.get("store"), None);
    assert_eq!(body.get("top_k"), Some(&json!(4)));
    let warnings = capture.warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("option=fake.*.store"), "{warnings:?}");
    // `check` took it out of the request, so the encode reports nothing.
    assert_eq!(
        request.provider_options.get::<Fake>(),
        Some(&sections(json!({"*": {"top_k": 4}})))
    );

    // `request_params` alone skips it too.
    let unchecked = self::request(&FakeOptions {
        shared: Shared {
            store: Some(true),
            ..Shared::default()
        },
        ..FakeOptions::default()
    })
    .options(GenerationOptions::default().on_unsupported(OnUnsupported::Ignore));
    assert_eq!(self::body(&CHAT, &unchecked, RawAt::Top).get("store"), None);
}

#[test]
fn a_deserialized_entry_is_sent_as_written() {
    let request = request(&FakeOptions {
        shared: Shared {
            store: Some(true),
            ..Shared::default()
        },
        ..FakeOptions::default()
    });
    let value = serde_json::to_value(&request).unwrap_or_default();
    let back = serde_json::from_value::<CompletionRequest>(value)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        body(&CHAT, &back, RawAt::Top).get("store"),
        Some(&json!(true))
    );
}

#[test]
fn param_reads_the_provider_layer_under_the_raw_one() {
    let mut request = request(&FakeOptions {
        shared: Shared {
            store: Some(true),
            top_k: Some(4),
            ..Shared::default()
        },
        ..FakeOptions::default()
    })
    .options(GenerationOptions::default().top_p(0.5));
    assert_eq!(param(&RESPONSES, &request, "top_k"), Some(json!(4)));
    assert_eq!(param(&RESPONSES, &request, "store"), Some(json!(true)));
    assert_eq!(
        param(&RESPONSES, &request, "sampling"),
        Some(json!({"top_p": 0.5}))
    );
    // A field the route refuses adds nothing.
    assert_eq!(param(&CHAT, &request, "store"), None);
    request.additional_params = Some(json!({"top_k": 9}));
    assert_eq!(param(&RESPONSES, &request, "top_k"), Some(json!(9)));
}

#[test]
fn the_base_builder_sees_the_provider_layer() {
    let request = request(&options(4));
    let body = request_params(
        &RESPONSES,
        &request,
        |input| {
            Ok(Map::from_iter([(
                "seen".to_owned(),
                input.param("top_k").cloned().unwrap_or(Value::Null),
            )]))
        },
        RawAt::Top,
        &[],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(body.get("seen"), Some(&json!(4)));
}

#[test]
fn a_runs_entry_replaces_the_agents_for_its_provider() {
    let agent = ProviderOptions::new()
        .with::<Fake>(&options(4))
        .and_then(|set| set.with::<Other>(&options(5)))
        .unwrap_or_else(|error| panic!("{error}"));
    let run = provider_options(&FakeOptions {
        shared: Shared {
            store: Some(true),
            ..Shared::default()
        },
        ..FakeOptions::default()
    });
    let resolved = agent.clone().overlay(&run);
    assert_eq!(resolved.get::<Fake>(), run.get::<Fake>());
    assert_eq!(resolved.get::<Other>(), agent.get::<Other>());
    // The run's typed options came along: the chat route still refuses.
    let request = CompletionRequest::new("hi").provider_options(resolved);
    assert!(request_params(&CHAT, &request, base, RawAt::Top, &[]).is_err());
    assert_eq!(agent.clone().overlay(&ProviderOptions::new()), agent);
}

fn reply(provider: &str, raw: Value) -> CompletionResponse {
    CompletionResponse::new(
        Vec::new(),
        Usage::new(),
        Origin::new("fake.responses", provider, "fake-1"),
        raw,
    )
}

#[test]
fn extras_are_read_only_from_the_providers_own_reply() {
    let reply = reply("fake", json!({"cost": 0.25}));
    let extras = reply
        .extras::<Fake>()
        .map(|extras| extras.unwrap_or_else(|error| panic!("{error}")));
    assert_eq!(
        extras,
        Some(FakeExtras {
            route: "fake.responses".to_owned(),
            cost: Some(0.25),
        })
    );
    assert!(reply.extras::<Other>().is_none());
}

#[test]
fn extras_of_the_wrong_shape_are_an_error() {
    let reply = reply("fake", json!({"cost": "free"}));
    assert!(matches!(reply.extras::<Fake>(), Some(Err(_))));
}

/// A request and what carries one stay unwind safe with provider options
/// in them, as they were before options held typed entries.
#[test]
fn requests_stay_unwind_safe() {
    fn unwind_safe<T: std::panic::UnwindSafe + std::panic::RefUnwindSafe>() {}
    unwind_safe::<ProviderOptions>();
    unwind_safe::<CompletionRequest>();
    unwind_safe::<crate::effect::EffectKind>();
    unwind_safe::<crate::serve::Decision>();
}
