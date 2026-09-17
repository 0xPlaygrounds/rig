//! Provider bindings as data (`bus::binding`): a `ProviderBinding` is
//! serde, holds a credential *reference* and never a secret, and becomes a
//! served handler only when the host materializes it through its own
//! resolver and transport.
//!
//! | claim | test |
//! |---|---|
//! | the component round-trips verbatim, tagged by provider and naming its dialect, with no credential in it; an unknown provider and an unknown dialect are both refused by the reader | `a_binding_round_trips_verbatim` |
//! | materializing binds on the binding's own entity, with the descriptor a hand-registered adapter produces, for every provider configuration | `a_materialized_binding_describes_itself_as_a_hand_registered_adapter` |
//! | a dialect the provider does not speak, and a value its typed options do not take, are refused where the names live: the config does not deserialize | `a_dialect_the_provider_does_not_speak_is_refused_by_the_reader` |
//! | a provider's options are its config's own typed fields: the `anthropic-version` and `anthropic-beta` headers, and the endpoint the route names, are what the world's config says | `provider_options_are_the_configs_own_fields` |
//! | the built client sends to the binding's base URL through the host's transport, one construction per binding | `the_host_transport_is_built_once_per_binding_and_sends_to_the_base_url` |
//! | all or nothing: one refused credential registers nothing | `materialization_is_all_or_nothing` |
//! | precedence: a served key wins, the binding is reported kept | `an_existing_handler_wins_over_a_binding` |
//! | a replayer under the key wins: no transport, no credential | `a_replayer_wins_and_no_transport_is_built` |
//! | duplicate keys, key mismatch, no materializer are refused before any registration | `refusals_are_deterministic_and_register_nothing` |
//! | no secret in `Debug`, in diagnostics, in JSON | `secrets_never_leave_the_resolver` |
//! | a checkpoint saves the binding with its bound descriptor and no secret; loading spawns it bound and unserved, resolving nothing; materializing after the load serves it under the saved descriptor; saved again it is the same data | `a_checkpoint_loads_its_bindings_as_data_and_materializes_on_the_hosts_word` |
//! | a load is refused, before any spawn, for a served key of another family; a served key of the same family keeps the handler and attaches the binding | `a_checkpoint_load_validates_its_bindings` |
//! | a binding whose built descriptor differs from the saved one is refused (`DescriptorDrift`), nothing served | `a_loaded_binding_that_would_build_another_descriptor_is_refused` |
//! | the system: a refusal is left in `MaterializeFailed`, a success clears it | `the_materialize_system_reports_through_the_resource` |
//! | a binding beside its own unserved `Bound` whose key another entity serves (a host registration made before it, a replayer) or merely holds is kept: the served handler untouched, its own `Bound` untouched, nothing built | `a_binding_beside_its_own_bound_keeps_the_key_served_elsewhere` |
//! | a binding whose registration `bind` would refuse (its key bound to another family elsewhere) beside a good one: checked before the first registration — the good one materializes, the other is kept, nothing half-registered | `a_key_bound_to_another_family_is_kept_beside_a_good_binding` |

use crate::bus_support;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use bevy_ecs::prelude::*;
use rig_core::{
    driver::Bind,
    effect::{HandlerDescriptor, HandlerKey},
    error::ErrorKind,
    http_client::BoxedHttpClient,
    providers::{
        anthropic::wire::{ANTHROPIC, Anthropic, Dialect as AnthropicDialect, ZAI},
        gemini::Gemini,
        openai::wire::{DEEPSEEK, Dialect as OpenAiDialect, OPENAI, OpenAI, Route, VENICE},
    },
    serve::{ErasedHandler, adapters::CompletionAdapter},
    test_utils::RecordingHttpClient,
};
use rig_ecs::{
    bus::{
        Bound, CredentialRef, EffectLogResource, EffectOutcome, Handler, Handlers,
        MaterializeError, MaterializeFailed, MaterializeReport, Materializer, PendingEffect,
        ProviderBinding, ProviderConfig, Replay, Secret,
    },
    checkpoint::{Checkpoint, load_world, save_world},
};
use rig_effect_log::EffectLogRecorder;

const KEY: &str = "t/model:default";
const BASE: &str = "http://cassette.invalid/v1";
const SENTINEL: &str = "sentinel-secret-value-3f9a";

const ANTHROPIC_BODY: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#;
const CHAT_BODY: &str = r#"{"id":"c","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;

/// A unary completion effect with the `max_tokens` Anthropic insists on.
fn completion() -> rig_core::effect::EffectKind {
    let mut request = bus_support::request();
    request.max_tokens = Some(64);
    rig_core::effect::EffectKind::Completion {
        request,
        stream: false,
    }
}

/// The binding under test on `config`: the key, the model and the
/// credential *reference* every test here shares.
fn binding(config: ProviderConfig) -> ProviderBinding {
    ProviderBinding::new(KEY, config, "model-x", "cassette").labelled("default")
}

/// Anthropic itself at the cassette's base URL, credential-less: what a
/// scene carries.
fn anthropic() -> ProviderConfig {
    anthropic_on(&ANTHROPIC)
}

/// A Messages-format gateway at the cassette's base URL.
fn anthropic_on(dialect: &AnthropicDialect) -> ProviderConfig {
    ProviderConfig::Anthropic(
        Anthropic::with_dialect(Secret::default(), dialect).with_base_url(BASE),
    )
}

/// An OpenAI-shaped gateway at the cassette's base URL, on `route` when the
/// configuration names one rather than taking the dialect's flagship.
fn openai_on(dialect: &OpenAiDialect, route: Option<Route>) -> ProviderConfig {
    let config = OpenAI::with_key(dialect, Secret::default()).with_base_url(BASE);
    ProviderConfig::OpenAi(match route {
        Some(route) => config.with_route(route),
        None => config,
    })
}

/// Gemini at the cassette's base URL.
fn gemini() -> ProviderConfig {
    ProviderConfig::Gemini(Gemini::new(Secret::default()).with_base_url(BASE))
}

/// A materializer over a sentinel credential and a recording transport:
/// the transport it hands out, and how many it built.
fn materializer(body: &'static str) -> (Materializer, RecordingHttpClient, Arc<AtomicUsize>) {
    let transport = RecordingHttpClient::new(body);
    let built = Arc::new(AtomicUsize::new(0));
    let handed = transport.clone();
    let count = built.clone();
    let materializer = Materializer::new(
        |credential: &CredentialRef| {
            if credential.as_str() == "cassette" {
                Ok(Secret::from(SENTINEL))
            } else {
                Err("not a cassette".to_owned())
            }
        },
        move || {
            count.fetch_add(1, Ordering::SeqCst);
            BoxedHttpClient::new(handed.clone())
        },
    );
    (materializer, transport, built)
}

fn panicking_materializer() -> Materializer {
    Materializer::new(
        |credential: &CredentialRef| panic!("a credential was resolved: {credential}"),
        || panic!("a transport was built"),
    )
}

/// How many handler entities are served.
fn served(world: &mut World) -> usize {
    world.query::<&Handler>().iter(world).count()
}

#[test]
fn a_binding_round_trips_verbatim() {
    // The other providers, tagged before the subject shadows the helper.
    let deepseek = binding(openai_on(&DEEPSEEK, None));
    let gemini = binding(gemini());
    let binding = binding(ProviderConfig::Anthropic(
        Anthropic::new(Secret::default())
            .with_base_url(BASE)
            .with_beta("b-1"),
    ));
    let json = serde_json::to_string(&binding).unwrap();
    assert!(json.contains(r#""credential":"cassette""#), "{json}");
    // The provider is the config's own tag, and the gateway its dialect's
    // name: both data, neither a variant of this crate's.
    assert!(json.contains(r#""provider":"anthropic""#), "{json}");
    assert!(json.contains(r#""dialect":"anthropic""#), "{json}");
    // A typed provider option travels as itself.
    assert!(json.contains(r#""betas":["b-1"]"#), "{json}");
    // And no credential does: the config's key is the empty secret, which
    // serializes redacted.
    assert!(binding.config.credential().is_empty());
    assert!(json.contains(r#""api_key":"[redacted]""#), "{json}");
    let again: ProviderBinding = serde_json::from_str(&json).unwrap();
    assert_eq!(again, binding);
    assert_eq!(again.credential, CredentialRef::new("cassette"));
    // The other providers tag themselves the same way, dialect included.
    let openai = serde_json::to_string(&deepseek).unwrap();
    assert!(openai.contains(r#""provider":"open_ai""#), "{openai}");
    assert!(openai.contains(r#""dialect":"deepseek""#), "{openai}");
    assert_eq!(
        serde_json::from_str::<ProviderBinding>(&openai).unwrap(),
        deepseek
    );
    let gemini = serde_json::to_string(&gemini).unwrap();
    assert!(gemini.contains(r#""provider":"gemini""#), "{gemini}");

    // An unknown provider is refused by the reader, before anything is
    // spawned, and the error names it.
    let unknown_provider = json.replace(r#""provider":"anthropic""#, r#""provider":"telepathy""#);
    let error = serde_json::from_str::<ProviderBinding>(&unknown_provider).unwrap_err();
    assert!(
        error.to_string().contains("unknown variant `telepathy`"),
        "{error}"
    );
    // So is a dialect name no Messages-format gateway answers to: the
    // dialect is where gateway names live, so that is where one that does
    // not exist is refused.
    let unknown_dialect = json.replace(r#""dialect":"anthropic""#, r#""dialect":"telepathy""#);
    let error = serde_json::from_str::<ProviderBinding>(&unknown_dialect).unwrap_err();
    assert!(error.to_string().contains("telepathy"), "{error}");
}

/// Every provider configuration, and — because a gateway is a dialect name
/// inside the config rather than a variant of this crate's — two gateways
/// no enum arm could have expressed: DeepSeek, which used to need its own
/// kind, and Venice, which never had one at all. Both endpoints are
/// reached the same way, through the config's `route`.
#[test]
fn a_materialized_binding_describes_itself_as_a_hand_registered_adapter() {
    type ByHand = Box<dyn Fn(BoxedHttpClient) -> ErasedHandler>;
    let cases: Vec<(&str, ProviderConfig, ByHand)> = vec![
        (
            "anthropic",
            anthropic_on(&ANTHROPIC),
            Box::new(|http| {
                ErasedHandler::new(CompletionAdapter::new(
                    "default",
                    Anthropic::new(SENTINEL)
                        .with_base_url(BASE)
                        .bind(http)
                        .completion("model-x"),
                ))
            }),
        ),
        (
            "anthropic/zai",
            anthropic_on(&ZAI),
            Box::new(|http| {
                ErasedHandler::new(CompletionAdapter::new(
                    "default",
                    Anthropic::with_dialect(SENTINEL, &ZAI)
                        .with_base_url(BASE)
                        .bind(http)
                        .completion("model-x"),
                ))
            }),
        ),
        (
            "openai/chat",
            openai_on(&OPENAI, Some(Route::Chat)),
            Box::new(|http| {
                ErasedHandler::new(CompletionAdapter::new(
                    "default",
                    OpenAI::new(SENTINEL)
                        .with_base_url(BASE)
                        .bind(http)
                        .chat("model-x"),
                ))
            }),
        ),
        (
            "deepseek/chat",
            openai_on(&DEEPSEEK, None),
            Box::new(|http| {
                ErasedHandler::new(CompletionAdapter::new(
                    "default",
                    OpenAI::with_key(&DEEPSEEK, SENTINEL)
                        .with_base_url(BASE)
                        .bind(http)
                        .chat("model-x"),
                ))
            }),
        ),
        (
            "venice/chat",
            openai_on(&VENICE, None),
            Box::new(|http| {
                ErasedHandler::new(CompletionAdapter::new(
                    "default",
                    OpenAI::with_key(&VENICE, SENTINEL)
                        .with_base_url(BASE)
                        .bind(http)
                        .chat("model-x"),
                ))
            }),
        ),
        (
            "openai/responses",
            openai_on(&OPENAI, Some(Route::Responses)),
            Box::new(|http| {
                ErasedHandler::new(CompletionAdapter::new(
                    "default",
                    OpenAI::new(SENTINEL)
                        .with_base_url(BASE)
                        .bind(http)
                        .responses("model-x"),
                ))
            }),
        ),
        (
            "gemini",
            gemini(),
            Box::new(|http| {
                ErasedHandler::new(CompletionAdapter::new(
                    "default",
                    Gemini::new(SENTINEL)
                        .with_base_url(BASE)
                        .bind(http)
                        .completion("model-x"),
                ))
            }),
        ),
    ];
    for (named, config, by_hand) in cases {
        let mut app = bus_support::app();
        let (materializer, transport, _) = materializer(ANTHROPIC_BODY);
        app.world_mut().insert_resource(materializer);
        let entity = app.world_mut().spawn(binding(config)).id();
        assert!(app.world().get::<Bound>(entity).is_none());
        let report = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
        assert_eq!(
            report,
            MaterializeReport {
                materialized: vec![HandlerKey::from(KEY)],
                kept: vec![],
            },
            "{named}"
        );
        // The binding's own entity is the handler entity.
        let bound = app
            .world()
            .get::<Bound>(entity)
            .unwrap_or_else(|| panic!("{named}: bound"));
        assert_eq!(bound.key, HandlerKey::from(KEY));
        assert!(app.world().get::<Handler>(entity).is_some());
        // The descriptor is the one `CompletionAdapter::new("default", model)`
        // registered by hand under the key would produce.
        let by_hand = by_hand(BoxedHttpClient::new(transport.clone()));
        let mut hand = bus_support::app();
        let hand_entity = Handlers::with(hand.world_mut(), |handlers| {
            handlers.register_erased(KEY, by_hand)
        })
        .unwrap()
        .unwrap();
        hand.world_mut().flush();
        let expected: HandlerDescriptor = hand
            .world()
            .get::<Bound>(hand_entity)
            .unwrap()
            .descriptor
            .clone();
        assert_eq!(bound.descriptor, expected, "{named}: the bound descriptor");
        assert_eq!(
            serde_json::to_value(&bound.descriptor).unwrap(),
            serde_json::to_value(&expected).unwrap(),
            "{named}: the descriptor's JSON (what the policy hash folds)"
        );
        // Materializing again finds the key served and builds nothing.
        let again = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
        assert_eq!(again.kept, vec![HandlerKey::from(KEY)], "{named}");
        assert!(again.materialized.is_empty(), "{named}");
    }
}

/// A dialect a provider does not speak — and a value a provider option does
/// not take — is refused by the reader, where the names and the types live:
/// the binding does not deserialize, so no world can hold one to
/// materialize. Nothing is left for `materialize_bindings` to refuse, which
/// is why it has no dialect and no option error any more.
#[test]
fn a_dialect_the_provider_does_not_speak_is_refused_by_the_reader() {
    let anthropic = serde_json::to_string(&binding(anthropic())).unwrap();
    let openai = serde_json::to_string(&binding(openai_on(&OPENAI, Some(Route::Chat)))).unwrap();
    for (what, json, offender) in [
        // A name no OpenAI-shaped gateway answers to.
        (
            "openai/telepathy",
            openai.replace(r#""dialect":"openai""#, r#""dialect":"telepathy""#),
            "telepathy",
        ),
        // Anthropic's dialects are their own list: an OpenAI gateway name is
        // not one of them.
        (
            "anthropic/deepseek",
            anthropic.replace(r#""dialect":"anthropic""#, r#""dialect":"deepseek""#),
            "deepseek",
        ),
        // And the other way round: Anthropic's own name is not an
        // OpenAI-shaped provider. (Most Messages-format gateways — zai,
        // minimax, moonshot, xiaomimimo — serve both shapes, and *are*
        // dialects of both lists: which shape a name belongs to is
        // rig-core's to say, and each list says it for itself.)
        (
            "openai/anthropic",
            openai.replace(r#""dialect":"openai""#, r#""dialect":"anthropic""#),
            "anthropic",
        ),
        // A typed option refuses a value it does not take, naming what it
        // wanted — which is what an untyped parameter bag had to check by
        // hand.
        (
            "anthropic/betas",
            anthropic.replace(r#""betas":[]"#, r#""betas":"b-1""#),
            "expected a sequence",
        ),
        (
            "anthropic/version",
            anthropic.replace(r#""version":"2023-06-01""#, r#""version":7"#),
            "expected a string",
        ),
        (
            "openai/route",
            openai.replace(r#""route":"Chat""#, r#""route":"Telepathy""#),
            "Telepathy",
        ),
    ] {
        let error = serde_json::from_str::<ProviderBinding>(&json).unwrap_err();
        assert!(
            error.to_string().contains(offender),
            "{what}: {error} does not name {offender}"
        );
    }
    // Gemini speaks one dialect, its own, and has no field to name another:
    // a gateway cannot be smuggled into a Gemini binding at all.
    let gemini_json = serde_json::to_string(&binding(gemini())).unwrap();
    assert!(!gemini_json.contains("dialect"), "{gemini_json}");
    let smuggled = gemini_json.replace(
        r#""provider":"gemini""#,
        r#""provider":"gemini","dialect":"openai""#,
    );
    assert_eq!(
        serde_json::from_str::<ProviderBinding>(&smuggled).unwrap(),
        binding(gemini()),
        "a dialect named at a Gemini config reaches no gateway"
    );
}

/// A provider's options are fields of its own configuration, so what the
/// world holds is what the socket sees: Anthropic's `version` and `betas`
/// as headers, and the endpoint the OpenAI config's `route` names.
#[test]
fn provider_options_are_the_configs_own_fields() {
    // Anthropic: the version and the beta flags the config carries.
    let mut app = bus_support::app();
    let (host, transport, _) = materializer(ANTHROPIC_BODY);
    app.world_mut().insert_resource(host);
    let entity = app.world_mut().spawn(binding(anthropic())).id();
    // Set through the component, in the world: the config is the binding's
    // data, and a host editing it edits what will be built.
    app.world_mut()
        .entity_mut(entity)
        .get_mut::<ProviderBinding>()
        .unwrap()
        .config = ProviderConfig::Anthropic(
        Anthropic::new(Secret::default())
            .with_base_url(BASE)
            .with_version("2024-01-01")
            .with_beta("b-1")
            .with_beta("b-2"),
    );
    rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(KEY, completion()))
        .id();
    bus_support::tick_until(&mut app, "the bound client answers", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(
        bus_support::text_of(&app.world().get::<EffectOutcome>(effect).unwrap().0),
        "hi"
    );
    let requests = transport.requests();
    let header = |name: &str| {
        requests[0]
            .headers
            .get(name)
            .map(|value| value.to_str().unwrap().to_owned())
    };
    assert_eq!(header("anthropic-version").as_deref(), Some("2024-01-01"));
    assert_eq!(header("anthropic-beta").as_deref(), Some("b-1,b-2"));

    // OpenAI: the route the config names is the endpoint the request goes
    // to, whichever the dialect's flagship is. The cassette answers both
    // with a chat reply, so what each cell pins is where the request went —
    // the reply's shape is the wire's business, not the binding's.
    for (named, route, path) in [
        ("chat", Route::Chat, "/chat/completions"),
        ("responses", Route::Responses, "/responses"),
    ] {
        let mut app = bus_support::app();
        let (materializer, transport, _) = materializer(CHAT_BODY);
        app.world_mut().insert_resource(materializer);
        app.world_mut()
            .spawn(binding(openai_on(&OPENAI, Some(route))));
        rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
        let effect = app
            .world_mut()
            .spawn(PendingEffect::new(KEY, completion()))
            .id();
        bus_support::tick_until(&mut app, "the routed client answers", |world| {
            world.get::<EffectOutcome>(effect).is_some()
        });
        let requests = transport.requests();
        assert_eq!(requests.len(), 1, "{named}");
        assert!(
            requests[0].uri.starts_with(BASE) && requests[0].uri.ends_with(path),
            "{named}: {}",
            requests[0].uri
        );
    }
}

#[test]
fn the_host_transport_is_built_once_per_binding_and_sends_to_the_base_url() {
    let mut app = bus_support::app();
    let (materializer, transport, built) = materializer(ANTHROPIC_BODY);
    app.world_mut().insert_resource(materializer);
    app.world_mut().spawn(binding(anthropic()));
    app.world_mut().spawn(
        ProviderBinding::new(
            "t/model:chat",
            ProviderConfig::OpenAi(
                OpenAI::with_key(&DEEPSEEK, Secret::default())
                    .with_base_url("http://other.invalid/v1"),
            ),
            "deepseek-x",
            "cassette",
        )
        .labelled("chat"),
    );
    assert_eq!(
        built.load(Ordering::SeqCst),
        0,
        "nothing built before materialize"
    );
    let report = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
    assert_eq!(report.materialized.len(), 2);
    assert_eq!(
        built.load(Ordering::SeqCst),
        2,
        "one client construction per binding"
    );
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(KEY, completion()))
        .id();
    bus_support::tick_until(&mut app, "the bound client answers", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    let outcome = app.world().get::<EffectOutcome>(effect).unwrap();
    assert_eq!(bus_support::text_of(&outcome.0), "hi");
    let requests = transport.requests();
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0].uri.starts_with(BASE),
        "the request went to the binding's base URL: {}",
        requests[0].uri
    );
    assert_eq!(
        requests[0]
            .headers
            .get("x-api-key")
            .map(|v| v.to_str().unwrap()),
        Some(SENTINEL),
        "the resolved credential reached the wire, and only the wire"
    );
    assert_eq!(built.load(Ordering::SeqCst), 2, "serving builds no client");
}

#[test]
fn materialization_is_all_or_nothing() {
    let mut app = bus_support::app();
    let (materializer, _, built) = materializer(ANTHROPIC_BODY);
    app.world_mut().insert_resource(materializer);
    let good = app.world_mut().spawn(binding(anthropic())).id();
    let bad = app
        .world_mut()
        .spawn(ProviderBinding::new(
            "t/model:other",
            openai_on(&OPENAI, Some(Route::Chat)),
            "m",
            "vault:missing",
        ))
        .id();
    let error = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap_err();
    assert_eq!(
        error,
        MaterializeError::MissingCredential {
            key: HandlerKey::from("t/model:other"),
            credential: CredentialRef::new("vault:missing"),
            detail: "not a cassette".to_owned(),
        }
    );
    assert!(
        app.world().get::<Bound>(good).is_none(),
        "the good binding was not registered"
    );
    assert!(app.world().get::<Bound>(bad).is_none());
    assert!(served(app.world_mut()) == 0);
    // The one client that was built before the refusal is dropped, unserved.
    assert!(built.load(Ordering::SeqCst) <= 1);
    // Fixing the reference materializes both.
    app.world_mut()
        .entity_mut(bad)
        .get_mut::<ProviderBinding>()
        .unwrap()
        .credential = CredentialRef::new("cassette");
    let report = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
    assert_eq!(report.materialized.len(), 2);
    assert_eq!(served(app.world_mut()), 2);
}

#[test]
fn an_existing_handler_wins_over_a_binding() {
    let mut app = bus_support::app();
    let counters = Arc::new(bus_support::Counters::default());
    let by_hand = bus_support::register(
        &mut app,
        KEY,
        bus_support::MockModel::saying(&counters, "by hand"),
    );
    app.world_mut().insert_resource(panicking_materializer());
    // A binding on its own entity for a key another entity serves.
    let standalone = app.world_mut().spawn(binding(anthropic())).id();
    let report = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
    assert_eq!(report.kept, vec![HandlerKey::from(KEY)]);
    assert!(report.materialized.is_empty());
    assert!(app.world().get::<Bound>(standalone).is_none());
    // A binding attached to the served entity itself (what a scene load
    // does when the host bound the key first).
    app.world_mut().entity_mut(standalone).despawn();
    app.world_mut()
        .entity_mut(by_hand)
        .insert(binding(anthropic()));
    let report = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
    assert_eq!(report.kept, vec![HandlerKey::from(KEY)]);
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(KEY, completion()))
        .id();
    bus_support::tick_until(&mut app, "the hand-registered handler answers", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(
        bus_support::text_of(&app.world().get::<EffectOutcome>(effect).unwrap().0),
        "by hand"
    );
}

#[test]
fn a_replayer_wins_and_no_transport_is_built() {
    // A live world records one answer through a materialized binding.
    let mut live = bus_support::app();
    EffectLogResource::install(live.world_mut(), EffectLogRecorder::new());
    let (materializer, _, _) = materializer(ANTHROPIC_BODY);
    live.world_mut().insert_resource(materializer);
    live.world_mut().spawn(binding(anthropic()));
    rig_ecs::bus::materialize_bindings(live.world_mut()).unwrap();
    let effect = live
        .world_mut()
        .spawn(PendingEffect::new(KEY, completion()))
        .id();
    bus_support::tick_until(&mut live, "live answer", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    let log = live.world().resource::<EffectLogResource>().log();

    // A replay world binds the log's replayer under the key first, then
    // carries the same binding: the replayer wins, and the materializer's
    // resolver and transport are never called.
    let mut replay = bus_support::app();
    Handlers::with(replay.world_mut(), |handlers| {
        Replay::default().register(handlers, &log)
    })
    .unwrap()
    .unwrap();
    replay.world_mut().flush();
    let replayer = replay
        .world_mut()
        .query::<(Entity, &Bound)>()
        .iter(replay.world())
        .find(|(_, bound)| bound.key == HandlerKey::from(KEY))
        .map(|(entity, _)| entity)
        .expect("the replayer is bound under the key");
    replay
        .world_mut()
        .entity_mut(replayer)
        .insert(binding(anthropic()));
    replay.world_mut().insert_resource(panicking_materializer());
    let report = rig_ecs::bus::materialize_bindings(replay.world_mut()).unwrap();
    assert_eq!(report.kept, vec![HandlerKey::from(KEY)]);
    let replayed = Replay::load(replay.world_mut(), &log)[0];
    bus_support::tick_until(&mut replay, "replayed answer", |world| {
        world.get::<EffectOutcome>(replayed).is_some()
    });
    assert_eq!(
        bus_support::text_of(&replay.world().get::<EffectOutcome>(replayed).unwrap().0),
        "hi"
    );
}

#[test]
fn refusals_are_deterministic_and_register_nothing() {
    // No materializer.
    let mut app = bus_support::app();
    app.world_mut().spawn(binding(anthropic()));
    assert_eq!(
        rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap_err(),
        MaterializeError::NoMaterializer
    );
    let (materializer, _, built) = materializer(CHAT_BODY);
    app.world_mut().insert_resource(materializer);
    // Duplicate key.
    let second = app
        .world_mut()
        .spawn(binding(openai_on(&DEEPSEEK, None)))
        .id();
    assert_eq!(
        rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap_err(),
        MaterializeError::DuplicateKey {
            key: HandlerKey::from(KEY)
        }
    );
    assert!(served(app.world_mut()) == 0);
    assert_eq!(
        built.load(Ordering::SeqCst),
        0,
        "a refused binding builds no transport"
    );
    app.world_mut().entity_mut(second).despawn();
    // Key mismatch: the binding sits beside a `Bound` of another key.
    let counters = Arc::new(bus_support::Counters::default());
    let other = bus_support::register(
        &mut app,
        "t/model:other",
        bus_support::MockModel::saying(&counters, "other"),
    );
    let standalone = app
        .world_mut()
        .query_filtered::<Entity, With<ProviderBinding>>()
        .single(app.world())
        .unwrap();
    app.world_mut().entity_mut(standalone).despawn();
    app.world_mut()
        .entity_mut(other)
        .insert(binding(anthropic()));
    assert_eq!(
        rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap_err(),
        MaterializeError::KeyMismatch {
            key: HandlerKey::from(KEY),
            bound: HandlerKey::from("t/model:other"),
        }
    );
    // A dispatch to a bound-but-unserved key is refused, never a panic.
    let mut fresh = bus_support::app();
    fresh.world_mut().spawn((
        binding(anthropic()),
        Bound {
            key: HandlerKey::from(KEY),
            descriptor: HandlerDescriptor {
                key: HandlerKey::from(KEY),
                family: rig_core::effect::FamilyDescriptor::Completion {
                    model: rig_core::completion::ModelRef::new("default"),
                    capabilities: rig_core::completion::ProviderCapabilities::default(),
                },
                layers: vec![],
            },
        },
    ));
    let effect = fresh
        .world_mut()
        .spawn(PendingEffect::new(KEY, completion()))
        .id();
    bus_support::tick_until(&mut fresh, "the unserved key refuses", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    let outcome = fresh.world().get::<EffectOutcome>(effect).unwrap();
    assert!(
        matches!(&outcome.0, Err(report) if report.kind == ErrorKind::HandlerUnavailable),
        "{outcome:?}"
    );
}

#[test]
fn secrets_never_leave_the_resolver() {
    let secret = Secret::from(SENTINEL);
    assert_eq!(format!("{secret:?}"), "[redacted]");
    assert_eq!(serde_json::to_string(&secret).unwrap(), "\"[redacted]\"");
    assert_eq!(secret.expose(), SENTINEL);
    let binding = binding(anthropic());
    let debug = format!("{binding:?}");
    assert!(debug.contains("cassette"), "{debug}");
    assert!(!debug.contains(SENTINEL));
    assert!(!serde_json::to_string(&binding).unwrap().contains(SENTINEL));
    // A config that *does* hold a credential redacts it too: the secret
    // lives in the config, so this is where it must not print.
    let carrying = ProviderConfig::Anthropic(Anthropic::new(SENTINEL));
    assert!(!format!("{carrying:?}").contains(SENTINEL));
    assert!(!serde_json::to_string(&carrying).unwrap().contains(SENTINEL));
    assert_eq!(carrying.credential().expose(), SENTINEL);
    // Diagnostics name the reference, never what it resolved to.
    let mut app = bus_support::app();
    let (materializer, _, _) = materializer(ANTHROPIC_BODY);
    app.world_mut().insert_resource(materializer);
    app.world_mut().spawn(ProviderBinding::new(
        KEY,
        anthropic(),
        "model-x",
        "vault:missing",
    ));
    let error = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap_err();
    let text = format!("{error} / {error:?}");
    assert!(text.contains("t/model:default"), "{text}");
    assert!(text.contains("vault:missing"), "{text}");
    assert!(!text.contains(SENTINEL), "{text}");
    let report = MaterializeReport {
        materialized: vec![HandlerKey::from(KEY)],
        kept: vec![],
    };
    assert!(!format!("{report:?}").contains(SENTINEL));
}

/// A bus-and-agent world: what a checkpoint saves from and loads into.
fn world_app() -> bevy_app::App {
    let mut app = bus_support::app();
    rig_ecs::systems::AgentPlugin::install(app.world_mut());
    app
}

/// The checkpoint entities carrying a `Bound`.
fn bound_entities(checkpoint: &Checkpoint) -> Vec<&rig_ecs::checkpoint::CheckpointEntity> {
    checkpoint
        .entities
        .iter()
        .filter(|entity| entity.contains_key(std::any::type_name::<Bound>()))
        .collect()
}

fn descriptor(label: &str) -> HandlerDescriptor {
    HandlerDescriptor {
        key: HandlerKey::from(KEY),
        family: rig_core::effect::FamilyDescriptor::Completion {
            model: rig_core::completion::ModelRef::new(label),
            capabilities: rig_core::completion::ProviderCapabilities::default(),
        },
        layers: vec![],
    }
}

/// The checkpoint of a world with the Anthropic binding materialized: the
/// binding beside the `Bound` the materializer gave it.
fn materialized_binding_checkpoint() -> Checkpoint {
    let mut head = world_app();
    let (head_materializer, _, _) = materializer(ANTHROPIC_BODY);
    head.world_mut().insert_resource(head_materializer);
    head.world_mut().spawn(binding(anthropic()));
    rig_ecs::bus::materialize_bindings(head.world_mut()).unwrap();
    save_world(head.world_mut()).unwrap()
}

#[test]
fn a_checkpoint_loads_its_bindings_as_data_and_materializes_on_the_hosts_word() {
    // The saving world: a materialized binding, and one nothing served yet.
    let mut head = world_app();
    let (head_materializer, _, _) = materializer(ANTHROPIC_BODY);
    head.world_mut().insert_resource(head_materializer);
    head.world_mut().spawn(binding(anthropic()));
    rig_ecs::bus::materialize_bindings(head.world_mut()).unwrap();
    head.world_mut()
        .spawn(ProviderBinding::new("t/model:later", gemini(), "g", "cassette").labelled("later"));
    let checkpoint = save_world(head.world_mut()).unwrap();
    let saved = head
        .world_mut()
        .query::<&Bound>()
        .single(head.world())
        .unwrap()
        .descriptor
        .clone();
    let json = checkpoint.to_json().unwrap();
    assert!(
        json.contains("cassette"),
        "the credential ref travels: {json}"
    );
    assert!(!json.contains(SENTINEL), "no secret in the checkpoint");

    // The loading world: a resolver and a transport that must not be
    // touched by the load.
    let checkpoint = Checkpoint::from_json(&json).unwrap();
    let mut app = world_app();
    app.world_mut().insert_resource(panicking_materializer());
    let loaded = load_world(&checkpoint, app.world_mut()).unwrap();
    assert_eq!(loaded.with::<ProviderBinding>(app.world()).len(), 2);
    let mut bound: Vec<(Entity, Bound, ProviderBinding)> = app
        .world_mut()
        .query::<(Entity, &Bound, &ProviderBinding)>()
        .iter(app.world())
        .map(|(entity, bound, binding)| (entity, bound.clone(), binding.clone()))
        .collect();
    assert_eq!(bound.len(), 1, "the materialized one is bound as it was");
    let (entity, restored, restored_binding) = bound.remove(0);
    assert_eq!(restored.key, HandlerKey::from(KEY));
    assert_eq!(restored.descriptor, saved);
    assert_eq!(restored_binding, binding(anthropic()));
    assert!(
        app.world().get::<Handler>(entity).is_none(),
        "bound, not served: the load built no client"
    );
    let unbound = app
        .world_mut()
        .query_filtered::<&ProviderBinding, Without<Bound>>()
        .single(app.world())
        .unwrap()
        .clone();
    assert_eq!(unbound.key, HandlerKey::from("t/model:later"));
    // A dispatch to the loaded key is refused until the host materializes.
    let refused = app
        .world_mut()
        .spawn(PendingEffect::new(KEY, completion()))
        .id();
    bus_support::tick_until(&mut app, "the loaded key refuses", |world| {
        world.get::<EffectOutcome>(refused).is_some()
    });
    assert!(matches!(
        &app.world().get::<EffectOutcome>(refused).unwrap().0,
        Err(report) if report.kind == ErrorKind::HandlerUnavailable
    ));
    // The host's word: the same key, the same descriptor, now served.
    let (host_materializer, transport, built) = materializer(ANTHROPIC_BODY);
    app.world_mut().insert_resource(host_materializer);
    let report = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
    assert_eq!(
        report.materialized,
        vec![HandlerKey::from(KEY), HandlerKey::from("t/model:later")]
    );
    assert_eq!(built.load(Ordering::SeqCst), 2);
    assert_eq!(app.world().get::<Bound>(entity).unwrap().descriptor, saved);
    assert!(
        app.world().get::<Handler>(entity).is_some(),
        "the binding's entity is the handler entity"
    );
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(KEY, completion()))
        .id();
    bus_support::tick_until(&mut app, "the materialized key answers", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(
        bus_support::text_of(&app.world().get::<EffectOutcome>(effect).unwrap().0),
        "hi"
    );
    assert_eq!(transport.requests().len(), 1);
    // Saved again, the bound binding is the same data.
    let again = save_world(app.world_mut()).unwrap();
    let first = bound_entities(&checkpoint);
    assert_eq!(first.len(), 1);
    assert!(
        bound_entities(&again).contains(&first[0]),
        "{}",
        again.to_json().unwrap()
    );
}

#[test]
fn a_checkpoint_load_validates_its_bindings() {
    let checkpoint = materialized_binding_checkpoint();
    let spawned = |app: &mut bevy_app::App| {
        app.world_mut()
            .query::<&ProviderBinding>()
            .iter(app.world())
            .count()
    };
    // The key served by a handler of another family.
    let mut app = world_app();
    let counters = Arc::new(bus_support::Counters::default());
    Handlers::with(app.world_mut(), |handlers| {
        handlers.register_open(
            KEY,
            rig_core::effect::FamilyDescriptor::Tool {
                name: "add".to_owned(),
                description: String::new(),
                parameters: serde_json::json!({}),
                embedding: None,
            },
        )
    })
    .unwrap()
    .unwrap();
    app.world_mut().flush();
    let before = app.world().entities().len();
    let error = load_world(&checkpoint, app.world_mut()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Request);
    assert!(error.message.contains("completion"), "{error}");
    assert_eq!(spawned(&mut app), 0, "refused before any spawn");
    assert_eq!(app.world().entities().len(), before);
    // The key served by a completion handler: the handler wins, the
    // binding rides on its entity, and a later materialization keeps it.
    let mut app = world_app();
    let by_hand = bus_support::register(
        &mut app,
        KEY,
        bus_support::MockModel::saying(&counters, "by hand"),
    );
    app.world_mut().insert_resource(panicking_materializer());
    let loaded = load_world(&checkpoint, app.world_mut()).unwrap();
    assert_eq!(loaded.with::<ProviderBinding>(app.world()), vec![by_hand]);
    assert_eq!(spawned(&mut app), 1);
    let report = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
    assert_eq!(report.kept, vec![HandlerKey::from(KEY)]);
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(KEY, completion()))
        .id();
    bus_support::tick_until(&mut app, "the hand-registered handler answers", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(
        bus_support::text_of(&app.world().get::<EffectOutcome>(effect).unwrap().0),
        "by hand"
    );
}

#[test]
fn a_loaded_binding_that_would_build_another_descriptor_is_refused() {
    // Bound under the label `default`; the binding then says `renamed`.
    let mut head = world_app();
    let (head_materializer, _, _) = materializer(ANTHROPIC_BODY);
    head.world_mut().insert_resource(head_materializer);
    let entity = head.world_mut().spawn(binding(anthropic())).id();
    rig_ecs::bus::materialize_bindings(head.world_mut()).unwrap();
    head.world_mut()
        .entity_mut(entity)
        .insert(binding(anthropic()).labelled("renamed"));
    let checkpoint = save_world(head.world_mut()).unwrap();
    let mut app = world_app();
    load_world(&checkpoint, app.world_mut()).unwrap();
    let (materializer, _, _) = materializer(ANTHROPIC_BODY);
    app.world_mut().insert_resource(materializer);
    match rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap_err() {
        MaterializeError::DescriptorDrift { key, saved, built } => {
            assert_eq!(key, HandlerKey::from(KEY));
            assert!(saved.contains("default"), "{saved}");
            assert!(built.contains("renamed"), "{built}");
        }
        other => panic!("{other:?}"),
    }
    assert!(served(app.world_mut()) == 0);
}

#[test]
fn the_materialize_system_reports_through_the_resource() {
    let mut app = bus_support::app();
    app.world_mut().spawn(binding(openai_on(&DEEPSEEK, None)));
    rig_ecs::bus::materialize(app.world_mut());
    assert_eq!(
        app.world().get_resource::<MaterializeFailed>(),
        Some(&MaterializeFailed(MaterializeError::NoMaterializer))
    );
    let (materializer, _, _) = materializer(CHAT_BODY);
    app.world_mut().insert_resource(materializer);
    rig_ecs::bus::materialize(app.world_mut());
    assert!(app.world().get_resource::<MaterializeFailed>().is_none());
    assert_eq!(served(app.world_mut()), 1);
}

#[test]
fn a_binding_beside_its_own_bound_keeps_the_key_served_elsewhere() {
    // The host registered the key first; the binding, with the `Bound` a
    // scene saved beside it, then arrived on its own entity (a reflected
    // scene, a hand-built entity). `Handlers::bind` would re-serve the
    // host's entity, not this one: the binding is kept, nothing replaced.
    let mut app = bus_support::app();
    let counters = Arc::new(bus_support::Counters::default());
    let by_hand = bus_support::register(
        &mut app,
        KEY,
        bus_support::MockModel::saying(&counters, "by hand"),
    );
    app.world_mut().flush();
    app.world_mut().insert_resource(panicking_materializer());
    let saved = Bound {
        key: HandlerKey::from(KEY),
        descriptor: descriptor("default"),
    };
    let loaded = app
        .world_mut()
        .spawn((binding(anthropic()), saved.clone()))
        .id();
    let report = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
    assert_eq!(
        report,
        MaterializeReport {
            materialized: vec![],
            kept: vec![HandlerKey::from(KEY)],
        }
    );
    {
        let world = app.world();
        assert!(
            world.get::<Handler>(by_hand).is_some(),
            "the host's handler stays"
        );
        assert!(
            world.get::<Handler>(loaded).is_none(),
            "the loaded entity is not served"
        );
    }
    assert_eq!(served(app.world_mut()), 1);
    let bound = app.world().get::<Bound>(loaded).unwrap();
    assert_eq!(
        bound.descriptor, saved.descriptor,
        "its own `Bound` untouched"
    );
    // (Two `Bound`s under one key is a world the registry never makes —
    // which of them a dispatch resolves to is not pinned here; what is
    // pinned is that materializing replaced nothing.)

    // A replayer under the key, registered before the bound binding
    // arrived: the same.
    let mut live = bus_support::app();
    EffectLogResource::install(live.world_mut(), EffectLogRecorder::new());
    let (materializer, _, _) = materializer(ANTHROPIC_BODY);
    live.world_mut().insert_resource(materializer);
    live.world_mut().spawn(binding(anthropic()));
    rig_ecs::bus::materialize_bindings(live.world_mut()).unwrap();
    let recorded = live
        .world_mut()
        .spawn(PendingEffect::new(KEY, completion()))
        .id();
    bus_support::tick_until(&mut live, "live answer", |world| {
        world.get::<EffectOutcome>(recorded).is_some()
    });
    let log = live.world().resource::<EffectLogResource>().log();
    let live_bound = live
        .world_mut()
        .query::<&Bound>()
        .single(live.world())
        .unwrap()
        .clone();
    let mut replay = bus_support::app();
    Handlers::with(replay.world_mut(), |handlers| {
        Replay::default().register(handlers, &log)
    })
    .unwrap()
    .unwrap();
    replay.world_mut().flush();
    let replayer = replay
        .world_mut()
        .query::<(Entity, &Bound)>()
        .iter(replay.world())
        .find(|(_, bound)| bound.key == HandlerKey::from(KEY))
        .map(|(entity, _)| entity)
        .expect("the replayer is bound under the key");
    let loaded = replay
        .world_mut()
        .spawn((binding(anthropic()), live_bound))
        .id();
    replay.world_mut().insert_resource(panicking_materializer());
    let report = rig_ecs::bus::materialize_bindings(replay.world_mut()).unwrap();
    assert_eq!(report.kept, vec![HandlerKey::from(KEY)]);
    assert!(report.materialized.is_empty());
    {
        let world = replay.world();
        assert!(
            world.get::<Handler>(replayer).is_some(),
            "the replayer stays"
        );
        assert!(world.get::<Handler>(loaded).is_none());
        assert_eq!(served(replay.world_mut()), 1);
    }

    // Another entity merely holds the key in a `Bound` nothing serves: kept
    // too, and neither `Bound` is touched.
    let mut fresh = bus_support::app();
    fresh.world_mut().insert_resource(panicking_materializer());
    let holder = fresh.world_mut().spawn(saved.clone()).id();
    let loaded = fresh
        .world_mut()
        .spawn((binding(anthropic()), saved.clone()))
        .id();
    let report = rig_ecs::bus::materialize_bindings(fresh.world_mut()).unwrap();
    assert_eq!(report.kept, vec![HandlerKey::from(KEY)]);
    assert!(served(fresh.world_mut()) == 0);
    assert_eq!(
        fresh.world().get::<Bound>(holder).unwrap().descriptor,
        saved.descriptor
    );
    assert_eq!(
        fresh.world().get::<Bound>(loaded).unwrap().descriptor,
        saved.descriptor
    );
}

#[test]
fn a_key_bound_to_another_family_is_kept_beside_a_good_binding() {
    // `t/model:a` is free and good; `t/tool:add` is bound on another
    // entity to a tool served by the world, and a binding for it arrived
    // beside a completion `Bound` of its own. `Handlers::bind` would
    // refuse the second registration (another family) after the first
    // went through; every registration is checked first, so the good one
    // materializes and the other is kept, with nothing half-registered.
    let mut app = bus_support::app();
    let (materializer, _, built) = materializer(ANTHROPIC_BODY);
    app.world_mut().insert_resource(materializer);
    let tool_key = HandlerKey::from("t/tool:add");
    let tool = Handlers::with(app.world_mut(), |handlers| {
        handlers.register_open(
            tool_key.clone(),
            rig_core::effect::FamilyDescriptor::Tool {
                name: "add".to_owned(),
                description: String::new(),
                parameters: serde_json::json!({}),
                embedding: None,
            },
        )
    })
    .unwrap()
    .unwrap();
    app.world_mut().flush();
    let good_key = HandlerKey::from("t/model:a");
    let good = app
        .world_mut()
        .spawn(
            ProviderBinding::new(good_key.clone(), anthropic(), "model-x", "cassette")
                .labelled("a"),
        )
        .id();
    let mut own = descriptor("default");
    own.key = tool_key.clone();
    let clashing = app
        .world_mut()
        .spawn((
            ProviderBinding::new(tool_key.clone(), anthropic(), "model-x", "cassette")
                .labelled("default"),
            Bound {
                key: tool_key.clone(),
                descriptor: own.clone(),
            },
        ))
        .id();
    let report = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
    assert_eq!(
        report,
        MaterializeReport {
            materialized: vec![good_key.clone()],
            kept: vec![tool_key.clone()],
        }
    );
    assert_eq!(
        built.load(Ordering::SeqCst),
        1,
        "one client, for the good one"
    );
    assert!(app.world().get::<Handler>(good).is_some());
    assert!(
        app.world().get::<Handler>(tool).is_some(),
        "the tool stays served"
    );
    assert!(
        app.world().get::<Handler>(clashing).is_none(),
        "the clashing binding is not served"
    );
    assert_eq!(served(app.world_mut()), 2);
    assert_eq!(app.world().get::<Bound>(good).unwrap().key, good_key);
    assert_eq!(
        app.world().get::<Bound>(clashing).unwrap().descriptor,
        own,
        "its own `Bound` untouched"
    );
    assert!(
        app.world().get::<Bound>(tool).unwrap().family() == rig_core::effect::EffectFamily::Tool,
        "the tool's `Bound` untouched"
    );
}
