//! Provider bindings as data (`bus::binding`): a `ProviderBinding` is
//! serde, holds a credential *reference* and never a secret, and becomes a
//! served handler only when the host materializes it through its own
//! resolver and transport.
//!
//! | claim | test |
//! |---|---|
//! | a registered reference writes as its canonical string, a configuration as an object; the credential reference is a name, saved as given; an unknown selection is refused by the reader | `a_binding_round_trips_verbatim` |
//! | materializing binds on the binding's own entity, with the descriptor a hand-registered adapter produces, for every configuration the old vocabulary could express | `a_materialized_binding_describes_itself_as_a_hand_registered_adapter` |
//! | the built client sends to the configuration's base URL through the host's transport, one construction per binding | `the_host_transport_is_built_once_per_binding_and_sends_to_the_base_url` |
//! | all or nothing: one refused credential registers nothing | `materialization_is_all_or_nothing` |
//! | precedence: a served key wins, the binding is reported kept | `an_existing_handler_wins_over_a_binding` |
//! | a replayer under the key wins: no transport, no credential | `a_replayer_wins_and_no_transport_is_built` |
//! | duplicate keys, key mismatch, no materializer are refused before any registration | `refusals_are_deterministic_and_register_nothing` |
//! | no secret in `Debug`, in diagnostics, in JSON | `secrets_never_leave_the_resolver` |
//! | a checkpoint saves the binding with its bound descriptor and no secret; loading spawns it bound and unserved, resolving nothing; materializing after the load serves it under the saved descriptor; saved again it is the same data | `a_checkpoint_loads_its_bindings_as_data_and_materializes_on_the_hosts_word` |
//! | a load is refused, before any spawn, for a served key of another family; a served key of the same family keeps the handler and attaches the binding | `a_checkpoint_load_validates_its_bindings` |
//! | a binding whose built descriptor differs from the saved one is refused (`DescriptorDrift`), nothing served | `a_loaded_binding_that_would_build_another_descriptor_is_refused` |
//! | the system: a refusal is left in `MaterializeFailed`, a success clears it | `the_materialize_system_reports_through_the_resource` |
//! | every former binding option — host, `anthropic-version`, betas, instruction placement — is expressible as data and reaches the wire | `every_former_binding_option_reaches_the_wire` |
//! | a gateway the old provider enum could not name materializes | `a_gateway_the_old_vocabulary_could_not_name_materializes` |
//! | diagnostics report what materialization sees: a binding's own stale `Bound` is not somebody else serving its key | `diagnostics_agree_with_materialization` |
//! | a binding beside its own unserved `Bound` whose key another entity serves (a host registration made before it, a replayer) or merely holds is kept: the served handler untouched, its own `Bound` untouched, nothing built | `a_binding_beside_its_own_bound_keeps_the_key_served_elsewhere` |
//! | a binding whose registration `bind` would refuse (its key bound to another family elsewhere) beside a good one: checked before the first registration — the good one materializes, the other is kept, nothing half-registered | `a_key_bound_to_another_family_is_kept_beside_a_good_binding` |

use crate::bus_support;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use bevy_ecs::prelude::*;
use rig_cassette::ecs::EffectLogResource;
use rig_cassette::ecs::Replay;
use rig_cassette::effect_log::EffectLogRecorder;
use rig_core::{
    driver::Bind,
    effect::{HandlerDescriptor, HandlerKey},
    error::ErrorKind,
    http_client::BoxedHttpClient,
    providers::{
        anthropic::wire::Anthropic,
        gemini::Gemini,
        openai::wire::{DEEPSEEK, OpenAI, Route},
        registry::{ProviderConfig, ProviderId},
    },
    serve::{ErasedHandler, adapters::CompletionAdapter},
    test_utils::RecordingHttpClient,
};
use rig_ecs::{
    bus::{
        Bound, CredentialRef, EffectOutcome, Handler, Handlers, MaterializeError,
        MaterializeFailed, MaterializeReport, Materializer, PendingEffect, ProviderBinding, Secret,
        provider_diagnostics,
    },
    checkpoint::{Checkpoint, load_world, save_world},
};

const KEY: &str = "t/model:default";
const BASE: &str = "http://cassette.invalid/v1";
const MODEL: &str = "model-x";
const SENTINEL: &str = "sentinel-secret-value-3f9a";

const ANTHROPIC_BODY: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#;
const RESPONSES_BODY: &str = r#"{"id":"resp_1","object":"response","created_at":0,"status":"completed","model":"m","output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"hi","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}"#;
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

/// The provider configurations the retired `ProviderKind` enum could
/// express, each at the test's base URL. A binding carries one of these as
/// data; the credential is empty because the host's resolver supplies it.
fn anthropic() -> ProviderConfig {
    ProviderConfig::Anthropic(Anthropic::new("").with_base_url(BASE))
}

fn openai_chat() -> ProviderConfig {
    ProviderConfig::OpenAi(OpenAI::new("").with_base_url(BASE).with_route(Route::Chat))
}

fn openai_responses() -> ProviderConfig {
    ProviderConfig::OpenAi(
        OpenAI::new("")
            .with_base_url(BASE)
            .with_route(Route::Responses),
    )
}

fn gemini() -> ProviderConfig {
    ProviderConfig::Gemini(Gemini::new("").with_base_url(BASE))
}

fn deepseek() -> ProviderConfig {
    ProviderConfig::OpenAi(OpenAI::with_key(&DEEPSEEK, "").with_base_url(BASE))
}

/// One configuration, beside the hand-written construction whose descriptor
/// a binding built from it must equal. `by_hand` names the endpoint
/// explicitly — a `Route` in data must reach the same wire a caller reaches
/// by asking for it.
struct Shape {
    name: &'static str,
    config: fn() -> ProviderConfig,
    by_hand: fn(BoxedHttpClient) -> ErasedHandler,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "anthropic/anthropic",
        config: anthropic,
        by_hand: |http| {
            ErasedHandler::new(CompletionAdapter::new(
                "default",
                Anthropic::new(SENTINEL)
                    .with_base_url(BASE)
                    .bind(http)
                    .completion(MODEL),
            ))
        },
    },
    Shape {
        name: "openai/openai on Chat Completions",
        config: openai_chat,
        by_hand: |http| {
            ErasedHandler::new(CompletionAdapter::new(
                "default",
                OpenAI::new(SENTINEL)
                    .with_base_url(BASE)
                    .bind(http)
                    .chat(MODEL),
            ))
        },
    },
    Shape {
        name: "openai/openai on Responses",
        config: openai_responses,
        by_hand: |http| {
            ErasedHandler::new(CompletionAdapter::new(
                "default",
                OpenAI::new(SENTINEL)
                    .with_base_url(BASE)
                    .bind(http)
                    .responses(MODEL),
            ))
        },
    },
    Shape {
        name: "gcp.gemini/gemini",
        config: gemini,
        by_hand: |http| {
            ErasedHandler::new(CompletionAdapter::new(
                "default",
                Gemini::new(SENTINEL)
                    .with_base_url(BASE)
                    .bind(http)
                    .completion(MODEL),
            ))
        },
    },
    Shape {
        name: "deepseek/openai",
        config: deepseek,
        by_hand: |http| {
            ErasedHandler::new(CompletionAdapter::new(
                "default",
                OpenAI::with_key(&DEEPSEEK, SENTINEL)
                    .with_base_url(BASE)
                    .bind(http)
                    .completion(MODEL),
            ))
        },
    },
];

/// A binding of [`KEY`] to `config`, labelled `default`.
fn binding(config: ProviderConfig) -> ProviderBinding {
    ProviderBinding::configured(KEY, config, MODEL, "cassette").labelled("default")
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
    // A registered selection writes as its canonical string. Shorthand in,
    // qualified out: the stored form never depends on how many endpoints a
    // vendor happens to have.
    let registered = ProviderBinding::parse(KEY, "deepseek:deepseek-chat", "cassette")
        .expect("shorthand parses")
        .labelled("default");
    let json = serde_json::to_string(&registered).unwrap();
    assert!(json.contains(r#""credential":"cassette""#), "{json}");
    assert!(
        json.contains(r#""provider":"deepseek/openai:deepseek-chat""#),
        "{json}"
    );
    let again: ProviderBinding = serde_json::from_str(&json).unwrap();
    assert_eq!(again, registered);
    assert_eq!(again.credential, CredentialRef::new("cassette"));

    // A configuration writes as an object and keeps everything it names.
    let configured = binding(ProviderConfig::Anthropic(
        Anthropic::new("").with_base_url(BASE).with_beta("b-1"),
    ));
    let json = serde_json::to_string(&configured).unwrap();
    assert!(json.contains(r#""betas":["b-1"]"#), "{json}");
    assert!(json.contains("cassette.invalid"), "{json}");
    assert_eq!(
        serde_json::from_str::<ProviderBinding>(&json).unwrap(),
        configured,
        "the configuration round-trips: nothing but the credential is lost"
    );

    // An unknown selection is refused by the reader, before anything is
    // spawned — there is no materialization error left for it to become.
    let unknown = json_with_provider(r#""mistral/anthropic:m""#);
    let error = serde_json::from_str::<ProviderBinding>(&unknown).unwrap_err();
    assert!(error.to_string().contains("mistral"), "{error}");
    let ambiguous = json_with_provider(r#""zai:glm-4.6""#);
    let error = serde_json::from_str::<ProviderBinding>(&ambiguous).unwrap_err();
    assert!(error.to_string().contains("zai/anthropic"), "{error}");
}

/// A binding JSON whose `provider` field is `provider`: how a stored
/// binding names a selection this build must refuse.
fn json_with_provider(provider: &str) -> String {
    format!(r#"{{"key":"{KEY}","provider":{provider},"label":"default","credential":"cassette"}}"#)
}

#[test]
fn a_materialized_binding_describes_itself_as_a_hand_registered_adapter() {
    for shape in SHAPES {
        let kind = shape.name;
        let mut app = bus_support::app();
        let (materializer, transport, _) = materializer(ANTHROPIC_BODY);
        app.world_mut().insert_resource(materializer);
        let entity = app.world_mut().spawn(binding((shape.config)())).id();
        assert!(app.world().get::<Bound>(entity).is_none());
        let report = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
        assert_eq!(
            report,
            MaterializeReport {
                materialized: vec![HandlerKey::from(KEY)],
                kept: vec![],
            },
            "{kind}"
        );
        // The binding's own entity is the handler entity.
        let bound = app
            .world()
            .get::<Bound>(entity)
            .unwrap_or_else(|| panic!("{kind}: bound"));
        assert_eq!(bound.key, HandlerKey::from(KEY));
        assert!(app.world().get::<Handler>(entity).is_some());
        // The descriptor is the one `CompletionAdapter::new("default", model)`
        // registered by hand under the key would produce.
        let by_hand: ErasedHandler = (shape.by_hand)(BoxedHttpClient::new(transport.clone()));
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
        assert_eq!(bound.descriptor, expected, "{kind}: the bound descriptor");
        assert_eq!(
            serde_json::to_value(&bound.descriptor).unwrap(),
            serde_json::to_value(&expected).unwrap(),
            "{kind}: the descriptor's JSON (what the policy hash folds)"
        );
        // Materializing again finds the key served and builds nothing.
        let again = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
        assert_eq!(again.kept, vec![HandlerKey::from(KEY)], "{kind}");
        assert!(again.materialized.is_empty(), "{kind}");
    }
}

#[test]
fn the_host_transport_is_built_once_per_binding_and_sends_to_the_base_url() {
    let mut app = bus_support::app();
    let (materializer, transport, built) = materializer(ANTHROPIC_BODY);
    app.world_mut().insert_resource(materializer);
    app.world_mut().spawn(binding(anthropic()));
    app.world_mut().spawn(
        ProviderBinding::configured(
            "t/model:chat",
            ProviderConfig::OpenAi(
                OpenAI::with_key(&DEEPSEEK, "").with_base_url("http://other.invalid/v1"),
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
        .spawn(ProviderBinding::configured(
            "t/model:other",
            openai_chat(),
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
    Replay::default()
        .register(replay.world_mut(), &log)
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
    let second = app.world_mut().spawn(binding(deepseek())).id();
    assert_eq!(
        rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap_err(),
        MaterializeError::DuplicateKey {
            key: HandlerKey::from(KEY)
        }
    );
    assert!(served(app.world_mut()) == 0);
    app.world_mut().entity_mut(second).despawn();
    assert_eq!(
        built.load(Ordering::SeqCst),
        0,
        "a refused binding builds no transport"
    );
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
    // Diagnostics name the reference, never what it resolved to.
    let mut app = bus_support::app();
    let (materializer, _, _) = materializer(ANTHROPIC_BODY);
    app.world_mut().insert_resource(materializer);
    app.world_mut().spawn(
        ProviderBinding::configured(KEY, anthropic(), MODEL, "vault:missing").labelled("default"),
    );
    let error = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap_err();
    let text = format!("{error} / {error:?}");
    assert!(text.contains("t/model:default"), "{text}");
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
    head.world_mut().spawn(
        ProviderBinding::configured("t/model:later", gemini(), "g", "cassette").labelled("later"),
    );
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
    app.world_mut().spawn(binding(deepseek()));
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
    Replay::default()
        .register(replay.world_mut(), &log)
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
            ProviderBinding::configured(good_key.clone(), anthropic(), MODEL, "cassette")
                .labelled("a"),
        )
        .id();
    let mut own = descriptor("default");
    own.key = tool_key.clone();
    let clashing = app
        .world_mut()
        .spawn((
            ProviderBinding::configured(tool_key.clone(), anthropic(), MODEL, "cassette")
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

/// Every option the retired `extra_params` bag carried has a typed home, and
/// each reaches the bytes: the `anthropic-version` and `anthropic-beta`
/// headers, and the Responses instruction placement in the request body.
///
/// The binding is built from JSON, so this is the whole path a host takes:
/// stored data, deserialization, materialization, one dispatch, the request
/// the provider actually received.
#[test]
fn every_former_binding_option_reaches_the_wire() {
    // `anthropic_version` / `anthropic_betas` -> the Messages configuration.
    let stored = format!(
        r#"{{"key":"{KEY}","provider":{{"config":{{"anthropic":{{"api_key":"[redacted]","base_url":"{BASE}","version":"2023-01-01","betas":["beta-one","beta-two"],"dialect":"anthropic"}}}},"model":"{MODEL}"}},"label":"default","credential":"cassette"}}"#
    );
    let binding: ProviderBinding = serde_json::from_str(&stored).expect("the options read back");
    let mut app = bus_support::app();
    let (materializer, transport, _) = materializer(ANTHROPIC_BODY);
    app.world_mut().insert_resource(materializer);
    app.world_mut().spawn(binding);
    rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(KEY, completion()))
        .id();
    bus_support::tick_until(&mut app, "the Messages wire answers", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    let requests = transport.requests();
    let header = |name: &str| {
        requests[0]
            .headers
            .get(name)
            .map(|value| value.to_str().unwrap().to_owned())
    };
    assert_eq!(header("anthropic-version").as_deref(), Some("2023-01-01"));
    assert_eq!(
        header("anthropic-beta").as_deref(),
        Some("beta-one,beta-two"),
        "both betas, in order"
    );

    // `system_instructions_as_messages` -> the OpenAI configuration's
    // placement, which decides where the preamble lands in the body.
    // Absent: the dialect's default — top-level `instructions`.
    let default = responses_body_with_placement(None);

    assert!(
        default.get("instructions").is_some(),
        "the preamble is top-level by default: {default}"
    );
    // Overridden: a `system` message in `input`, and no top-level field.
    let overridden = responses_body_with_placement(Some("input_system_messages"));
    assert!(
        overridden.get("instructions").is_none(),
        "no top-level instructions: {overridden}"
    );
    assert!(
        overridden["input"]
            .as_array()
            .expect("an input array")
            .iter()
            .any(|item| item.get("role").and_then(|role| role.as_str()) == Some("system")),
        "the preamble travels as a system message instead: {overridden}"
    );
}

/// The Responses request body a binding sends when its stored configuration
/// names `placement` (or names none): built from JSON, materialized,
/// dispatched once, and read off the transport.
fn responses_body_with_placement(placement: Option<&str>) -> serde_json::Value {
    let option = placement
        .map(|value| format!(r#","system_instructions":"{value}""#))
        .unwrap_or_default();
    let stored = format!(
        r#"{{"key":"{KEY}","provider":{{"config":{{"openai":{{"api_key":"[redacted]","base_url":"{BASE}","dialect":"openai","auth":"Bearer","route":"Responses"{option}}}}},"model":"{MODEL}"}},"label":"default","credential":"cassette"}}"#
    );
    let binding: ProviderBinding = serde_json::from_str(&stored).expect("the placement reads back");
    let mut app = bus_support::app();
    let (resources, transport, _) = materializer(RESPONSES_BODY);
    app.world_mut().insert_resource(resources);
    app.world_mut().spawn(binding);
    rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
    // The placement decides where Rig's system instructions go, so the
    // probe carries a leading system message — which is what a preamble is
    // by the time a request reaches a wire.
    let mut request = bus_support::request();
    request.max_tokens = Some(64);
    request.chat_history.insert(
        0,
        rig_core::completion::Message::system("be brief".to_owned()),
    );
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(
            KEY,
            rig_core::effect::EffectKind::Completion {
                request,
                stream: false,
            },
        ))
        .id();
    bus_support::tick_until(&mut app, "the Responses wire answers", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    let requests = transport.requests();
    assert_eq!(requests.len(), 1);
    serde_json::from_slice::<serde_json::Value>(&requests[0].body).expect("a JSON body")
}

/// A gateway the retired provider enum had no variant for — Venice — is
/// expressible as data and materializes, because the vocabulary is the
/// provider's own configuration rather than an ECS enum.
#[test]
fn a_gateway_the_old_vocabulary_could_not_name_materializes() {
    assert!(
        ProviderId::resolve("venice/openai").is_ok(),
        "the registry has it"
    );
    let stored = format!(
        r#"{{"key":"{KEY}","provider":{{"config":{{"openai":{{"api_key":"[redacted]","base_url":"{BASE}","dialect":"venice","auth":"Bearer"}}}},"model":"venice-uncensored"}},"label":"default","credential":"cassette"}}"#
    );
    let binding: ProviderBinding = serde_json::from_str(&stored).expect("Venice reads from data");
    assert_eq!(binding.provider.id().to_string(), "venice/openai");
    let mut app = bus_support::app();
    let (materializer, transport, _) = materializer(CHAT_BODY);
    app.world_mut().insert_resource(materializer);
    app.world_mut().spawn(binding);
    let report = rig_ecs::bus::materialize_bindings(app.world_mut()).unwrap();
    assert_eq!(report.materialized, vec![HandlerKey::from(KEY)]);
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(KEY, completion()))
        .id();
    bus_support::tick_until(&mut app, "the Venice wire answers", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    let requests = transport.requests();
    assert!(requests[0].uri.starts_with(BASE), "{}", requests[0].uri);
    assert_eq!(
        requests[0]
            .headers
            .get("authorization")
            .map(|value| value.to_str().unwrap().to_owned()),
        Some(format!("Bearer {SENTINEL}")),
        "Venice takes a bearer token"
    );
}

/// Diagnostics report the state materialization acts on, and nothing else:
/// a binding's own stale `Bound` — what a checkpoint load leaves — is not
/// another entity serving its key, and a real second server is.
#[test]
fn diagnostics_agree_with_materialization() {
    // A checkpoint-shaped world: the binding sits beside the `Bound` it was
    // saved with, and nothing serves the key.
    let checkpoint = materialized_binding_checkpoint();
    let mut app = world_app();
    app.world_mut().insert_resource(panicking_materializer());
    load_world(&checkpoint, app.world_mut()).unwrap();
    let report = provider_diagnostics(app.world());
    assert_eq!(report.bindings.len(), 1);
    let stale = &report.bindings[0];
    assert!(
        !stale.served,
        "a checkpoint-loaded binding is bound but not served"
    );
    assert!(
        !stale.kept(),
        "and so the next materialization will build it, not keep it"
    );
    assert!(
        !stale.served_elsewhere,
        "its own stale `Bound` is not somebody else serving it"
    );
    assert_eq!(stale.key, HandlerKey::from(KEY));
    assert_eq!(stale.model, MODEL);
    assert_eq!(stale.label, "default");
    assert_eq!(stale.selection, "anthropic/anthropic");
    assert_eq!(stale.credential, CredentialRef::new("cassette"));
    assert_eq!(stale.guidance.env, "ANTHROPIC_API_KEY");
    assert!(stale.guidance.required);
    assert_eq!(
        app.world()
            .get::<ProviderBinding>(stale.entity)
            .map(|b| &b.key),
        Some(&stale.key),
        "the entity is the binding's own"
    );
    assert!(report.refusal.is_none());
    // The registry it reports is this build's, and every entry resolves.
    assert_eq!(report.registered.len(), ProviderId::all().count());
    for provider in &report.registered {
        assert!(
            ProviderId::resolve(&provider.selection).is_ok(),
            "{provider:?}"
        );
    }
    assert!(
        report.registered.iter().any(
            |provider| provider.selection == "llamacpp/openai" && !provider.credential.required
        ),
        "an optional-auth selection is a hint, not a requirement"
    );
    assert!(
        !format!("{report:?}").contains(SENTINEL),
        "no secret in the report"
    );
    // Collecting the report resolved no credential (the installed resolver
    // panics if asked), served nothing and changed nothing: reading it twice
    // gives the same answer.
    assert_eq!(served(app.world_mut()), 0);
    assert_eq!(provider_diagnostics(app.world()), report);

    // A different entity serving the same key: `served_elsewhere`, and the
    // same binding materialization reports `kept`.
    let mut second = bus_support::app();
    let counters = Arc::new(bus_support::Counters::default());
    bus_support::register(
        &mut second,
        KEY,
        bus_support::MockModel::saying(&counters, "by hand"),
    );
    second.world_mut().flush();
    let binding_entity = second.world_mut().spawn(binding(anthropic())).id();
    let report = provider_diagnostics(second.world());
    assert_eq!(report.bindings.len(), 1);
    assert_eq!(report.bindings[0].entity, binding_entity);
    assert!(
        report.bindings[0].served_elsewhere,
        "another entity serves the key"
    );
    assert!(
        !report.bindings[0].served,
        "the binding's own entity serves nothing"
    );
    assert!(report.bindings[0].kept());
    second.world_mut().insert_resource(panicking_materializer());
    let outcome = rig_ecs::bus::materialize_bindings(second.world_mut()).unwrap();
    assert_eq!(
        outcome.kept,
        vec![HandlerKey::from(KEY)],
        "materialization agrees: the existing handler wins"
    );

    // The third shape, and the one the two halves of the predicate exist to
    // tell apart: a binding already materialized on its own entity. It is
    // `served`, not `served_elsewhere`, and materialization keeps it — where
    // the checkpoint-loaded binding above, identical but for the `Handler`,
    // is built.
    let mut own = bus_support::app();
    let (materializer, _, _) = materializer(ANTHROPIC_BODY);
    own.world_mut().insert_resource(materializer);
    let entity = own.world_mut().spawn(binding(anthropic())).id();
    rig_ecs::bus::materialize_bindings(own.world_mut()).unwrap();
    let report = provider_diagnostics(own.world());
    assert_eq!(report.bindings.len(), 1);
    let live = &report.bindings[0];
    assert_eq!(live.entity, entity);
    assert!(live.served, "its own entity is serving");
    assert!(!live.served_elsewhere, "nobody else is");
    assert!(live.kept());
    own.world_mut().insert_resource(panicking_materializer());
    assert_eq!(
        rig_ecs::bus::materialize_bindings(own.world_mut())
            .unwrap()
            .kept,
        vec![HandlerKey::from(KEY)],
        "materialization agrees with `kept()`"
    );
}
