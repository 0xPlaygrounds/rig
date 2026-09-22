//! The `reflect` feature: every component of the bus and the graph
//! round-trips through reflection — reflected off its entity, serialized
//! by `ReflectSerializer` (an `Entity` as itself here), deserialized by
//! `ReflectDeserializer`, made concrete by `FromReflect`, serialized again
//! to the same JSON — and every registered component occurs in the world
//! the test builds, so the round trip is of the whole vocabulary.
//!
//! | claim | test |
//! |---|---|
//! | every registered component reflects off an entity and round-trips by value | `every_component_round_trips_through_reflection` |
//! | the remote wrappers serialize as the wire form: an effect's kind reflected is its serde JSON | `a_reflected_effect_is_its_wire_form` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod run_support;

use bevy_ecs::{prelude::*, reflect::AppTypeRegistry, reflect::ReflectComponent};
use bevy_reflect::{
    ReflectFromReflect,
    serde::{ReflectDeserializer, ReflectSerializer},
};
use rig_core::{
    completion::message::{Message, ToolCallId, ToolChoice},
    effect::{EffectId, HandlerKey},
    error::{ErrorKind, ErrorReport},
};
use rig_ecs::{
    agent::{
        Cancelled, Conversation, Failed, Failure, InvalidCall, MemoryAppendScheduled, MessageParts,
        Remembered, Remembering, Reprompt, RequestPatch, Resolution, Retrievable, Retrieval,
        RetrievalKind, Retrieves, Retrieving, Retry, Route, StreamRequested, ToolChoiceSpec,
        ToolContextSpec, ToolPolicy, Utterance,
    },
    bus::{EffectOutcome, Held, IdCounter, PendingEffect, Reserved, Streamed},
    systems::RunCommands,
};
use run_support::*;
use serde::de::DeserializeSeed;

const MODEL: &str = "t/model:default";
const ADD: &str = "t/tool:add#0";

/// A world holding every component: a run with a tool turn and an answer,
/// plus one entity carrying the components that run did not produce.
fn populated() -> bevy_app::App {
    let mut app = app();
    rig_ecs::reflect::install_reflect(app.world_mut());
    app.world_mut().resource_mut::<IdCounter>().0 = 1;
    let (model, _) = Scripted::new(
        MODEL,
        vec![
            vec![call("c1", "add", serde_json::json!({"x": 1, "y": 2}))],
            vec![AssistantContent::text("3")],
        ],
    );
    let model = register(&mut app, MODEL, model);
    let add = register(&mut app, ADD, Adder::new(ADD));
    let agent = spawn_agent(app.world_mut(), "t", model);
    let world = app.world_mut();
    world.entity_mut(agent).insert((
        rig_ecs::agent::MaxTurns(2),
        rig_ecs::agent::OutputToolConfig::default(),
        ToolChoiceSpec(Some(ToolChoice::Auto)),
        ToolContextSpec(rig_core::tool::ToolContext::new()),
        ToolPolicy { concurrency: 2 },
        rig_ecs::agent::content::parts::ToolResultLimit::new(4096),
        rig_ecs::agent::AdditionalParams(Some(serde_json::json!({"k": 1}))),
        rig_ecs::agent::DocumentProps(std::collections::HashMap::from([(
            "a".to_owned(),
            "b".to_owned(),
        )])),
    ));
    let document = world
        .spawn((
            rig_ecs::agent::DocumentId("d".to_owned()),
            rig_ecs::agent::DocumentText("text".to_owned()),
        ))
        .id();
    world.spawn((rig_ecs::agent::Context(document), ChildOf(agent)));
    world.spawn((rig_ecs::agent::Grant(add), ChildOf(agent)));
    let run = world.spawn_run(agent, &[], "add one and two", false, None);
    rig_ecs::agent::checkpoint::hold_after_tool_turn(world, run, "reflection", 99)
        .expect("a future hold preserves normal settlement");
    tick_until(&mut app, "the run", |world| {
        world.get::<rig_ecs::agent::Settled>(run).is_some()
    });
    // What that run did not produce, on one entity.
    app.world_mut().spawn(rig_ecs::agent::Outputs {
        usage_recorded: true,
        stream_validated: 5,
        ..Default::default()
    });
    app.world_mut().spawn((
        Cancelled("why".to_owned()),
        Retry {
            feedback: Some("again".to_owned()),
        },
        RequestPatch {
            preamble: Some("p".to_owned()),
            ..Default::default()
        },
        Reprompt(Message::user("again")),
        InvalidCall {
            id: ToolCallId::new("i").unwrap(),
            name: "nope".to_owned(),
            arguments: serde_json::json!({"z": true}),
            prefix: vec![
                AssistantContent::text("delivered prefix"),
                call("i", "nope", serde_json::json!({"z": true})),
            ],
            stream_offset: Some(3),
        },
        Resolution::Skip {
            reason: "no".to_owned(),
        },
        Failed(Failure::MaxTurns { limit: 1 }),
    ));
    let held = app
        .world_mut()
        .spawn((
            Held,
            Reserved(EffectId::from_raw(77)),
            Conversation("c".to_owned()),
            Remembered,
            Remembering,
            (
                rig_ecs::systems::BatchHeld,
                rig_ecs::agent::RunPhase::LoadingMemory,
            ),
            (
                MemoryAppendScheduled,
                rig_ecs::agent::PolicyVersion("reflect-test/v1".into()),
                rig_ecs::bus::ToolOutputs(rig_core::tool::ToolContext::new()),
            ),
            Retrievable,
            Retrieving,
            Retrieval {
                samples: 2,
                what: RetrievalKind::Tools,
            },
            Retrieves(model),
            Route(model),
            rig_ecs::agent::Remembers(model),
            rig_ecs::agent::Attachment(document),
            StreamRequested(true),
        ))
        .id();
    rig_ecs::bus::acquire_hold(
        app.world_mut(),
        held,
        rig_core::observe::Emitter::named("policy/reflect"),
    )
    .expect("a named owner");
    let utterance = app
        .world_mut()
        .spawn((
            Utterance,
            rig_ecs::agent::Role::User,
            Streamed {
                errors: vec![(0, ErrorReport::new(ErrorKind::Cancelled, "stopped"))],
                events: Vec::new(),
                text: "so far".to_owned(),
                outcome: Some(Err(ErrorReport::new(ErrorKind::Cancelled, "stopped"))),
            },
            rig_ecs::agent::OutputToolName(Some("out".to_owned())),
            rig_ecs::agent::Retry { feedback: None },
            rig_ecs::agent::ToolCallSlot {
                index: 0,
                id: ToolCallId::new("c9").expect("an id"),
                provider: None,
                name: "add".to_owned(),
            },
            rig_ecs::systems::Fresh,
            rig_ecs::systems::Folded(rig_ecs::agent::OutputKind::Auto),
            rig_ecs::agent::Prompt(vec![rig_core::message::UserContent::text("p")]),
            (
                rig_ecs::agent::RunPhase::Assembling,
                rig_ecs::agent::ProviderRetries(2),
                rig_ecs::agent::ProviderRetrying,
            ),
            rig_ecs::agent::Batch { calls: 2 },
            rig_ecs::bus::InFlight {
                key: HandlerKey::from(ADD),
            },
        ))
        .id();
    rig_ecs::agent::content::parts::write_message(
        app.world_mut(),
        utterance,
        MessageParts::User {
            content: vec![rig_core::message::UserContent::text("u")],
        },
    )
    .expect("valid reflected content");

    use rig_core::message::{
        AssistantContent, Audio, Document, Image, Reasoning, UserContent, Video,
    };
    for parts in [
        MessageParts::User {
            content: vec![
                UserContent::Image(Image::default()),
                UserContent::Audio(Audio::default()),
                UserContent::Video(Video::default()),
                UserContent::Document(Document::default()),
            ],
        },
        MessageParts::Assistant {
            id: Some("message".into()),
            content: vec![AssistantContent::Reasoning(Reasoning::new("thought"))],
        },
    ] {
        let entity = app.world_mut().spawn(Utterance).id();
        rig_ecs::agent::content::parts::write_message(app.world_mut(), entity, parts)
            .expect("valid reflected variants");
    }
    use rig_ecs::agent::content::parts::{EditTarget, RequestPartEdit};
    let target = app
        .world()
        .get::<Children>(utterance)
        .unwrap()
        .iter()
        .next()
        .unwrap();
    app.world_mut()
        .spawn((EditTarget(target), RequestPartEdit::Text("patched".into())));
    app
}

#[test]
fn content_variants_are_reflected_enum_data() {
    use bevy_reflect::{PartialReflect, ReflectRef};
    use rig_ecs::agent::content::parts::ContentPart;
    use std::collections::BTreeSet;

    let mut app = populated();
    let world = app.world_mut();
    let variants: BTreeSet<_> = world
        .query::<&ContentPart>()
        .iter(world)
        .map(|part| {
            let ReflectRef::Enum(value) = part.reflect_ref() else {
                panic!("content must be reflected enum data");
            };
            assert!(value.field_len() > 0);
            value.variant_name().to_owned()
        })
        .collect();
    assert_eq!(
        variants,
        BTreeSet::from(
            [
                "Text",
                "Image",
                "Audio",
                "Video",
                "Document",
                "ToolCall",
                "ToolResult",
                "Reasoning",
                "Json",
            ]
            .map(str::to_owned)
        )
    );
}

#[test]
fn every_component_round_trips_through_reflection() {
    let mut app = populated();
    let world = app.world_mut();
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    // The graph's entities: not the resource entities (`Time` and the
    // like are the app's, registered by their own plugins).
    let entities: Vec<Entity> = world
        .query_filtered::<Entity, Without<bevy_ecs::resource::IsResource>>()
        .iter(world)
        .collect();
    let mut seen: Vec<&str> = Vec::new();
    let mut relationships: Vec<&str> = Vec::new();
    let mut checked = 0usize;
    // The crate's components: its own and the hierarchy's, never a
    // resource (a resource is a component on its resource entity in Bevy).
    let ours = |registration: &bevy_reflect::TypeRegistration| {
        let path = registration.type_info().type_path();
        (path.starts_with("rig_ecs::") || path.starts_with("bevy_ecs::hierarchy::"))
            && registration
                .data::<bevy_ecs::reflect::ReflectResource>()
                .is_none()
    };
    for (registration, component) in registry.iter_with_data::<ReflectComponent>() {
        let path = registration.type_info().type_path();
        if !ours(registration) {
            continue;
        }
        for entity in &entities {
            let Some(value) = component.reflect(world.entity(*entity)) else {
                continue;
            };
            if !seen.contains(&path) {
                seen.push(path);
            }
            let json = match serde_json::to_string(&ReflectSerializer::new(
                value.as_partial_reflect(),
                &registry,
            )) {
                Ok(json) => json,
                // A relationship holds an `Entity`, which has no serde form
                // of its own: the scene serializes it as an index
                // (`tests/reflect_scene.rs`); here it is named and skipped.
                Err(error) if error.to_string().contains("bevy_ecs::entity::Entity") => {
                    if !relationships.contains(&path) {
                        relationships.push(path);
                    }
                    continue;
                }
                Err(error) => panic!("{path} serializes: {error}"),
            };
            let mut deserializer = serde_json::Deserializer::from_str(&json);
            let dynamic = ReflectDeserializer::new(&registry)
                .deserialize(&mut deserializer)
                .unwrap_or_else(|error| panic!("{path} deserializes: {error}\n{json}"));
            let concrete = registration
                .data::<ReflectFromReflect>()
                .unwrap_or_else(|| panic!("{path} registers FromReflect"))
                .from_reflect(dynamic.as_ref())
                .unwrap_or_else(|| panic!("{path} is concrete again"));
            let again = serde_json::to_string(&ReflectSerializer::new(
                concrete.as_partial_reflect(),
                &registry,
            ))
            .unwrap();
            assert_eq!(json, again, "{path} round-trips by value");
            checked += 1;
        }
    }
    // Every registered component occurs: the vocabulary is covered.
    let missing: Vec<&str> = registry
        .iter_with_data::<ReflectComponent>()
        .filter(|(registration, _)| ours(registration))
        .map(|(registration, _)| registration.type_info().type_path())
        .filter(|path| !seen.contains(path))
        .collect();
    assert!(missing.is_empty(), "components never seen: {missing:?}");
    assert!(checked > 100, "{checked} values checked");
    relationships.sort_unstable();
    assert_eq!(
        relationships,
        [
            "bevy_ecs::hierarchy::ChildOf",
            "bevy_ecs::hierarchy::Children",
            "rig_ecs::agent::Advert",
            "rig_ecs::agent::AdvertisedOn",
            "rig_ecs::agent::AttachedTo",
            "rig_ecs::agent::Attachment",
            "rig_ecs::agent::Context",
            "rig_ecs::agent::ContextOf",
            "rig_ecs::agent::Grant",
            "rig_ecs::agent::Grants",
            "rig_ecs::agent::ModelOf",
            "rig_ecs::agent::RememberedBy",
            "rig_ecs::agent::Remembers",
            "rig_ecs::agent::RetrievedBy",
            "rig_ecs::agent::Retrieves",
            "rig_ecs::agent::Route",
            "rig_ecs::agent::RoutedTo",
            "rig_ecs::agent::RunOf",
            "rig_ecs::agent::Runs",
            "rig_ecs::agent::UsesModel",
            "rig_ecs::agent::checkpoint::AssistantForTurns",
            "rig_ecs::agent::checkpoint::ResultsForTurns",
            "rig_ecs::agent::checkpoint::TurnAssistant",
            "rig_ecs::agent::checkpoint::TurnResults",
            "rig_ecs::agent::content::parts::EditTarget",
            "rig_ecs::agent::content::parts::EditedBy",
            "rig_ecs::bus::handlers::ServedBy",
            "rig_ecs::bus::handlers::Serves",
        ],
        "exactly the relationships hold an Entity"
    );
}

#[test]
fn a_reflected_effect_is_its_wire_form() {
    let mut app = populated();
    let world = app.world_mut();
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let mut effects = world.query::<(&PendingEffect, &EffectOutcome)>();
    let (effect, outcome) = effects
        .iter(world)
        .find(|(effect, _)| effect.key == HandlerKey::from(ADD))
        .expect("the tool call");
    let reflected: serde_json::Value =
        serde_json::to_value(ReflectSerializer::new(effect, &registry)).unwrap();
    let wire = serde_json::json!({
        "rig_ecs::bus::effect::PendingEffect": {
            "key": serde_json::to_value(&effect.key).unwrap(),
            "kind": serde_json::to_value(&effect.kind).unwrap(),
        }
    });
    assert_eq!(reflected, wire);
    let reflected: serde_json::Value =
        serde_json::to_value(ReflectSerializer::new(outcome, &registry)).unwrap();
    assert_eq!(
        reflected["rig_ecs::bus::effect::EffectOutcome"],
        serde_json::to_value(&outcome.0).unwrap()
    );
}
