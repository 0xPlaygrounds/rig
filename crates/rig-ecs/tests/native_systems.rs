//! Same-pass ownership, publication and run-isolation regressions.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use bevy_ecs::{prelude::*, schedule::LogLevel, system::RunSystemOnce};
use rig_core::{
    completion::{CompletionRequestBuilder, CompletionResponse, ModelRef, ProviderCapabilities},
    effect::{EffectKind, FamilyDescriptor, Outcome, RetrievedDocuments},
    message::AssistantContent,
    serve::ServingPolicy,
};
use rig_ecs::{
    agent::{
        Attachment, AwaitingModel, Completion, DocumentId, DocumentText, Failed, Failure,
        InvalidCall, InvalidCalls, Order, OutputKind, Outputs, Owner, Resolution, Retrieval,
        RetrievalKind, Retrieving, Role, RunOf, RunResult, Settled, ToolCallSlot, Turn, Unhandled,
        UsesModel, Utterance,
    },
    bus::{Bus, EffectOutcome, Handlers, Held, PendingEffect, RigSchedule},
    systems::{
        Folded, Fresh, Materialised, RigSet, RunBundle, RunCommands, RunConfig, attach_retrieved,
        fold, install_agent,
    },
};

fn world() -> World {
    let mut world = World::new();
    Bus::with_policy(ServingPolicy::default())
        .ambiguity_detection(LogLevel::Error)
        .install(&mut world);
    install_agent(&mut world);
    world
}

fn completion(content: Vec<AssistantContent>) -> (PendingEffect, EffectOutcome) {
    (
        PendingEffect::new(
            "model",
            EffectKind::Completion {
                request: CompletionRequestBuilder::unbound("prompt").build(),
                stream: false,
            },
        ),
        EffectOutcome(Ok(Outcome::Completion(CompletionResponse::new(
            content,
            rig_core::completion::Usage::new(),
            "model",
        )))),
    )
}

#[test]
fn retrieved_ids_are_reserved_across_results_and_turns_in_one_pass() {
    let mut world = world();
    let agent = world.spawn_empty().id();
    let mut turns = Vec::new();
    for _ in 0..2 {
        let run = world.spawn(RunOf(agent)).id();
        let turn = world.spawn((Turn, Fresh, Retrieving, ChildOf(run))).id();
        turns.push(turn);
        for _ in 0..2 {
            let request = rig_core::vector_store::request::VectorSearchRequest::builder()
                .query("query")
                .samples(2)
                .build()
                .map_filter(rig_core::vector_store::request::Filter::interpret);
            world.spawn((
                PendingEffect::new(
                    "index",
                    EffectKind::Retrieve {
                        query: rig_core::effect::RetrieveQuery::TopN { req: request },
                    },
                ),
                Retrieval {
                    samples: 2,
                    what: RetrievalKind::Documents,
                },
                EffectOutcome(Ok(Outcome::Documents(RetrievedDocuments::Scored(vec![
                    (1.0, "same".into(), serde_json::json!({"text":"first"})),
                    (0.5, "same".into(), serde_json::json!({"text":"later"})),
                ])))),
                ChildOf(turn),
            ));
        }
    }
    world.run_system_once(attach_retrieved).unwrap();
    let docs: Vec<_> = world
        .query::<(Entity, &DocumentId, &DocumentText)>()
        .iter(&world)
        .collect();
    assert_eq!(docs.len(), 1);
    let (document, _, text) = docs.first().unwrap();
    assert_eq!(
        text.0,
        serde_json::to_string_pretty(&serde_json::json!({"text":"first"})).unwrap()
    );
    for turn in turns {
        let attachments: Vec<_> = world
            .get::<Children>(turn)
            .unwrap()
            .iter()
            .filter_map(|child| world.get::<Attachment>(child))
            .collect();
        assert_eq!(attachments.len(), 4);
        assert!(
            attachments
                .iter()
                .all(|attachment| attachment.0 == *document)
        );
        assert!(world.get::<Retrieving>(turn).is_none());
    }
}

#[test]
fn committed_duplicate_document_id_reuses_first_query_match_without_replacing_text() {
    let mut world = world();
    world.spawn((DocumentId("same".into()), DocumentText("first".into())));
    world.spawn((
        DocumentId("same".into()),
        DocumentText("second".into()),
        Owner("extra archetype".into()),
    ));
    let expected = world
        .query::<(Entity, &DocumentId)>()
        .iter(&world)
        .next()
        .unwrap()
        .0;
    let agent = world.spawn_empty().id();
    let run = world.spawn(RunOf(agent)).id();
    let turn = world.spawn((Turn, Fresh, Retrieving, ChildOf(run))).id();
    let request = rig_core::vector_store::request::VectorSearchRequest::builder()
        .query("query")
        .samples(1)
        .build()
        .map_filter(rig_core::vector_store::request::Filter::interpret);
    world.spawn((
        PendingEffect::new(
            "index",
            EffectKind::Retrieve {
                query: rig_core::effect::RetrieveQuery::TopN { req: request },
            },
        ),
        Retrieval {
            samples: 1,
            what: RetrievalKind::Documents,
        },
        EffectOutcome(Ok(Outcome::Documents(RetrievedDocuments::Scored(vec![(
            1.0,
            "same".into(),
            serde_json::json!("replacement"),
        )])))),
        ChildOf(turn),
    ));
    world.run_system_once(attach_retrieved).unwrap();
    let attachment = world.query::<&Attachment>().single(&world).unwrap();
    assert_eq!(attachment.0, expected);
    assert_ne!(
        world.get::<DocumentText>(expected).unwrap().0,
        "\"replacement\""
    );
}

#[test]
fn tool_children_cannot_complete_or_replace_the_models_outputs() {
    let mut world = world();
    let turn = world.spawn((Turn, Outputs::default())).id();
    world.spawn((
        PendingEffect::new(
            "tool",
            EffectKind::ToolCall {
                name: "tool".into(),
                args: "{}".into(),
            },
        ),
        ToolCallSlot {
            index: 0,
            id: rig_core::message::ToolCallId::new("call").unwrap(),
            provider: None,
            name: "tool".into(),
        },
        EffectOutcome(Err(rig_core::serve::cancelled())),
        ChildOf(turn),
    ));
    let (effect, outcome) = completion(vec![AssistantContent::text("model answer")]);
    let model = world.spawn((effect, Completion, ChildOf(turn))).id();
    world.run_system_once(fold).unwrap();
    assert!(!world.get::<Outputs>(turn).unwrap().done);
    world.entity_mut(model).insert(outcome);
    world.run_system_once(fold).unwrap();
    let outputs = world.get::<Outputs>(turn).unwrap();
    assert!(outputs.done);
    assert_eq!(
        outputs.content,
        vec![AssistantContent::text("model answer")]
    );
}

#[derive(Resource, Default)]
struct Reads(Vec<Entity>);
#[derive(Resource, Default)]
struct Checkpoints(Vec<(bool, bool)>);

#[test]
fn invalid_wait_never_publishes_read_and_ignore_materialises_once_in_same_pass() {
    let mut world = world();
    world.init_resource::<Reads>();
    world.init_resource::<Checkpoints>();
    world.add_observer(|added: On<Add, Materialised>, mut reads: ResMut<Reads>| {
        reads.0.push(added.event().entity);
    });
    let agent = world
        .spawn((
            Owner("invalid".into()),
            InvalidCalls {
                retries: 0,
                unhandled: Unhandled::Ignore,
            },
        ))
        .id();
    let bundle = RunBundle::new(&mut world, agent, false);
    let run = world.spawn((bundle, AwaitingModel)).id();
    let turn = world
        .spawn((
            Turn,
            Outputs::default(),
            Folded(OutputKind::Auto),
            Order(0),
            ChildOf(run),
        ))
        .id();
    let (effect, outcome) = completion(vec![
        AssistantContent::text("kept"),
        AssistantContent::tool_call("bad", "unknown", serde_json::json!({})),
    ]);
    world.spawn((effect, outcome, Completion, ChildOf(turn)));
    world
        .resource_mut::<Schedules>()
        .get_mut(RigSchedule)
        .unwrap()
        .add_systems(
            (move |turns: Query<Has<Materialised>, With<Turn>>,
                   runs: Query<Has<Settled>, With<RunOf>>,
                   mut checkpoints: ResMut<Checkpoints>| {
                checkpoints
                    .0
                    .push((turns.get(turn).unwrap(), runs.get(run).unwrap()));
            })
            .in_set(RigSet::Checkpoint),
        );
    world.run_schedule(RigSchedule);
    assert!(world.resource::<Reads>().0.is_empty());
    assert!(world.get::<AwaitingModel>(run).is_some());
    assert_eq!(world.query::<&InvalidCall>().iter(&world).count(), 1);
    assert_eq!(world.resource::<Checkpoints>().0, vec![(false, false)]);
    world.run_schedule(RigSchedule);
    assert_eq!(world.resource::<Reads>().0, vec![turn]);
    assert_eq!(world.get::<RunResult>(run).unwrap().0, "kept");
    assert!(world.get::<AwaitingModel>(run).is_none());
    assert!(world.get::<Settled>(run).is_some());
    assert_eq!(world.query::<&Resolution>().iter(&world).count(), 0);
    assert_eq!(
        world.resource::<Checkpoints>().0,
        vec![(false, false), (true, true)]
    );
    world.run_schedule(RigSchedule);
    assert_eq!(world.resource::<Reads>().0, vec![turn]);
}

#[test]
fn malformed_earlier_run_does_not_delay_later_valid_assembly() {
    let mut world = world();
    let model = Handlers::with(&mut world, |handlers| {
        handlers.register_open(
            "model",
            FamilyDescriptor::Completion {
                model: ModelRef::new("model"),
                capabilities: ProviderCapabilities::default(),
            },
        )
    })
    .unwrap()
    .unwrap();
    // Keep folded requests available for host inspection instead of dispatching.
    world.add_observer(|added: On<Add, PendingEffect>, mut commands: Commands| {
        commands.entity(added.event().entity).insert(Held);
    });
    let agent = world.spawn((Owner("owner".into()), UsesModel(model))).id();
    let bad = world.spawn_run(agent, "bad", RunConfig::default());
    let good = world.spawn_run(agent, "good", RunConfig::default());
    let utterance = world
        .get::<Children>(bad)
        .unwrap()
        .iter()
        .find(|entity| world.get::<Utterance>(*entity).is_some())
        .unwrap();
    world.entity_mut(utterance).remove::<Role>();
    world.run_schedule(RigSchedule);
    assert!(matches!(
        world.get::<Failed>(bad),
        Some(Failed(Failure::Content(_)))
    ));
    assert!(world.get::<AwaitingModel>(good).is_some());
    let effects: Vec<_> = world
        .query::<(&PendingEffect, &ChildOf)>()
        .iter(&world)
        .collect();
    assert_eq!(effects.len(), 1);
    let (effect, parent) = effects.first().unwrap();
    assert_eq!(
        world.get::<ChildOf>(parent.parent()).unwrap().parent(),
        good
    );
    let EffectKind::Completion { request, .. } = &effect.kind else {
        panic!("expected completion");
    };
    assert_eq!(
        request.chat_history,
        vec![rig_core::message::Message::user("good")]
    );
}

#[test]
fn malformed_materialisation_keeps_later_run_and_partial_publication_independent() {
    let mut world = world();
    let agent = world.spawn(Owner("materialise".into())).id();
    let mut runs = Vec::new();
    let mut turns = Vec::new();
    for content in [
        vec![
            AssistantContent::text("unpublishable"),
            AssistantContent::Image(rig_core::message::Image {
                data: rig_core::message::DocumentSourceKind::Base64("invalid!".into()),
                ..Default::default()
            }),
        ],
        vec![AssistantContent::text("valid")],
    ] {
        let bundle = RunBundle::new(&mut world, agent, false);
        let run = world.spawn((bundle, AwaitingModel)).id();
        let turn = world
            .spawn((
                Turn,
                Outputs::default(),
                Folded(OutputKind::Auto),
                Order(0),
                ChildOf(run),
            ))
            .id();
        let (effect, outcome) = completion(content);
        world.spawn((effect, outcome, Completion, ChildOf(turn)));
        runs.push(run);
        turns.push(turn);
    }
    world.run_schedule(RigSchedule);
    let bad = *runs.first().unwrap();
    let good = *runs.last().unwrap();
    assert!(matches!(
        world.get::<Failed>(bad),
        Some(Failed(Failure::Content(_)))
    ));
    assert!(world.get::<Materialised>(*turns.first().unwrap()).is_some());
    assert!(
        !world
            .get::<Children>(bad)
            .unwrap()
            .iter()
            .any(|entity| world.get::<Utterance>(entity).is_some())
    );
    assert_eq!(world.get::<RunResult>(good).unwrap().0, "valid");
    assert!(world.get::<Settled>(good).is_some());
}
