//! Cached utterance views (CONTRACT §1, §13): the request is what a fresh
//! render gives, and only what changed is rendered again.
//!
//! | claim | test |
//! |---|---|
//! | over a multi-turn tool run every request equals a fresh uncached render, and each turn renders only the utterances the last turn added | `every_request_equals_a_fresh_render_and_only_new_utterances_are_rendered` |
//! | an unchanged history is served from the views: no render | `an_unchanged_history_is_served_from_the_views` |
//! | a text part change re-renders its utterance | `a_text_part_change_re_renders_its_utterance` |
//! | a change to an item nested under a tool result re-renders the results utterance | `a_nested_tool_result_item_change_re_renders_the_results` |
//! | a part spawned under an utterance re-renders it | `a_part_added_re_renders_its_utterance` |
//! | a part despawned re-renders its utterance | `a_part_despawned_re_renders_its_utterance` |
//! | a sibling `Order` change re-renders | `an_order_change_re_renders` |
//! | reparenting a part re-renders both utterances | `reparenting_a_part_re_renders_both_utterances` |
//! | a `MessageId` change re-renders | `a_message_id_change_re_renders` |
//! | `write_message` re-renders | `write_message_re_renders` |
//! | a component removed from a live part is seen: the run fails as the uncached path fails | `a_component_removed_from_a_live_entity_is_seen` |
//! | a `RequestPartEdit` is rendered with the edit, uncached; the view survives for the next turn | `a_request_part_edit_is_rendered_uncached_and_the_view_survives` |
//! | a `ToolResultLimit` applies over the view without a render | `a_limit_applies_over_the_view` |
//! | an asset collection drops every view | `an_asset_collection_drops_the_views` |
//! | a streamed and a unary run over the same history assemble the same request | `streaming_and_unary_assemble_the_same_request` |
//! | a scene saved mid-run and loaded elsewhere has no views, assembles the requests the first world would, and holds views after | `a_loaded_scene_assembles_identical_requests_and_rebuilds_its_views` |

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::type_complexity
)]

use crate::run_support;

use bevy_ecs::prelude::*;
use rig_core::{
    completion::{CompletionRequest, ModelRef, ProviderCapabilities},
    effect::{EffectKind, FamilyDescriptor},
    message::{AssistantContent, Message, ToolResultContent, UserContent},
    serve::ServingPolicy,
};
use rig_ecs::{
    agent::{
        Failed, Failure, Grant, MaxTurns, MessageParts, Order, Owner, Run, Settled, Turn,
        UsesModel, Utterance,
        checkpoint::{ToolTurnCommit, hold_after_tool_turn, release_tool_turn_hold},
        content::{
            cache::{AssemblyStats, CachedMessage},
            parts::{
                ContentError, ContentGraph, ContentPart, EditTarget, JsonPart, MessageId,
                RequestPartEdit, TOOL_RESULT_LIMIT_MARKER, TextPart, ToolResultLimit,
                ToolResultPart, collect_binary_assets, read_message, write_message,
            },
        },
        scene::{WorldScene, load_world, save_world},
    },
    bus::{Bus, Handlers, PendingEffect, RigSchedule},
    systems::{Fresh, RigSet, RunCommands, install_agent},
};
use run_support::*;

const MODEL: &str = "t/model:default";
const ADD: &str = "t/tool:add#0";

// ---------------------------------------------------------------------------
// The world fixture: one open model, one agent, a run over a prior history
// (an assistant call, its result of one text and one JSON item), a prompt.

fn history() -> Vec<MessageParts> {
    vec![
        MessageParts::Assistant {
            id: None,
            content: vec![AssistantContent::tool_call(
                "c1",
                "probe",
                serde_json::json!({"q": 1}),
            )],
        },
        MessageParts::User {
            content: vec![UserContent::ToolResult(rig_core::message::ToolResult {
                call: rig_core::message::ToolCallId::new("c1").unwrap(),
                provider: None,
                name: "probe".to_owned(),
                content: vec![
                    ToolResultContent::text("the result"),
                    ToolResultContent::Json {
                        value: serde_json::json!({"n": 1}),
                    },
                ],
            })],
        },
    ]
}

struct Fixture {
    world: World,
    agent: Entity,
    run: Entity,
    next_turn: u64,
}

impl Fixture {
    fn new() -> Self {
        let mut world = World::new();
        Bus::with_policy(ServingPolicy::default()).install(&mut world);
        install_agent(&mut world);
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
        let agent = world.spawn((Owner("owner".into()), UsesModel(model))).id();
        let run = world.spawn_run(agent, &history(), "next", false, None);
        Self {
            world,
            agent,
            run,
            next_turn: 100,
        }
    }

    /// Spawn a fresh turn and run the schedule once: one request folded.
    /// Returns the request and the stats delta of the pass.
    fn turn(&mut self) -> (CompletionRequest, AssemblyStats) {
        let before = stats(&self.world);
        self.world
            .spawn((Turn, Fresh, Order(self.next_turn), ChildOf(self.run)));
        self.next_turn += 1;
        self.world.run_schedule(RigSchedule);
        let after = stats(&self.world);
        let requests = requests(&mut self.world, self.run);
        (
            requests.last().expect("a request").clone(),
            AssemblyStats {
                assemblies: after.assemblies - before.assemblies,
                renders: after.renders - before.renders,
                hits: after.hits - before.hits,
                evictions: after.evictions - before.evictions,
            },
        )
    }

    /// A turn whose request must equal a fresh render, with `renders`
    /// utterances rendered and the rest served from their views.
    fn turn_expecting(&mut self, renders: u64) -> CompletionRequest {
        let fresh = graph_messages(&mut self.world, self.run);
        let (request, delta) = self.turn();
        assert_eq!(request.chat_history, fresh, "the request is a fresh render");
        assert_eq!(delta.assemblies, 1);
        assert_eq!(delta.renders, renders, "renders: {delta:?}");
        assert_eq!(
            delta.hits,
            fresh.len() as u64 - renders,
            "the rest are hits: {delta:?}"
        );
        request
    }

    /// The run's utterances in order: assistant, results, prompt.
    fn utterances(&mut self) -> Vec<Entity> {
        utterances_of(&mut self.world, self.run)
    }

    fn part<C: Component>(&self, utterance: Entity) -> Entity {
        parts_of::<C>(&self.world, utterance)
            .into_iter()
            .next()
            .expect("the part")
    }
}

fn stats(world: &World) -> AssemblyStats {
    *world.resource::<AssemblyStats>()
}

fn utterances_of(world: &mut World, run: Entity) -> Vec<Entity> {
    let mut utterances: Vec<_> = world
        .query_filtered::<(Entity, &ChildOf, &Order), With<Utterance>>()
        .iter(world)
        .filter(|(_, parent, _)| parent.parent() == run)
        .map(|(entity, _, order)| (order.0, entity))
        .collect();
    utterances.sort();
    utterances.into_iter().map(|(_, entity)| entity).collect()
}

/// The direct children of `parent` holding `C`, in sibling order.
fn parts_of<C: Component>(world: &World, parent: Entity) -> Vec<Entity> {
    let mut found: Vec<(u64, Entity)> = world
        .get::<Children>(parent)
        .into_iter()
        .flat_map(|children| children.iter())
        .filter(|child| world.get::<C>(*child).is_some())
        .map(|child| (world.get::<Order>(child).unwrap().0, child))
        .collect();
    found.sort();
    found.into_iter().map(|(_, entity)| entity).collect()
}

/// The fresh, uncached render of `run`'s history.
fn graph_messages(world: &mut World, run: Entity) -> Vec<Message> {
    utterances_of(world, run)
        .into_iter()
        .map(|entity| read_message(world, entity).unwrap().to_message())
        .collect()
}

/// The completion requests folded under `run`, by turn order.
fn requests(world: &mut World, run: Entity) -> Vec<CompletionRequest> {
    let mut found: Vec<_> = world
        .query::<(&PendingEffect, &ChildOf)>()
        .iter(world)
        .filter_map(|(effect, parent)| match &effect.kind {
            EffectKind::Completion { request, .. } => {
                let turn = parent.parent();
                (world.get::<ChildOf>(turn)?.parent() == run)
                    .then(|| (world.get::<Order>(turn).map(|o| o.0), request.clone()))
            }
            _ => None,
        })
        .collect();
    found.sort_by_key(|(order, _)| *order);
    found.into_iter().map(|(_, request)| request).collect()
}

fn views_of(world: &mut World, run: Entity) -> Vec<bool> {
    utterances_of(world, run)
        .into_iter()
        .map(|entity| world.get::<CachedMessage>(entity).is_some())
        .collect()
}

// ---------------------------------------------------------------------------
// The invalidation matrix, one trigger per test.

#[test]
fn an_unchanged_history_is_served_from_the_views() {
    let mut f = Fixture::new();
    assert_eq!(views_of(&mut f.world, f.run), [false, false, false]);
    let first = f.turn_expecting(3);
    assert_eq!(views_of(&mut f.world, f.run), [true, true, true]);
    let second = f.turn_expecting(0);
    assert_eq!(second.chat_history, first.chat_history);
    let third = f.turn_expecting(0);
    assert_eq!(third.chat_history, first.chat_history);
}

#[test]
fn a_text_part_change_re_renders_its_utterance() {
    let mut f = Fixture::new();
    let first = f.turn_expecting(3);
    let prompt = f.utterances()[2];
    let text = f.part::<TextPart>(prompt);
    f.world.get_mut::<TextPart>(text).unwrap().0.text = "changed".to_owned();
    let second = f.turn_expecting(1);
    assert_ne!(second.chat_history, first.chat_history);
    assert_eq!(
        second.chat_history[2],
        Message::User {
            content: vec![UserContent::text("changed")]
        }
    );
    f.turn_expecting(0);
}

#[test]
fn a_nested_tool_result_item_change_re_renders_the_results() {
    let mut f = Fixture::new();
    f.turn_expecting(3);
    let results = f.utterances()[1];
    let result = f.part::<ToolResultPart>(results);
    let json = parts_of::<JsonPart>(&f.world, result)[0];
    f.world.get_mut::<JsonPart>(json).unwrap().0 = serde_json::json!({"n": 2});
    let second = f.turn_expecting(1);
    let Message::User { content } = &second.chat_history[1] else {
        panic!("the results");
    };
    let UserContent::ToolResult(result) = &content[0] else {
        panic!("the result");
    };
    assert_eq!(
        result.content[1],
        ToolResultContent::Json {
            value: serde_json::json!({"n": 2})
        }
    );
}

#[test]
fn a_part_added_re_renders_its_utterance() {
    let mut f = Fixture::new();
    f.turn_expecting(3);
    let prompt = f.utterances()[2];
    f.world.spawn((
        ContentPart,
        Order(1),
        TextPart(rig_core::message::Text::from("more")),
        ChildOf(prompt),
    ));
    let second = f.turn_expecting(1);
    assert_eq!(
        second.chat_history[2],
        Message::User {
            content: vec![UserContent::text("next"), UserContent::text("more")]
        }
    );
}

#[test]
fn a_part_despawned_re_renders_its_utterance() {
    let mut f = Fixture::new();
    f.turn_expecting(3);
    let results = f.utterances()[1];
    let result = f.part::<ToolResultPart>(results);
    let json = parts_of::<JsonPart>(&f.world, result)[0];
    f.world.despawn(json);
    let second = f.turn_expecting(1);
    let Message::User { content } = &second.chat_history[1] else {
        panic!("the results");
    };
    let UserContent::ToolResult(result) = &content[0] else {
        panic!("the result");
    };
    assert_eq!(result.content.len(), 1, "the text item alone");
}

#[test]
fn an_order_change_re_renders() {
    let mut f = Fixture::new();
    f.turn_expecting(3);
    let results = f.utterances()[1];
    let result = f.part::<ToolResultPart>(results);
    let text = parts_of::<TextPart>(&f.world, result)[0];
    let json = parts_of::<JsonPart>(&f.world, result)[0];
    f.world.get_mut::<Order>(text).unwrap().0 = 1;
    f.world.get_mut::<Order>(json).unwrap().0 = 0;
    let second = f.turn_expecting(1);
    let Message::User { content } = &second.chat_history[1] else {
        panic!("the results");
    };
    let UserContent::ToolResult(result) = &content[0] else {
        panic!("the result");
    };
    assert!(matches!(result.content[0], ToolResultContent::Json { .. }));
    assert!(matches!(result.content[1], ToolResultContent::Text(_)));
}

#[test]
fn reparenting_a_part_re_renders_both_utterances() {
    let mut f = Fixture::new();
    f.turn_expecting(3);
    let utterances = f.utterances();
    let (results, prompt) = (utterances[1], utterances[2]);
    let result = f.part::<ToolResultPart>(results);
    let text = parts_of::<TextPart>(&f.world, result)[0];
    // The result's text item becomes the prompt's second part.
    f.world.entity_mut(text).insert((ChildOf(prompt), Order(1)));
    let second = f.turn_expecting(2);
    assert_eq!(
        second.chat_history[2],
        Message::User {
            content: vec![UserContent::text("next"), UserContent::text("the result")]
        }
    );
    let Message::User { content } = &second.chat_history[1] else {
        panic!("the results");
    };
    let UserContent::ToolResult(result) = &content[0] else {
        panic!("the result");
    };
    assert_eq!(result.content.len(), 1, "the JSON item alone");
}

#[test]
fn a_message_id_change_re_renders() {
    let mut f = Fixture::new();
    f.turn_expecting(3);
    let assistant = f.utterances()[0];
    f.world
        .entity_mut(assistant)
        .insert(MessageId(Some("m1".to_owned())));
    let second = f.turn_expecting(1);
    let Message::Assistant { id, .. } = &second.chat_history[0] else {
        panic!("the assistant");
    };
    assert_eq!(id.as_deref(), Some("m1"));
}

#[test]
fn write_message_re_renders() {
    let mut f = Fixture::new();
    f.turn_expecting(3);
    let prompt = f.utterances()[2];
    write_message(
        &mut f.world,
        prompt,
        MessageParts::User {
            content: vec![UserContent::text("rewritten")],
        },
    )
    .unwrap();
    let second = f.turn_expecting(1);
    assert_eq!(
        second.chat_history[2],
        Message::User {
            content: vec![UserContent::text("rewritten")]
        }
    );
}

#[test]
fn a_component_removed_from_a_live_entity_is_seen() {
    let mut f = Fixture::new();
    f.turn_expecting(3);
    let assistant = f.utterances()[0];
    // An assistant without its `MessageId` does not render: the run fails
    // as it would without a view.
    f.world.entity_mut(assistant).remove::<MessageId>();
    assert!(read_message(&f.world, assistant).is_err());
    let before = requests(&mut f.world, f.run).len();
    f.world.spawn((Turn, Fresh, Order(200), ChildOf(f.run)));
    f.world.run_schedule(RigSchedule);
    assert_eq!(requests(&mut f.world, f.run).len(), before, "no request");
    assert!(matches!(
        f.world.get::<Failed>(f.run),
        Some(Failed(Failure::Content(ContentError::Missing)))
    ));
}

#[test]
fn a_request_part_edit_is_rendered_uncached_and_the_view_survives() {
    let mut f = Fixture::new();
    let first = f.turn_expecting(3);
    let prompt = f.utterances()[2];
    let text = f.part::<TextPart>(prompt);
    let turn = f
        .world
        .spawn((Turn, Fresh, Order(f.next_turn), ChildOf(f.run)))
        .id();
    f.next_turn += 1;
    f.world.spawn((
        RequestPartEdit::Text("edited".to_owned()),
        EditTarget(text),
        Order(0),
        ChildOf(turn),
    ));
    let before = stats(&f.world);
    f.world.run_schedule(RigSchedule);
    let after = stats(&f.world);
    let second = requests(&mut f.world, f.run).pop().unwrap();
    assert_eq!(
        second.chat_history[2],
        Message::User {
            content: vec![UserContent::text("edited")]
        }
    );
    assert_eq!(second.chat_history[..2], first.chat_history[..2]);
    assert_eq!(after.renders - before.renders, 1, "the edited utterance");
    assert_eq!(after.hits - before.hits, 2);
    assert_eq!(after.evictions, before.evictions);
    assert_eq!(views_of(&mut f.world, f.run), [true, true, true]);
    // The next turn has no edit: the verbatim view, no render.
    let third = f.turn_expecting(0);
    assert_eq!(third.chat_history, first.chat_history);
}

#[test]
fn a_limit_applies_over_the_view() {
    let mut f = Fixture::new();
    let first = f.turn_expecting(3);
    f.world.entity_mut(f.run).insert(ToolResultLimit::new(4));
    let fresh = graph_messages(&mut f.world, f.run);
    let (second, delta) = f.turn();
    assert_eq!(delta.renders, 0, "{delta:?}");
    assert_eq!(delta.hits, 3);
    let Message::User { content } = &second.chat_history[1] else {
        panic!("the results");
    };
    let UserContent::ToolResult(result) = &content[0] else {
        panic!("the result");
    };
    let ToolResultContent::Text(text) = &result.content[0] else {
        panic!("the text");
    };
    assert_eq!(
        text.text,
        format!("th{}lt", TOOL_RESULT_LIMIT_MARKER.replace("{omitted}", "6"))
    );
    assert_eq!(fresh, first.chat_history, "the graph is verbatim");
    assert_eq!(graph_messages(&mut f.world, f.run), first.chat_history);
}

#[test]
fn an_asset_collection_drops_the_views() {
    let mut f = Fixture::new();
    let first = f.turn_expecting(3);
    collect_binary_assets(&mut f.world, []).unwrap();
    let second = f.turn_expecting(3);
    assert_eq!(second.chat_history, first.chat_history);
    f.turn_expecting(0);
}

#[test]
fn streaming_and_unary_assemble_the_same_request() {
    let mut f = Fixture::new();
    // The streamed run takes its first turn from `Advance`; the fixture's
    // unary run takes the one the fixture spawns. One pass, two requests.
    let streamed = f.world.spawn_run(f.agent, &history(), "next", true, None);
    let fresh = graph_messages(&mut f.world, f.run);
    assert_eq!(graph_messages(&mut f.world, streamed), fresh);
    let (unary, delta) = f.turn();
    assert_eq!(delta.assemblies, 2);
    assert_eq!(delta.renders, 6);
    assert_eq!(delta.hits, 0);
    assert_eq!(unary.chat_history, fresh);
    let streamed_request = requests(&mut f.world, streamed).pop().unwrap();
    assert_eq!(streamed_request.chat_history, unary.chat_history);
    let streams: Vec<bool> = f
        .world
        .query::<&PendingEffect>()
        .iter(&f.world)
        .filter_map(|effect| match &effect.kind {
            EffectKind::Completion { stream, .. } => Some(*stream),
            _ => None,
        })
        .collect();
    assert_eq!(streams.iter().filter(|s| **s).count(), 1);
    assert_eq!(streams.iter().filter(|s| !**s).count(), 1);
    assert_eq!(views_of(&mut f.world, streamed), [true, true, true]);
    let before = stats(&f.world);
    f.world.spawn((Turn, Fresh, Order(301), ChildOf(streamed)));
    f.world.run_schedule(RigSchedule);
    assert_eq!(stats(&f.world).renders, before.renders, "served from views");
}

// ---------------------------------------------------------------------------
// A controlled multi-turn tool run, live: every request against a fresh
// render, taken in `RigSet::Patch` the pass the request was folded.

#[derive(Resource, Default)]
struct Observed(Vec<(Vec<Message>, Vec<Message>, AssemblyStats)>);

fn observe(
    effects: Query<(&PendingEffect, &ChildOf), Added<PendingEffect>>,
    parents: Query<&ChildOf>,
    utterances: Query<(Entity, &ChildOf, &Order), With<Utterance>>,
    content: ContentGraph,
    stats: Res<AssemblyStats>,
    mut observed: ResMut<Observed>,
) {
    for (effect, turn) in &effects {
        let EffectKind::Completion { request, .. } = &effect.kind else {
            continue;
        };
        let run = parents.get(turn.parent()).unwrap().parent();
        let mut history: Vec<(u64, Entity)> = utterances
            .iter()
            .filter(|(_, parent, _)| parent.parent() == run)
            .map(|(entity, _, order)| (order.0, entity))
            .collect();
        history.sort();
        let fresh = history
            .into_iter()
            .map(|(_, entity)| content.message(entity).unwrap().to_message())
            .collect();
        observed
            .0
            .push((request.chat_history.clone(), fresh, *stats));
    }
}

fn script(tool_turns: usize) -> Vec<Vec<AssistantContent>> {
    let mut turns: Vec<Vec<AssistantContent>> = (0..tool_turns)
        .map(|i| {
            vec![call(
                &format!("c{i}"),
                "add",
                serde_json::json!({"x": i, "y": 1}),
            )]
        })
        .collect();
    turns.push(vec![AssistantContent::text("done")]);
    turns
}

fn tooling(turns: Vec<Vec<AssistantContent>>) -> (bevy_app::App, Entity, RequestsSeen) {
    let mut app = app();
    let (model, requests) = Scripted::new(MODEL, turns);
    let model = register(&mut app, MODEL, model);
    let tool = register(&mut app, ADD, Adder::new(ADD));
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut().entity_mut(agent).insert(MaxTurns(12));
    app.world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)));
    (app, agent, requests)
}

type RequestsSeen = std::sync::Arc<std::sync::Mutex<Vec<CompletionRequest>>>;

#[test]
fn every_request_equals_a_fresh_render_and_only_new_utterances_are_rendered() {
    let (mut app, agent, requests) = tooling(script(4));
    app.insert_resource(Observed::default());
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, observe.in_set(RigSet::Patch));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "add things", false, None);
    tick_until(&mut app, "the run settles", |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
    assert!(app.world().get::<Settled>(run).is_some());
    let observed = app.world().resource::<Observed>();
    assert_eq!(observed.0.len(), 5, "five requests");
    assert_eq!(requests.lock().unwrap().len(), 5);
    let mut last = AssemblyStats::default();
    let mut per_turn = Vec::new();
    for (turn, (request, fresh, stats)) in observed.0.iter().enumerate() {
        assert!(matches!(request[0], Message::System { .. }));
        assert_eq!(&request[1..], fresh.as_slice(), "turn {turn}");
        assert_eq!(fresh.len(), 1 + 2 * turn);
        per_turn.push((stats.renders - last.renders, stats.hits - last.hits));
        last = *stats;
    }
    // The prompt; then, each turn, the assistant utterance and the results.
    assert_eq!(per_turn, [(1, 0), (2, 1), (2, 3), (2, 5), (2, 7)]);
    assert_eq!(last.evictions, 0, "nothing was changed behind a view");
    let mut expected = vec![true; 9];
    expected.push(false);
    assert_eq!(
        views_of(app.world_mut(), run),
        expected,
        "every utterance a request read holds a view; the final answer, never read, none"
    );
}

#[test]
fn a_loaded_scene_assembles_identical_requests_and_rebuilds_its_views() {
    // The control: the whole run in one world, with a hold after the
    // second tool turn released at once.
    let (mut control, agent, control_requests) = tooling(script(3));
    let run = control
        .world_mut()
        .spawn_run(agent, &[], "add things", false, None);
    hold_after_tool_turn(control.world_mut(), run, "cache", 2).unwrap();
    tick_until(&mut control, "held", |world| {
        assert!(world.get::<Failed>(run).is_none());
        world
            .query::<&ToolTurnCommit>()
            .iter(world)
            .any(|commit| commit.turn >= 2)
    });
    assert_eq!(control_requests.lock().unwrap().len(), 2);
    let json = serde_json::to_string(&save_world(control.world_mut()).unwrap()).unwrap();
    assert!(
        !json.contains("cached_message") && !json.contains("CachedMessage"),
        "views are not scene data"
    );
    release_tool_turn_hold(control.world_mut(), run, "cache").unwrap();
    tick_until(&mut control, "the control settles", |world| {
        world.get::<Settled>(run).is_some()
    });
    let control_requests: Vec<_> = control_requests.lock().unwrap().clone();
    assert_eq!(control_requests.len(), 4);

    // The second world: the same script's tail, bound before the load.
    let scene: WorldScene = serde_json::from_str(&json).unwrap();
    let mut app = app();
    let (model, requests) = Scripted::new(MODEL, script(3).split_off(2));
    register(&mut app, MODEL, model);
    register(&mut app, ADD, Adder::new(ADD));
    let loaded = load_world(&scene, app.world_mut()).unwrap();
    let run = loaded
        .graph
        .iter()
        .copied()
        .find(|entity| app.world().get::<Run>(*entity).is_some())
        .unwrap();
    assert_eq!(
        views_of(app.world_mut(), run),
        vec![false; 5],
        "a loaded utterance has no view"
    );
    let fresh_before = graph_messages(app.world_mut(), run);
    release_tool_turn_hold(app.world_mut(), run, "cache").unwrap();
    tick_until(&mut app, "the resumed run settles", |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
    assert!(app.world().get::<Settled>(run).is_some());
    let requests: Vec<_> = requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].chat_history, control_requests[2].chat_history);
    assert_eq!(requests[1].chat_history, control_requests[3].chat_history);
    assert_eq!(&requests[0].chat_history[1..], fresh_before.as_slice());
    let stats = stats(app.world());
    assert_eq!(stats.renders, 5 + 2, "the loaded five, then the new two");
    assert_eq!(stats.hits, 5);
    let mut expected = vec![true; 7];
    expected.push(false);
    assert_eq!(
        views_of(app.world_mut(), run),
        expected,
        "rebuilt after the load; the final answer, never read, has none"
    );
}
