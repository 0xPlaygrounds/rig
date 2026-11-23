use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use bevy_app::prelude::*;
use bevy_ecs::message::Messages;
use bevy_ecs::prelude::*;
use rig_core::catalog::Catalog;
use rig_core::completion::message::Origin;
use rig_core::completion::{CompletionRequest, CompletionResponse, Message, Usage};
use rig_core::conversation::Author;
use rig_core::effect::{Answered, Decider};
use rig_core::message::{AssistantContent, ToolResultContent, UserContent};
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::streaming::{Part, PartKind, StreamEvent};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_core::tool::ToolResult;

use super::{
    Fed, ModelRequest, STEERED_SKIP, StreamDelta, feed, stream_partials, write_unstreamed,
};
use crate::AgentPlugin;
use crate::agent::TurnOutcome;
use crate::effects::Handler;
use crate::inbox::{Deliver, DeliveryMode};
use crate::model::Connection;
use crate::spawn::spawn_root;
use crate::stage::{
    Admitted, AgentPass, Decision, HoldOf, Outcome, Settled, Stage, SupersededBy, ToolCalls,
};
use crate::tasks::Wake;
use crate::tools::{AppToolsExt, ToolCalled, ToolOptions, ToolRequest};

/// A world the kernel's delta writers run in.
fn world() -> World {
    let mut world = World::new();
    world.init_resource::<Messages<StreamDelta>>();
    world
}

/// A model call's request.
fn call() -> ModelRequest {
    ModelRequest::new(CompletionRequest::new(Message::user("hi")))
}

/// What a pass does with deltas: what streamed is read in Arrive, and
/// what did not stream is written in Commit.
fn pass(world: &mut World) {
    assert!(world.run_system_cached(stream_partials).is_ok());
    assert!(world.run_system_cached(write_unstreamed).is_ok());
}

/// A reply of `text`, as its outcome.
fn replied(text: &str) -> Outcome {
    let choice = vec![AssistantContent::text(text)];
    let origin = Origin::new("test.api", "test", "model");
    let reply = CompletionResponse::new(choice, Usage::default(), origin, serde_json::Value::Null);
    Outcome(Ok(Answered::Completion(Arc::new(reply))))
}

/// Runs [`pass`] `passes` times; the text each call's deltas
/// carried.
fn texts(world: &mut World, passes: usize) -> Vec<(Entity, String)> {
    for _ in 0..passes {
        pass(world);
    }
    let deltas = world.resource::<Messages<StreamDelta>>();
    let mut cursor = deltas.get_cursor();
    let texts = cursor.read(deltas).filter_map(|delta| match &delta.event {
        StreamEvent::Text { text, .. } => Some((delta.effect, text.clone())),
        _ => None,
    });
    texts.collect()
}

fn text_event(text: &str) -> Fed {
    let text = StreamEvent::Text {
        part: Part::new(0),
        text: text.to_owned(),
    };
    Fed::Item(web_time::SystemTime::now(), Some(text))
}

#[test]
fn a_reply_that_did_not_stream_comes_as_deltas_once() {
    let mut world = world();
    let answered = world.spawn((call(), replied("in place"))).id();
    // A stream that ended with no event.
    let (sender, fed) = feed();
    let ended = world.spawn((call(), fed, replied("ended"))).id();
    assert!(
        sender
            .try_send(Fed::Item(web_time::SystemTime::now(), None))
            .is_ok()
    );
    let mut texts = texts(&mut world, 3);
    // Calls are read in no set order.
    texts.sort_by(|left, right| left.1.cmp(&right.1));
    assert_eq!(
        texts,
        [
            (ended, "ended".to_owned()),
            (answered, "in place".to_owned())
        ]
    );
}

#[test]
fn a_reply_that_streamed_any_text_gets_no_copy() {
    let mut world = world();
    // Part of a reply came, then the call was answered otherwise.
    let (sender, fed) = feed();
    let call = world.spawn((call(), fed)).id();
    assert!(sender.try_send(text_event("Hel")).is_ok());
    pass(&mut world);
    world.entity_mut(call).insert(replied("Hello"));
    assert!(sender.try_send(text_event("lo")).is_ok());
    let texts = texts(&mut world, 2);
    assert_eq!(texts, [(call, "Hel".to_owned()), (call, "lo".to_owned())]);
}

#[test]
fn a_superseded_reply_writes_no_deltas() {
    let mut world = world();
    let successor = world.spawn(call()).id();
    world.spawn((call(), replied("replaced"), SupersededBy(successor)));
    assert_eq!(texts(&mut world, 2), []);
}

#[test]
fn each_part_of_a_reply_that_did_not_stream_starts_grows_and_ends() {
    let mut world = world();
    let choice = vec![
        AssistantContent::reasoning("hm"),
        AssistantContent::text(""),
    ];
    let origin = Origin::new("test.api", "test", "model");
    let reply = CompletionResponse::new(choice, Usage::default(), origin, serde_json::Value::Null);
    world.spawn((call(), Outcome(Ok(Answered::Completion(Arc::new(reply))))));
    pass(&mut world);
    let deltas = world.resource::<Messages<StreamDelta>>();
    let events: Vec<StreamEvent> = deltas
        .get_cursor()
        .read(deltas)
        .map(|delta| delta.event.clone())
        .collect();
    let (reasoning, text) = (Part::new(0), Part::new(1));
    assert_eq!(
        events,
        [
            StreamEvent::Start {
                part: reasoning,
                kind: PartKind::Reasoning,
                name: None,
            },
            StreamEvent::Reasoning {
                part: reasoning,
                text: "hm".to_owned(),
            },
            StreamEvent::End {
                part: reasoning,
                content: AssistantContent::reasoning("hm"),
            },
            // An empty text part grows by nothing.
            StreamEvent::Start {
                part: text,
                kind: PartKind::Text,
                name: None,
            },
            StreamEvent::End {
                part: text,
                content: AssistantContent::text(""),
            },
        ]
    );
}

#[test]
fn a_reply_whose_stream_had_no_text_gets_its_text_after_what_streamed() {
    let mut world = world();
    let (sender, fed) = feed();
    let call = world.spawn((call(), fed)).id();
    let reasoning = Part::new(0);
    let began = StreamEvent::Start {
        part: reasoning,
        kind: PartKind::Reasoning,
        name: None,
    };
    let thought = StreamEvent::Reasoning {
        part: reasoning,
        text: "hm".to_owned(),
    };
    for event in [began.clone(), thought.clone()] {
        let item = Fed::Item(web_time::SystemTime::now(), Some(event));
        assert!(sender.try_send(item).is_ok());
    }
    pass(&mut world);
    world.entity_mut(call).insert(replied("hello"));
    pass(&mut world);
    pass(&mut world);
    let deltas = world.resource::<Messages<StreamDelta>>();
    let events: Vec<StreamEvent> = deltas
        .get_cursor()
        .read(deltas)
        .map(|delta| delta.event.clone())
        .collect();
    let text = Part::new(1);
    assert_eq!(
        events,
        [
            began,
            thought,
            StreamEvent::Start {
                part: text,
                kind: PartKind::Text,
                name: None,
            },
            StreamEvent::Text {
                part: text,
                text: "hello".to_owned(),
            },
            StreamEvent::End {
                part: text,
                content: AssistantContent::text("hello"),
            },
        ]
    );
}

#[test]
fn what_a_superseded_call_streamed_before_it_was_read_is_still_written() {
    let mut world = world();
    let successor = world.spawn(call()).id();
    let (sender, fed) = feed();
    let call = world.spawn((call(), fed, SupersededBy(successor))).id();
    assert!(sender.try_send(text_event("Hel")).is_ok());
    assert_eq!(texts(&mut world, 1), [(call, "Hel".to_owned())]);
}

/// How long a scripted turn may take before a test gives up.
const DEADLINE: Duration = Duration::from_secs(10);

/// Open tool calls that started, in start order: `(call_id, entity)`.
#[derive(Resource, Default)]
struct StartedCalls(Vec<(String, Entity)>);

/// Turn outcomes recorded as turns settle.
#[derive(Resource, Default)]
struct Ended(Vec<TurnOutcome>);

fn step_tool(called: On<ToolCalled<serde_json::Value>>, mut started: ResMut<StartedCalls>) {
    started
        .0
        .push((called.tool_call.id.to_string(), called.call));
}

/// A headless kernel app with a scripted model and one conflicting open tool
/// (`step`), returning `(app, wakes, agent)`.
fn batch_app(model: &MockCompletionModel) -> Option<(App, Receiver<()>, Entity)> {
    let (sender, wakes) = channel();
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin))
        .insert_resource(Wake::new(move || {
            sender.send(()).ok();
        }))
        .add_open_tool("step", "Run one step.", ToolOptions::default(), step_tool)
        .init_resource::<StartedCalls>()
        .init_resource::<Ended>()
        .add_observer(
            |settled: On<bevy_ecs::lifecycle::Add<Settled>>,
             turns: Query<&TurnOutcome>,
             mut ended: ResMut<Ended>| {
                ended.0.extend(turns.get(settled.entity).cloned());
            },
        );
    app.finish();
    let spec = Catalog::builtin()
        .resolve("deepseek/deepseek-flash")
        .ok()?
        .shared();
    let handler = ErasedHandler::new(ModelAdapter::<Completion>::new(
        spec.reference(),
        model.clone(),
    ));
    let connection = Connection {
        spec,
        handler: Handler(handler),
    };
    let agent = spawn_root(app.world_mut(), connection).id();
    app.update();
    Some((app, wakes, agent))
}

fn run_until(app: &mut App, wakes: &Receiver<()>, done: impl Fn(&World) -> bool) {
    let started = Instant::now();
    while started.elapsed() < DEADLINE {
        app.update();
        if done(app.world()) {
            return;
        }
        wakes.recv_timeout(Duration::from_millis(100)).ok();
    }
}

fn finish_started(app: &mut App, at: usize, output: &str) {
    let entity = app
        .world()
        .resource::<StartedCalls>()
        .0
        .get(at)
        .map(|c| c.1);
    let decision = entity.and_then(|e| app.world_mut().get_mut::<Decision>(e));
    assert!(decision.is_some(), "started call {at} exists");
    if let Some(mut decision) = decision {
        decision.decide(Decider::plugin("step"), ToolResult::success(output.into()));
    }
}

fn started_ids(app: &App) -> Vec<String> {
    app.world()
        .resource::<StartedCalls>()
        .0
        .iter()
        .map(|(id, _)| id.clone())
        .collect()
}

/// Paired `(call_id, is_error, text)` from tool results in a model request's
/// chat history.
fn tool_results_in(request: &CompletionRequest) -> Vec<(String, bool, String)> {
    let mut results = Vec::new();
    for message in &request.chat_history {
        let Message::User { content } = &**message else {
            continue;
        };
        for item in content {
            if let UserContent::ToolResult(result) = item {
                let text = result
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        ToolResultContent::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                results.push((result.call.to_string(), result.is_error, text));
            }
        }
    }
    results
}

/// User text parts in a model request's chat history, in order.
fn user_texts_in(request: &CompletionRequest) -> Vec<String> {
    let mut texts = Vec::new();
    for message in &request.chat_history {
        let Message::User { content } = &**message else {
            continue;
        };
        for item in content {
            if let UserContent::Text(text) = item {
                texts.push(text.text.clone());
            }
        }
    }
    texts
}

#[test]
fn user_steer_skips_unstarted_calls_in_conflicting_batch() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "step", serde_json::json!({ "file": "a.rs" })),
            MockStreamEvent::tool_call("c2", "step", serde_json::json!({ "file": "b.rs" })),
            MockStreamEvent::tool_call("c3", "step", serde_json::json!({ "file": "test" })),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("understood"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let setup = batch_app(&model);
    assert!(setup.is_some(), "built-in catalog has test model");
    let Some((mut app, wakes, agent)) = setup else {
        return;
    };

    app.world_mut()
        .trigger(Deliver::new(agent, "run three steps", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<StartedCalls>().0.is_empty()
    });
    assert_eq!(started_ids(&app), ["c1"]);

    // User steers while c1 is still running.
    app.world_mut().trigger(Deliver::new(
        agent,
        "stop—don't touch b.rs",
        DeliveryMode::Steer,
    ));
    app.update();
    assert_eq!(started_ids(&app), ["c1"], "unstarted calls never start");

    // c1 finishes cleanly; c2 and c3 are already skipped.
    finish_started(&mut app, 0, "c1 ok");
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    assert_eq!(started_ids(&app), ["c1"]);
    assert_eq!(model.request_count(), 2);
    let requests = model.requests();
    assert!(requests.get(1).is_some(), "expected second model request");
    let Some(second) = requests.get(1) else {
        return;
    };
    assert_eq!(
        tool_results_in(second),
        [
            ("c1".to_owned(), false, "c1 ok".to_owned()),
            ("c2".to_owned(), true, STEERED_SKIP.to_owned()),
            ("c3".to_owned(), true, STEERED_SKIP.to_owned()),
        ]
    );
    let texts = user_texts_in(second);
    assert_eq!(
        texts.last().map(String::as_str),
        Some("stop—don't touch b.rs")
    );
    // Steering message comes after the paired tool results.
    let items: Vec<&UserContent> = second
        .chat_history
        .iter()
        .filter_map(|msg| match &**msg {
            Message::User { content } => Some(content.iter()),
            _ => None,
        })
        .flatten()
        .collect();
    let last_result = items
        .iter()
        .rposition(|item| matches!(item, UserContent::ToolResult(_)));
    let steer_pos = items
        .iter()
        .position(|item| matches!(item, UserContent::Text(t) if t.text == "stop—don't touch b.rs"));
    assert!(last_result.zip(steer_pos).is_some_and(|(r, s)| r < s));
}

#[test]
fn agent_steer_does_not_skip_unstarted_calls() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "step", serde_json::json!({})),
            MockStreamEvent::tool_call("c2", "step", serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("all done"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let setup = batch_app(&model);
    assert!(setup.is_some());
    let Some((mut app, wakes, agent)) = setup else {
        return;
    };

    app.world_mut()
        .trigger(Deliver::new(agent, "run two steps", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<StartedCalls>().0.is_empty()
    });
    assert_eq!(started_ids(&app), ["c1"]);

    // Background subagent report arrives with DeliveryMode::Steer.
    app.world_mut().trigger(
        Deliver::new(agent, "subagent finished", DeliveryMode::Steer)
            .with_author(Author::agent("sub-1", None)),
    );
    app.update();

    finish_started(&mut app, 0, "c1 ok");
    run_until(&mut app, &wakes, |world| {
        world.resource::<StartedCalls>().0.len() >= 2
    });
    assert_eq!(started_ids(&app), ["c1", "c2"]);

    finish_started(&mut app, 1, "c2 ok");
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    assert_eq!(model.request_count(), 2);
    let requests = model.requests();
    assert!(requests.get(1).is_some(), "expected second model request");
    let Some(second) = requests.get(1) else {
        return;
    };
    assert_eq!(
        tool_results_in(second),
        [
            ("c1".to_owned(), false, "c1 ok".to_owned()),
            ("c2".to_owned(), false, "c2 ok".to_owned()),
        ]
    );
    let texts = user_texts_in(second);
    assert!(
        texts
            .last()
            .is_some_and(|t| t.contains("subagent finished")),
        "{texts:?}"
    );
}

#[test]
fn queued_message_does_not_skip_unstarted_calls() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "step", serde_json::json!({})),
            MockStreamEvent::tool_call("c2", "step", serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("first done"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("follow-up done"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let setup = batch_app(&model);
    assert!(setup.is_some());
    let Some((mut app, wakes, agent)) = setup else {
        return;
    };

    app.world_mut()
        .trigger(Deliver::new(agent, "run two steps", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<StartedCalls>().0.is_empty()
    });
    assert_eq!(started_ids(&app), ["c1"]);

    // User queues a follow-up (Tab) while c1 is running.
    app.world_mut()
        .trigger(Deliver::new(agent, "follow up later", DeliveryMode::Queue));
    app.update();

    finish_started(&mut app, 0, "c1 ok");
    run_until(&mut app, &wakes, |world| {
        world.resource::<StartedCalls>().0.len() >= 2
    });
    assert_eq!(started_ids(&app), ["c1", "c2"]);

    finish_started(&mut app, 1, "c2 ok");
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    assert_eq!(model.request_count(), 3);
    let requests = model.requests();
    assert!(requests.get(1).is_some(), "expected second model request");
    let Some(second) = requests.get(1) else {
        return;
    };
    assert_eq!(
        tool_results_in(second),
        [
            ("c1".to_owned(), false, "c1 ok".to_owned()),
            ("c2".to_owned(), false, "c2 ok".to_owned()),
        ]
    );
    assert!(
        user_texts_in(second)
            .iter()
            .all(|t| !t.contains("follow up later")),
        "queued message waits until the turn would end"
    );
    assert!(requests.get(2).is_some(), "expected third model request");
    let Some(third) = requests.get(2) else {
        return;
    };
    assert_eq!(
        user_texts_in(third).last().map(String::as_str),
        Some("follow up later")
    );
}

#[test]
fn user_steer_on_same_frame_as_call_completion_skips_next_call() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "step", serde_json::json!({})),
            MockStreamEvent::tool_call("c2", "step", serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("stopped in time"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let setup = batch_app(&model);
    assert!(setup.is_some());
    let Some((mut app, wakes, agent)) = setup else {
        return;
    };

    app.world_mut()
        .trigger(Deliver::new(agent, "run two steps", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<StartedCalls>().0.is_empty()
    });
    assert_eq!(started_ids(&app), ["c1"]);

    // Answer c1 with Outcome directly (as an async task finishing in Arrive
    // would) and deliver a user steer on the exact same pass: c1 settles in
    // Commit and wait_in_order clears Waiting on c2 before accept runs in
    // Start, so start's steered guard must prevent c2 from launching.
    let c1 = app
        .world()
        .resource::<StartedCalls>()
        .0
        .first()
        .map(|c| c.1);
    assert!(c1.is_some(), "c1 should be started");
    if let Some(c1) = c1 {
        app.world_mut()
            .entity_mut(c1)
            .insert(Outcome::from(ToolResult::success("c1 ok".into())));
    }
    app.world_mut()
        .trigger(Deliver::new(agent, "stop now", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    assert_eq!(started_ids(&app), ["c1"]);
    let requests = model.requests();
    assert!(requests.get(1).is_some(), "expected second model request");
    let Some(second) = requests.get(1) else {
        return;
    };
    assert_eq!(
        tool_results_in(second),
        [
            ("c1".to_owned(), false, "c1 ok".to_owned()),
            ("c2".to_owned(), true, STEERED_SKIP.to_owned()),
        ]
    );
    assert_eq!(
        user_texts_in(second).last().map(String::as_str),
        Some("stop now")
    );
}

#[test]
fn user_steer_does_not_dismiss_plugin_held_call() {
    #[derive(Resource, Default)]
    struct Held(Option<(Entity, Entity)>);

    fn hold_c2(
        calls: Query<(Entity, &ToolRequest), (Added<Admitted>, ToolCalls)>,
        mut held: ResMut<Held>,
        mut commands: Commands,
    ) {
        for (entity, request) in &calls {
            if request.call.id.to_string() == "c2" {
                let hold = commands.spawn(HoldOf(entity)).id();
                held.0 = Some((entity, hold));
            }
        }
    }

    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "step", serde_json::json!({})),
            MockStreamEvent::tool_call("c2", "step", serde_json::json!({})),
            MockStreamEvent::tool_call("c3", "step", serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let setup = batch_app(&model);
    assert!(setup.is_some());
    let Some((mut app, wakes, agent)) = setup else {
        return;
    };
    app.init_resource::<Held>()
        .add_systems(AgentPass, hold_c2.in_set(Stage::Prepare));

    app.world_mut()
        .trigger(Deliver::new(agent, "run three steps", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<StartedCalls>().0.is_empty() && world.resource::<Held>().0.is_some()
    });

    app.world_mut()
        .trigger(Deliver::new(agent, "steer now", DeliveryMode::Steer));
    app.update();

    let held_c2 = app.world().resource::<Held>().0;
    assert!(held_c2.is_some(), "c2 should be held");
    let Some((c2, hold)) = held_c2 else {
        return;
    };
    assert!(
        app.world().get::<Outcome>(c2).is_none(),
        "held call is not dismissed by steering"
    );
    assert!(
        app.world()
            .get::<Decision>(c2)
            .is_some_and(|decision| !decision.is_decided()),
        "held call stays undecided while held"
    );

    // Plugin resolves its hold (e.g. user denies the prompt) and c1 finishes.
    if let Some(mut decision) = app.world_mut().get_mut::<Decision>(c2) {
        decision.decide(Decider::plugin("approval"), Outcome::refused("denied"));
    }
    app.world_mut().despawn(hold);
    finish_started(&mut app, 0, "c1 ok");
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    assert_eq!(started_ids(&app), ["c1"]);
    let requests = model.requests();
    assert!(requests.get(1).is_some(), "expected second model request");
    let Some(second) = requests.get(1) else {
        return;
    };
    assert_eq!(
        tool_results_in(second),
        [
            ("c1".to_owned(), false, "c1 ok".to_owned()),
            ("c2".to_owned(), true, "denied".to_owned()),
            ("c3".to_owned(), true, STEERED_SKIP.to_owned()),
        ]
    );
}
