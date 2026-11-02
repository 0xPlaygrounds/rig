use std::pin::pin;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use futures::FutureExt;

use rig_core::catalog::{Catalog, Connector};
use rig_core::effect::{Disposition, EffectKind, EffectRecord};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::prelude::*;
use rig_harness_test_support::{MemoryStore, app_on, calls, connect, reply, run_until, start_call};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::*;
use rig_ecs::spawn::spawn;

/// Every effect recorded so far.
#[derive(Resource, Default)]
struct Kept(Vec<EffectRecord>);

fn keep(mut recorded: MessageReader<EffectRecorded>, mut kept: ResMut<Kept>) {
    kept.0
        .extend(recorded.read().map(|EffectRecorded(record)| record.clone()));
}

/// The arguments of the test's tools, which take none.
#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

/// Answers a `look` call at once.
fn look(called: On<ToolCalled<NoArgs>>, mut commands: Commands) {
    let seen = Outcome::from(ToolResult::success("seen".into()));
    commands.entity(called.call).insert_if_new(seen);
}

/// On a hold the test's gate spawned.
#[derive(Component)]
struct Hold;

/// Holds every `touch` call, and refuses every `poke` call, when it is
/// about to start.
fn gate(calls: Query<(Entity, &ToolRequest), ReadyToStart>, mut commands: Commands) {
    for (call, request) in &calls {
        match request.call.function.name.as_str() {
            "touch" => {
                commands.spawn((Hold, HoldOf(call)));
            }
            "poke" => {
                commands
                    .entity(call)
                    .insert_if_new(Outcome::refused("no poking"));
            }
            _ => {}
        }
    }
}

/// An app with code mode, `look`, `touch` and `poke`, its gate, and an agent
/// answered by `model`.
fn app(model: &MockCompletionModel) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let options = |footprint| ToolOptions {
        footprint: Some(footprint),
    };
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .add_plugins((AgentPlugin, SessionLogPlugin, SteelPlugin))
        .add_open_tool(
            "look",
            "Look.",
            options(Touches::Reads(Scope::Anything)),
            look,
        )
        .add_open_tool(
            "touch",
            "Touch.",
            options(Touches::Writes(Scope::Anything)),
            look,
        )
        .add_open_tool(
            "poke",
            "Poke.",
            options(Touches::Writes(Scope::Anything)),
            look,
        )
        .init_resource::<Kept>()
        .add_systems(Last, keep)
        .add_systems(AgentPass, gate.in_set(Stage::Gate));
    let agent = spawn_root(app.world_mut()).id();
    if let Some(connection) = connect(model) {
        let choice = ModelChoice(connection.spec.reference());
        app.world_mut()
            .entity_mut(agent)
            .insert((connection, choice));
    }
    app.finish();
    app.update();
    (app, wakes, agent)
}

/// The records of calls of `tool`.
fn records<'a>(world: &'a World, tool: &str) -> Vec<&'a EffectRecord> {
    let kept = world.resource::<Kept>().0.iter();
    let named = |record: &&EffectRecord| matches!(&record.kind, EffectKind::ToolCall { name, .. } if name == tool);
    kept.filter(named).collect()
}

/// The `touch` calls the gate holds.
fn held_touches(world: &World) -> usize {
    let Some(mut calls) = world.try_query_filtered::<&ToolRequest, With<Holds>>() else {
        return 0;
    };
    calls
        .iter(world)
        .filter(|request| request.call.function.name.as_str() == "touch")
        .count()
}

/// Runs frames of `app` until `asked` answers, or gives up.
fn answered<T>(app: &mut App, wakes: &Receiver<()>, asked: impl Future<Output = T>) -> Option<T> {
    let mut asked = pin!(asked);
    for _ in 0..100 {
        if let Some(answer) = asked.as_mut().now_or_never() {
            return Some(answer);
        }
        app.update();
        wakes.recv_timeout(Duration::from_millis(100)).ok();
    }
    None
}

/// Whether `agent`'s conversation holds `text`.
fn heard(world: &World, agent: Entity, text: &str) -> bool {
    let messages = world.get::<Conversation>(agent).map(|said| said.messages());
    format!("{messages:?}").contains(text)
}

#[test]
fn a_programs_tool_calls_are_gated_and_recorded_under_its_call() {
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes, agent) = app(&model);
    let code = r#"(define seen (call-tool "look")) (call-tool "poke") seen"#;
    let call = start_call(
        app.world_mut(),
        agent,
        (RUN_STEEL, "s1"),
        json!({ "code": code }),
    );
    let call = call.unwrap_or(Entity::PLACEHOLDER);
    run_until(&mut app, &wakes, |world| {
        world.get::<Outcome>(call).is_some() && !records(world, "poke").is_empty()
    });

    // The gate refused `poke` before it ran, so the program failed on it.
    let result = app.world().get::<Outcome>(call).map(Outcome::tool_result);
    let said = result
        .map(|result| result.output().render())
        .unwrap_or_default();
    assert!(said.contains("no poking"), "{said}");
    app.update();
    let steel = records(app.world(), RUN_STEEL)
        .first()
        .map(|record| record.id);
    let (looked, poked) = (records(app.world(), "look"), records(app.world(), "poke"));
    assert_eq!((looked.len(), poked.len()), (1, 1));
    for record in looked.iter().chain(&poked) {
        assert!(
            steel.is_some() && record.parent == steel,
            "{:?}",
            record.parent
        );
    }
    let dispositions = (
        looked.first().map(|r| r.disposition),
        poked.first().map(|r| r.disposition),
    );
    assert_eq!(
        dispositions,
        (
            Some(Disposition::Ran),
            Some(Disposition::AnsweredBeforeStart)
        )
    );
}

#[test]
fn an_interrupt_ends_a_held_tool_call_of_a_program_and_its_hold() {
    let code = r#"(call-tool "touch")"#;
    let model =
        MockCompletionModel::from_stream_turns([calls("s1", RUN_STEEL, json!({ "code": code }))]);
    let (mut app, wakes, agent) = app(&model);
    app.world_mut()
        .trigger(Deliver::new(agent, "Touch it.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| held_touches(world) > 0);
    assert_eq!(held_touches(app.world()), 1);

    app.world_mut().trigger(StopTurn::new(agent));
    app.world_mut().flush();
    // At once: the program's call ended, and with it its held call and the
    // hold, so nothing it asked can still start.
    let world = app.world_mut();
    let touches = world.query::<&ToolRequest>().iter(world).count();
    let holds = world.query_filtered::<(), With<Hold>>().iter(world).count();
    assert_eq!(
        (touches, holds),
        (1, 0),
        "only the program's own call is left, settled"
    );
    app.update();
    let touched = records(app.world(), "touch");
    let disposition = touched.first().map(|record| record.disposition);
    assert_eq!(disposition, Some(Disposition::Cancelled));
    let steel = records(app.world(), RUN_STEEL);
    assert_eq!(
        steel.first().map(|record| record.disposition),
        Some(Disposition::Cancelled)
    );
}

#[test]
fn a_program_never_gives_an_agent_a_tool_its_parent_lacks() {
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes, agent) = app(&model);
    // As the drive test's intermediary: allowed `run_steel` and `look` only.
    let access = ToolAccess::all().narrow([RUN_STEEL, "look"]);
    app.world_mut().entity_mut(agent).insert(access);
    let run = |app: &mut App, id: &str, tools: &str| {
        let code = format!(r#"(spawn-agent "leaf" (hash 'tools (list {tools})))"#);
        let call = start_call(
            app.world_mut(),
            agent,
            (RUN_STEEL, id),
            json!({ "code": code }),
        );
        let call = call.unwrap_or(Entity::PLACEHOLDER);
        run_until(app, &wakes, |world| world.get::<Outcome>(call).is_some());
        let result = app.world().get::<Outcome>(call).map(Outcome::tool_result);
        result
            .map(|result| result.output().render())
            .unwrap_or_default()
    };
    let refused = run(&mut app, "s1", r#""poke""#);
    assert!(
        refused.contains("`poke` is not one of your tools (look, run_steel)"),
        "{refused}"
    );
    let granted = run(&mut app, "s2", r#""look""#);
    assert!(!granted.contains("not one of your tools"), "{granted}");
    let world = app.world_mut();
    let leaves: Vec<Option<Vec<String>>> = world
        .query_filtered::<&ToolAccess, With<SpawnedBy>>()
        .iter(world)
        .map(|access| access.names().map(<[String]>::to_vec))
        .collect();
    assert_eq!(leaves, [Some(vec!["look".to_owned()])]);
}

#[test]
fn a_program_never_spawns_an_agent_deeper_than_the_subagent_limit() {
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes, root) = app(&model);
    // At depth 2, the default `SpawnPolicy::max_depth`: `task` refuses
    // this agent a child.
    let child = spawn(app.world_mut(), Origin::ChildOf(root), AgentSpec::default());
    let child = child.unwrap_or(Entity::PLACEHOLDER);
    let deepest = spawn(
        app.world_mut(),
        Origin::ChildOf(child),
        AgentSpec::default(),
    );
    let deepest = deepest.unwrap_or(Entity::PLACEHOLDER);
    let code = json!({ "code": r#"(spawn-agent "leaf")"# });
    let call = start_call(app.world_mut(), deepest, (RUN_STEEL, "s1"), code);
    let call = call.unwrap_or(Entity::PLACEHOLDER);
    run_until(&mut app, &wakes, |world| {
        world.get::<Outcome>(call).is_some()
    });
    let world = app.world_mut();
    let mut parents = world.query::<&SpawnedBy>();
    let too_deep = parents
        .iter(world)
        .filter(|by| by.parent() == deepest)
        .count();
    assert_eq!(too_deep, 0, "an agent was spawned at depth 3");
}

#[test]
fn a_queued_job_whose_call_was_stopped_does_nothing() {
    let model = MockCompletionModel::from_stream_turns([calls("t1", "touch", json!({}))]);
    let (mut app, wakes, agent) = app(&model);
    app.world_mut()
        .trigger(Deliver::new(agent, "Touch it.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| held_touches(world) > 0);
    let world = app.world_mut();
    let call = world
        .query_filtered::<Entity, (With<ToolRequest>, With<Holds>)>()
        .iter(world)
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    let id = world.get::<AgentId>(agent).cloned().unwrap_or_default();
    let harness = world.resource::<Harness>().within(call);
    let author = Author::agent(id.0.clone(), None);
    let before = harness.send(id.clone(), "Sent before.".to_owned(), author.clone());
    let before = answered(&mut app, &wakes, before).and_then(Result::ok);
    let text = "Sent for a stopped call.";
    let mut sent = pin!(harness.send(id, text.to_owned(), author));
    assert!(
        sent.as_mut().now_or_never().is_none(),
        "answered before its frame"
    );

    app.world_mut().trigger(StopTurn::new(agent));
    app.world_mut().flush();
    let sent = answered(&mut app, &wakes, sent);
    assert!(matches!(sent, Some(Err(_))), "{sent:?}");
    app.update();
    assert!(!heard(app.world(), agent, text));
    // The receipt of what it sent before went with the call.
    assert!(before.is_some());
    let unbound = app.world().resource::<Harness>().clone();
    let reply = unbound.reply(before.unwrap_or(RequestId(String::new())));
    let reply = answered(&mut app, &wakes, reply);
    let gone = matches!(&reply, Some(Err(why)) if why.starts_with("no request"));
    assert!(gone, "{reply:?}");
}

#[test]
fn a_reply_read_after_a_compaction_still_arrives_and_only_once() {
    let model = MockCompletionModel::from_stream_turns([reply("Done.")]);
    let (mut app, wakes, agent) = app(&model);
    let id = app
        .world()
        .get::<AgentId>(agent)
        .cloned()
        .unwrap_or_default();
    let harness = app.world().resource::<Harness>().clone();
    let author = Author::agent(id.0.clone(), None);
    let sent = answered(
        &mut app,
        &wakes,
        harness.send(id, "Do it.".to_owned(), author),
    );
    let request = sent
        .and_then(Result::ok)
        .unwrap_or(RequestId(String::new()));
    run_until(&mut app, &wakes, |world| heard(world, agent, "Done."));
    run_until(&mut app, &wakes, |world| {
        world.get::<ActiveTurn>(agent).is_none()
    });

    // A compaction the user asked for is a turn of its own, which a plugin
    // ends.
    app.world_mut()
        .spawn((Name::new("compaction"), TurnOf(agent)));
    app.world_mut()
        .trigger(StopTurn::plugin_alone(agent, "compaction", "compacted"));
    app.world_mut().flush();
    run_until(&mut app, &wakes, |world| {
        world.get::<ActiveTurn>(agent).is_none()
    });

    let first = answered(&mut app, &wakes, harness.reply(request.clone()));
    let text = match &first {
        Some(Ok(TurnOutcome::Answered(message))) => message.final_answer(),
        _ => None,
    };
    assert_eq!(text.as_deref(), Some("Done."), "{first:?}");
    let again = answered(&mut app, &wakes, harness.reply(request));
    assert!(matches!(again, Some(Err(_))), "{again:?}");
}
