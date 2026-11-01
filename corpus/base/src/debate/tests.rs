use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::settings::Settings;
use rig_harness_test_support::MemoryStore;

use super::*;
use rig_harness_test_support::{app_on, reply, run_until, scripted_models};

#[test]
fn debaters_answer_each_other_and_the_judges_verdict_reaches_the_agent() {
    let main = MockCompletionModel::from_stream_turns([reply("unused")]);
    let pro = MockCompletionModel::from_stream_turns([reply("P1"), reply("P2")]);
    let con = MockCompletionModel::from_stream_turns([reply("C1"), reply("C2")]);
    let judge = MockCompletionModel::from_stream_turns([reply("Verdict: P")]);
    let models = scripted_models([
        ("debate-main/m", &main),
        ("debate-pro/m", &pro),
        ("debate-con/m", &con),
        ("debate-judge/m", &judge),
    ]);

    let (mut app, wakes) = app_on(&MemoryStore::default());
    let settings = "[debate]\npro = \"debate-pro/m\"\ncon = \"debate-con/m\"\n\
                    judge = \"debate-judge/m\"\n";
    app.insert_resource(models)
        .insert_resource(Settings::from_toml(settings))
        .add_plugins((AgentPlugin, SessionLogPlugin, DebatePlugin));
    let host = app
        .world_mut()
        .spawn((Agent, ModelChoice("debate-main/m".to_owned())))
        .id();
    app.finish();
    app.update();

    let line = "debate --rounds 2 Should the parser be hand-written?".to_owned();
    app.world_mut().trigger(RunCommand::new(host, line));
    let said = |world: &World| {
        let conversation = world.get::<Conversation>(host);
        format!(
            "{:?}",
            conversation.map(|conversation| conversation.messages())
        )
    };
    run_until(&mut app, &wakes, |world| {
        !world.entity(host).contains::<Debate>() && said(world).contains("Verdict: P")
    });

    let asked = |model: &MockCompletionModel, at: usize| format!("{:?}", model.requests().get(at));
    assert!(asked(&pro, 1).contains("C1"), "{}", asked(&pro, 1));
    assert!(asked(&con, 1).contains("P1"), "{}", asked(&con, 1));
    let judged = asked(&judge, 0);
    for answer in ["P1", "P2", "C1", "C2"] {
        assert!(judged.contains(answer), "{judged}");
    }
    assert!(said(app.world()).contains("Verdict: P"));
    assert_eq!(main.request_count(), 0, "a note starts no turn");
    let world = app.world_mut();
    let mut debaters = world.query::<(&Debater, &SpawnedBy)>();
    let spawned: Vec<Entity> = debaters.iter(world).map(|(_, by)| by.0).collect();
    assert_eq!(spawned, [host; 3], "three debaters, spawned by the agent");
}

/// Holds every model call, so the debaters' turns keep running.
fn hold_turn_calls(calls: Query<Entity, (ReadyToStart, With<TurnCall>)>, mut commands: Commands) {
    for call in &calls {
        commands.spawn(HoldOf(call));
    }
}

#[test]
fn a_plugin_ending_the_hosts_turn_alone_leaves_the_debate_running_and_the_users_stop_ends_it_once()
{
    let models = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let models = scripted_models([
        ("debate-pro/m", &models),
        ("debate-con/m", &models),
        ("debate-judge/m", &models),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let settings = "[debate]\npro = \"debate-pro/m\"\ncon = \"debate-con/m\"\n\
                    judge = \"debate-judge/m\"\n";
    app.insert_resource(models)
        .insert_resource(Settings::from_toml(settings))
        .add_plugins((AgentPlugin, SessionLogPlugin, DebatePlugin))
        .add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate))
        .init_resource::<StillRunning>()
        .add_observer(note_still_running);
    let host = app
        .world_mut()
        .spawn((Agent, ModelChoice("debate-pro/m".to_owned())))
        .id();
    app.finish();
    app.update();

    app.world_mut()
        .trigger(RunCommand::new(host, "debate Tabs or spaces?".to_owned()));
    // How many seated agents run a turn.
    let arguing = |world: &World| {
        let debate = world.get::<Debate>(host);
        let seated = debate.map_or(&[][..], |debate| &debate.seated);
        let running = |&&(_, agent): &&(Role, Entity)| world.get::<ActiveTurn>(agent).is_some();
        seated.iter().filter(running).count()
    };
    run_until(&mut app, &wakes, |world| arguing(world) == 2);

    // As `/compact` ends the host's own turn.
    let compacted = StopTurn::plugin_alone(host, "compaction", "compacted");
    app.world_mut().trigger(compacted);
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(arguing(app.world()), 2, "both debaters still argue");

    // Each arguing debater has streamed some text when the user stops.
    let world = app.world_mut();
    let mut arguers = world.query_filtered::<Entity, (With<Debater>, With<ActiveTurn>)>();
    let arguers: Vec<Entity> = arguers.iter(world).collect();
    let mut streaming = world.query_filtered::<&mut Partial, With<TurnCall>>();
    for mut partial in streaming.iter_mut(world) {
        partial.text = "half".to_owned();
    }
    app.world_mut().trigger(StopTurn::new(host));
    for _ in 0..3 {
        app.update();
    }
    let world = app.world_mut();
    assert!(world.get::<Debate>(host).is_none(), "the debate ended");
    let mut running = world.query_filtered::<(), (With<Debater>, With<ActiveTurn>)>();
    assert_eq!(running.iter(world).count(), 0, "no debater runs");
    assert_eq!(arguers.len(), 2);
    for conversation in arguers
        .iter()
        .filter_map(|&agent| world.get::<Conversation>(agent))
    {
        let aborted = conversation.messages().iter().filter(|message| {
            matches!(&***message, message::Message::Assistant(reply)
                if matches!(reply.stop, Some(message::StopReason::Aborted(_))))
        });
        assert_eq!(aborted.count(), 1, "what streamed is kept once");
        assert_eq!(
            conversation.halted(),
            Some(Halt::Stopped),
            "the user stopped it"
        );
    }
    // No `StopReached` came while its agent's turn was still on, and each
    // debater's first says what its turn record says: the user stopped it.
    // The debate's own stop of the idle debater comes after.
    let still = world.resource::<StillRunning>();
    assert_eq!(still.running, 0);
    for &arguer in &arguers {
        let first = still.causes.iter().find(|(agent, _)| *agent == arguer);
        assert_eq!(first.map(|(_, cause)| cause), Some(&StopCause::User));
    }
}

/// The `StopReached` that came: how many while the agent's turn was still
/// on, and each one's agent and cause.
#[derive(Resource, Default)]
struct StillRunning {
    running: usize,
    causes: Vec<(Entity, StopCause)>,
}

fn note_still_running(
    reached: On<StopReached>,
    turns: Query<(), With<ActiveTurn>>,
    mut still: ResMut<StillRunning>,
) {
    let agent = reached.event_target();
    if turns.contains(agent) {
        still.running += 1;
    }
    still.causes.push((agent, reached.cause().clone()));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(DebatePlugin);
}
