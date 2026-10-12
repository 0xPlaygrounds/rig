use std::collections::HashSet;
use std::sync::mpsc::Receiver;

use rig_basics::ProjectContextPlugin;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Ended, app_in, calls, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
};
use rig_telemetry::{SessionSpending, UsagePlugin};

use super::*;

/// How many classifier calls settled.
#[derive(Resource, Default)]
struct Classified(usize);

fn count_classified(
    settled: On<Add<Settled>>,
    calls: Query<(), With<Classifying>>,
    mut classified: ResMut<Classified>,
) {
    if calls.contains(settled.entity) {
        classified.0 += 1;
    }
}

/// While on, every classifier call is held before it starts, as a slow
/// classifier that has not answered yet.
#[derive(Resource, Default)]
struct Slow(bool);

/// The hold a slow classifier call waits on.
#[derive(Component)]
struct SlowHold;

fn hold_classifier(
    calls: Query<Entity, (With<Classifying>, Undecided, ModelCalls)>,
    slow: Res<Slow>,
    mut held: Local<HashSet<Entity>>,
    mut commands: Commands,
) {
    for call in &calls {
        if slow.0 && held.insert(call) {
            commands.spawn((SlowHold, HoldOf(call)));
        }
    }
}

/// `shell`, answered without running anything.
fn ran(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin("test"), ToolResult::success("done".into()));
    }
}

fn waiting(world: &mut World) -> Vec<Proposal> {
    let mut proposals = world.query_filtered::<&Proposal, Without<Outcome>>();
    proposals.iter(world).cloned().collect()
}

/// The user types `text`; frames run until `turns` turns ended.
fn say(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str, turns: usize) {
    let message = Deliver::new(agent, text, DeliveryMode::Queue);
    app.world_mut().trigger(message);
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() == turns
    });
}

fn run_command(app: &mut App, wakes: &Receiver<()>, agent: Entity, line: &str) {
    app.world_mut().trigger(RunCommand::new(agent, line));
    run_until(app, wakes, |world| {
        let open = world.try_query_filtered::<(), (With<Proposal>, Without<Outcome>)>();
        open.is_some_and(|mut open| open.iter(world).next().is_none())
    });
}

fn classified(app: &mut App, wakes: &Receiver<()>, count: usize) {
    run_until(app, wakes, |world| {
        world.resource::<Classified>().0 == count
    });
    app.update();
}

#[test]
fn a_correction_becomes_a_saved_rule_without_holding_up_the_turns() {
    let agent_model = MockCompletionModel::from_stream_turns([
        calls(
            "c1",
            "shell",
            serde_json::json!({ "command": "npm install" }),
        ),
        reply("Installed."),
        reply("Sorry, pnpm it is."),
        reply("Added."),
        reply("Understood."),
        reply("Right."),
        reply("Using make."),
        reply("The tests pass."),
    ]);
    let classifier = MockCompletionModel::from_stream_turns([
        reply("Use pnpm, not npm."),
        reply("Use pnpm, not npm."),
        reply("Never edit files under gen/."),
        reply("`Never edit files under gen/.`"),
        reply("Run the tests with make test."),
        reply("NONE"),
    ]);
    let home = scratch_home("fh-correction-learner");
    let project = home.root().join("project");
    fs::create_dir_all(&project).ok();
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    let settings = format!("[{NAME}]\nmodel = \"test/classifier\"\n");
    app.insert_resource(Settings::from_toml(&settings))
        .insert_resource(scripted_models([
            ("test/agent", &agent_model),
            ("test/classifier", &classifier),
        ]))
        .insert_resource(ProjectRoot(project.clone()))
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            ProjectContextPlugin,
            UsagePlugin,
        ))
        .add_plugins(CorrectionLearnerPlugin)
        .add_open_tool("shell", "Run a command.", ToolOptions::default(), ran)
        .add_systems(AgentPass, hold_classifier.in_set(Stage::Gate))
        .init_resource::<Slow>()
        .init_resource::<Classified>()
        .add_observer(count_classified)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    // Someone sits at a front.
    app.world_mut().spawn(Answerer::default());
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/agent".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });

    // The agent runs `npm install`; the user corrects it; the user saves
    // the rule proposed.
    say(&mut app, &wakes, agent, "Install the dependencies.", 1);
    say(&mut app, &wakes, agent, "no, we use pnpm here", 2);
    classified(&mut app, &wakes, 1);
    let shown = waiting(app.world_mut());
    let rules: Vec<&str> = shown
        .iter()
        .map(|proposal| proposal.rule.as_str())
        .collect();
    assert_eq!(rules, ["Use pnpm, not npm."]);
    run_command(&mut app, &wakes, agent, "rules save");
    let file = fs::read_to_string(project.join("AGENTS.md")).unwrap_or_default();
    assert!(
        file.ends_with("## Learned rules\n\n- Use pnpm, not npm.\n"),
        "{file}"
    );

    // The next request has it; the same rule again is not proposed.
    say(&mut app, &wakes, agent, "Add lodash.", 3);
    let sent = agent_model
        .requests()
        .last()
        .map(|request| format!("{request:?}"));
    assert!(sent.unwrap_or_default().contains("- Use pnpm, not npm."));
    classified(&mut app, &wakes, 2);
    assert!(waiting(app.world_mut()).is_empty());

    // A dismissed rule is not proposed again.
    say(
        &mut app,
        &wakes,
        agent,
        "don't touch the generated files",
        4,
    );
    classified(&mut app, &wakes, 3);
    assert_eq!(waiting(app.world_mut()).len(), 1);
    run_command(&mut app, &wakes, agent, "rules dismiss");
    say(&mut app, &wakes, agent, "I said leave gen/ alone", 5);
    classified(&mut app, &wakes, 4);
    assert!(waiting(app.world_mut()).is_empty());

    // A slow classifier: the next turns run while it has not answered.
    app.world_mut().resource_mut::<Slow>().0 = true;
    say(&mut app, &wakes, agent, "no, use the make target", 6);
    say(&mut app, &wakes, agent, "now run the tests", 7);
    assert_eq!(app.world().resource::<Ended>().0.len(), 7);
    assert_eq!(app.world().resource::<Classified>().0, 4);
    let mut pending = app
        .world_mut()
        .query_filtered::<(&Effect, &Classifying), Without<Outcome>>();
    let mut pending: Vec<(u64, u64)> = (pending.iter(app.world()))
        .map(|(effect, classifying)| (effect.id().as_u64(), classifying.turn))
        .collect();
    pending.sort_unstable();
    let turns: Vec<u64> = pending.iter().map(|(_, turn)| *turn).collect();
    assert_eq!(turns.len(), 2, "{pending:?}");

    // Its answer still comes, tied to the turn it read.
    app.world_mut().resource_mut::<Slow>().0 = false;
    let mut holds = app
        .world_mut()
        .query_filtered::<(Entity, &HoldOf), With<SlowHold>>();
    let holds: Vec<(Entity, Entity)> = holds
        .iter(app.world())
        .map(|(hold, of)| (hold, of.0))
        .collect();
    let mut by_id: Vec<(u64, Entity)> = holds
        .iter()
        .filter_map(|&(hold, call)| Some((app.world().get::<Effect>(call)?.id().as_u64(), hold)))
        .collect();
    by_id.sort_unstable();
    for (_, hold) in by_id.iter().take(1) {
        app.world_mut().despawn(*hold);
    }
    classified(&mut app, &wakes, 5);
    let shown = waiting(app.world_mut());
    let first = shown
        .first()
        .map(|proposal| (proposal.rule.as_str(), proposal.turn));
    assert_eq!(
        first,
        Some((
            "Run the tests with make test.",
            turns.first().copied().unwrap_or(u64::MAX)
        ))
    );
    for (_, hold) in by_id.iter().skip(1) {
        app.world_mut().despawn(*hold);
    }
    classified(&mut app, &wakes, 6);
    assert_eq!(waiting(app.world_mut()).len(), 1);

    // The classifier's cost is the session's.
    let spent = app.world().resource::<SessionSpending>();
    assert!(
        spent.0.purposes.contains_key(NAME),
        "{:?}",
        spent.0.purposes
    );
}

#[test]
fn rules_go_under_their_heading_and_come_out_by_number() {
    let text =
        "# Project\n\nBuild with make.\n\n## Learned rules\n\n- Use pnpm.\n\n## Other\n\nText.\n";
    let added = with_rule(text, "Never edit gen/.");
    assert_eq!(learned(&added), ["Use pnpm.", "Never edit gen/."]);
    assert!(added.ends_with("## Other\n\nText.\n"), "{added}");
    let (removed, rule) = without_rule(&added, 1).unwrap_or_default();
    assert_eq!(rule, "Use pnpm.");
    assert_eq!(learned(&removed), ["Never edit gen/."]);
    assert_eq!(
        with_rule("", "Use pnpm."),
        "## Learned rules\n\n- Use pnpm.\n"
    );
}
