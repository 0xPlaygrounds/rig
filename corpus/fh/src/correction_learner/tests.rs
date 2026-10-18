use std::collections::HashSet;
use std::fs;
use std::sync::mpsc::Receiver;

use rig_basics::telemetry::{SessionUsage, UsagePlugin};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Ended, answer_open, app_in, calls, first_agent, log_ended, reply, run_until, scratch_home,
    scripted_models,
};

use super::*;

/// The proposals' question.
const TITLE: &str = "Remember for this project?";

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
    calls: Query<Entity, (With<Classifying>, ReadyToStart, ModelCalls)>,
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

/// The rules of the proposals shown and not answered.
fn waiting(world: &mut World) -> Vec<(String, u64)> {
    let mut proposals = world.query_filtered::<&Proposal, (With<Ask>, Without<Outcome>)>();
    let shown = proposals.iter(world);
    shown
        .map(|shown| (shown.rule.clone(), shown.turn))
        .collect()
}

/// The user types `text`; frames run until `turns` turns ended.
fn say(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str, turns: usize) {
    let message = Deliver::new(agent, text, DeliveryMode::Queue);
    app.world_mut().trigger(message);
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() == turns
    });
}

/// Frames run until `count` classifier calls settled and what they
/// answered is checked.
fn classified(app: &mut App, wakes: &Receiver<()>, count: usize) {
    run_until(app, wakes, |world| {
        world.resource::<Classified>().0 == count
    });
    app.update();
}

/// The user answers the proposal shown; frames run until `file` says
/// `done`.
fn answer(
    app: &mut App,
    wakes: &Receiver<()>,
    choice: &str,
    file: &Path,
    done: impl Fn(&str) -> bool,
) {
    assert!(answer_open(app.world_mut(), TITLE, &Answer::choice(choice)));
    run_until(app, wakes, |_| {
        done(&fs::read_to_string(file).unwrap_or_default())
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
        reply("Fine."),
        reply("Using make."),
        reply("The tests pass."),
    ]);
    let classifier = MockCompletionModel::from_stream_turns([
        reply("Use pnpm, not npm."),
        reply("Use pnpm, not npm."),
        reply("Never edit files under gen/."),
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
        .add_plugins((AgentPlugin, SessionLogPlugin, UsagePlugin))
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
    let agents_md = project.join("AGENTS.md");
    let dismissed = project.join(".rig").join("dismissed-rules.txt");

    // The agent runs `npm install`; the user corrects it; the user saves
    // the rule proposed.
    say(&mut app, &wakes, agent, "Install the dependencies.", 1);
    say(&mut app, &wakes, agent, "no, we use pnpm here", 2);
    classified(&mut app, &wakes, 1);
    let shown = waiting(app.world_mut());
    let rules: Vec<&str> = shown.iter().map(|(rule, _)| rule.as_str()).collect();
    assert_eq!(rules, ["Use pnpm, not npm."]);
    answer(&mut app, &wakes, "save", &agents_md, |file| {
        file.contains("pnpm")
    });
    let file = fs::read_to_string(&agents_md).unwrap_or_default();
    assert!(
        file.ends_with("## Learned rules\n\n- Use pnpm, not npm.\n"),
        "{file}"
    );

    // The agent's next request has a note of the saved rule, which starts
    // no turn of its own; the same rule again is not proposed.
    say(&mut app, &wakes, agent, "Add lodash.", 3);
    let sent = agent_model
        .requests()
        .last()
        .map(|request| format!("{request:?}"))
        .unwrap_or_default();
    assert!(
        sent.contains("Saved to ") && sent.contains(": Use pnpm, not npm."),
        "{sent}"
    );
    classified(&mut app, &wakes, 2);
    assert!(waiting(app.world_mut()).is_empty());

    // The proposal waits beside the work: the user's next line is a
    // message to the agent, and the same rule is not shown twice.
    say(
        &mut app,
        &wakes,
        agent,
        "don't touch the generated files",
        4,
    );
    classified(&mut app, &wakes, 3);
    say(&mut app, &wakes, agent, "and not gen/ either", 5);
    classified(&mut app, &wakes, 4);
    assert_eq!(waiting(app.world_mut()).len(), 1);

    // A dismissed rule is not proposed again.
    answer(&mut app, &wakes, "dismiss", &dismissed, |file| {
        file.contains("gen/")
    });
    say(&mut app, &wakes, agent, "I said leave gen/ alone", 6);
    classified(&mut app, &wakes, 5);
    assert!(waiting(app.world_mut()).is_empty());

    // A slow classifier: the next turns run while it has not answered.
    app.world_mut().resource_mut::<Slow>().0 = true;
    say(&mut app, &wakes, agent, "no, use the make target", 7);
    say(&mut app, &wakes, agent, "now run the tests", 8);
    assert_eq!(app.world().resource::<Ended>().0.len(), 8);
    assert_eq!(app.world().resource::<Classified>().0, 5);

    // Its answer still comes, tied to the turn it read.
    app.world_mut().resource_mut::<Slow>().0 = false;
    let mut holds = app
        .world_mut()
        .query_filtered::<(Entity, &HoldOf), With<SlowHold>>();
    let holds: Vec<(Entity, Entity)> = (holds.iter(app.world()))
        .map(|(hold, of)| (hold, of.0))
        .collect();
    let mut by_id: Vec<(u64, u64, Entity)> = (holds.iter())
        .filter_map(|&(hold, call)| {
            let effect = app.world().get::<Effect>(call)?;
            let turn = app.world().get::<Classifying>(call)?.turn;
            Some((effect.id().as_u64(), turn, hold))
        })
        .collect();
    by_id.sort_unstable();
    assert_eq!(by_id.len(), 2, "{by_id:?}");
    for &(_, _, hold) in by_id.iter().take(1) {
        app.world_mut().despawn(hold);
    }
    classified(&mut app, &wakes, 6);
    let first_turn = by_id.first().map(|&(_, turn, _)| turn);
    let shown = waiting(app.world_mut());
    let shown: Vec<(&str, Option<u64>)> = (shown.iter())
        .map(|(rule, turn)| (rule.as_str(), Some(*turn)))
        .collect();
    assert_eq!(shown, [("Run the tests with make test.", first_turn)]);
    for &(_, _, hold) in by_id.iter().skip(1) {
        app.world_mut().despawn(hold);
    }
    classified(&mut app, &wakes, 7);
    assert_eq!(waiting(app.world_mut()).len(), 1);

    // The classifier's cost is the session's.
    let spent = app.world().resource::<SessionUsage>();
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
