use std::sync::mpsc::Receiver;

use rig_coding_tools::{EditTool, ReadTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, calls, log_ended, press, reply, run_until, scripted_models,
    tui_view,
};
use rig_tui::{Focus, TuiTestScreen};
use serde_json::json;

use super::*;

/// `shell` that only says it ran.
fn shell(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin("shell"), ToolResult::success("ok".into()));
    }
}

/// Whether the turns' answers are held before they are committed, as a
/// reply still streaming is.
#[derive(Resource, Default)]
struct HoldAnswers(bool);

/// A hold on an answer.
#[derive(Component)]
struct AnswerHold;

fn hold_answers(
    calls: Query<Entity, (ReadyToSettle, With<TurnCall>)>,
    holds: Query<&HoldOf, With<AnswerHold>>,
    hold: Res<HoldAnswers>,
    mut commands: Commands,
) {
    for call in calls.iter().filter(|_| hold.0) {
        if !holds.iter().any(|held| held.0 == call) {
            commands.spawn((AnswerHold, HoldOf(call)));
        }
    }
}

/// Sends `text` to `agent` and runs until `turns` turns ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str, turns: usize) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
}

/// Runs a few frames: the view lays out, pads and scrolls over them.
fn settle(app: &mut App) {
    for _ in 0..4 {
        app.update();
    }
}

fn screen(app: &App) -> Vec<String> {
    app.world().resource::<TuiTestScreen>().rows()
}

/// The comparison's side panes, left and right.
fn panes(app: &mut App) -> Option<[Entity; 2]> {
    let world = app.world_mut();
    let mut views = world.query::<&CompareView>();
    let view = views.iter(world).next()?;
    let [left, right] = &view.sides;
    Some([left.pane, right.pane])
}

fn drawn(app: &App, pane: Entity) -> DrawnTranscript {
    let drawn = app.world().get::<DrawnTranscript>(pane);
    drawn.cloned().unwrap_or_default()
}

fn top(app: &App, pane: Entity) -> Option<(TranscriptItem, usize)> {
    let view = app.world().get::<TranscriptView>(pane);
    view.and_then(|view| view.top().cloned())
}

/// Whether the header row of `call` is lit in `drawn`.
fn lit(drawn: &DrawnTranscript, call: &str) -> bool {
    let rows = drawn.rows_of(&TranscriptItem::Call(CallId::from_wire(call)));
    let row = rows.and_then(|rows| drawn.rows().get(rows.start));
    row.is_some_and(|row| row.lit_total() > 0)
}

#[test]
fn two_branches_side_by_side_aligned_by_turn_and_the_picked_one_goes_on() {
    let exe = std::env::current_exe().unwrap_or_default();
    let project = exe.with_file_name(format!("branch-compare-{}", std::process::id()));
    std::fs::create_dir_all(project.join("src")).ok();
    std::fs::write(project.join("src/x.rs"), "fn x() {}\n").ok();
    let project = std::fs::canonicalize(&project).unwrap_or(project);
    let edit = json!({ "path": "src/x.rs", "edits": [{ "old_text": "fn x() {}", "new_text": "fn x() { 1 }" }] });
    let model = MockCompletionModel::from_stream_turns([
        reply("ONE"),
        calls("r1", "read", json!({ "path": "src/x.rs" })),
        calls("e1", "edit", edit),
        vec![
            MockStreamEvent::text("A-DONE"),
            MockStreamEvent::final_response_with_total_tokens(50),
        ],
        calls("s1", "shell", json!({ "command": "cargo test" })),
        reply("B-DONE"),
        reply("B-MORE"),
        reply("A-PARTIAL text"),
        reply("fine"),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .insert_resource(ProjectRoot(project.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, ReadTool, EditTool))
        .add_open_tool("shell", "Run a command.", ToolOptions::default(), shell)
        .init_resource::<Ended>()
        .init_resource::<HoldAnswers>()
        .add_observer(log_ended)
        .add_systems(AgentPass, hold_answers.in_set(ReviewSystems::Act))
        .insert_resource(TuiTestScreen::new(140, 24));
    tui_view(&mut app);
    app.add_plugins(BranchComparePlugin);
    let agent = app
        .world_mut()
        .spawn((Agent, ModelChoice("test/model".to_owned())))
        .id();
    app.finish();
    app.cleanup();
    app.update();
    app.world_mut().trigger(Focus { entity: agent });

    // Turn 1 on the main line; then branch A from it, and branch B.
    ask(&mut app, &wakes, agent, "start", 1);
    let branch = |app: &mut App, at: Option<usize>, title: &str| {
        let mut conversation = app.world_mut().get_mut::<Conversation>(agent);
        if let Some(conversation) = conversation.as_mut() {
            let at = at.unwrap_or(conversation.messages().len());
            conversation.branch_off(at, title, Author::user()).ok();
        }
    };
    branch(&mut app, None, "A");
    ask(&mut app, &wakes, agent, "fix x", 2);
    branch(&mut app, Some(2), "B");
    ask(&mut app, &wakes, agent, "fix x", 3);
    ask(&mut app, &wakes, agent, "more", 4);
    let kept_a = app
        .world()
        .get::<Conversation>(agent)
        .and_then(|conversation| {
            let kept = conversation.branches().into_iter();
            kept.filter(|kept| kept.branch.title == "A")
                .map(|kept| kept.id)
                .next()
        });
    if let (Some(mut conversation), Some(a)) =
        (app.world_mut().get_mut::<Conversation>(agent), kept_a)
    {
        conversation.switch_branch(a, Author::user()).ok();
    }
    app.update();

    // `/compare`: A (current) on the left, B, the most recent, on the right.
    app.world_mut().trigger(RunCommand::new(agent, "compare"));
    settle(&mut app);
    let rows = screen(&app);
    let shown = rows.join("\n");
    let header = rows.first().cloned().unwrap_or_default();
    let (left, right) = header.split_at(header.char_indices().nth(70).map_or(0, |(at, _)| at));
    assert!(
        left.contains("A (current)") && right.contains("B"),
        "{shown}"
    );
    let stats = rows.get(1).cloned().unwrap_or_default();
    assert!(stats.contains("1 turn · 50 tokens"), "{shown}");
    let files = rows.get(2).cloned().unwrap_or_default();
    let (left, right) = files.split_at(files.char_indices().nth(70).map_or(0, |(at, _)| at));
    assert!(left.contains("files changed: src/x.rs"), "{shown}");
    assert!(right.contains("files changed: none"), "{shown}");
    let same_row = rows
        .iter()
        .any(|row| row.contains("turn 1 · A-DONE") && row.contains("turn 1 · B-DONE"));
    assert!(same_row, "{shown}");

    // The calls only one side made are lit: A's edit, B's shell.
    let opened = panes(&mut app);
    assert!(opened.is_some(), "no comparison: {shown}");
    let [left, right] = opened.unwrap_or([Entity::PLACEHOLDER; 2]);
    assert!(lit(&drawn(&app, left), "e1"), "{shown}");
    assert!(lit(&drawn(&app, right), "s1"), "{shown}");

    // Down a turn: both sides move; `s`, then up: only the focused one.
    let before = (top(&app, left), top(&app, right));
    press(&mut app, "down");
    settle(&mut app);
    let after = (top(&app, left), top(&app, right));
    assert!(
        after.0 != before.0 && after.1 != before.1,
        "{before:?} {after:?}"
    );
    press(&mut app, "s");
    settle(&mut app);
    press(&mut app, "up");
    settle(&mut app);
    let alone = (top(&app, left), top(&app, right));
    assert!(
        alone.0 != after.0 && alone.1 == after.1,
        "{after:?} {alone:?}"
    );

    // 80 columns: the sides are tabs, A's shown.
    app.world_mut()
        .resource_mut::<TuiTestScreen>()
        .resize(80, 24);
    settle(&mut app);
    let rows = screen(&app);
    let shown = rows.join("\n");
    assert!(
        rows.first()
            .is_some_and(|row| row.contains(" A ") && row.contains(" B "))
    );
    assert!(
        shown.contains("turn 1 · A-DONE") && !shown.contains("B-DONE"),
        "{shown}"
    );

    // A turn streaming on A shows on its side before it is committed.
    app.world_mut().resource_mut::<HoldAnswers>().0 = true;
    app.world_mut()
        .trigger(Deliver::new(agent, "again", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        let holds = world.try_query_filtered::<(), With<AnswerHold>>();
        holds.is_some_and(|mut holds| holds.iter(world).next().is_some())
    });
    settle(&mut app);
    let partial = drawn(&app, left);
    let streams = partial
        .rows()
        .iter()
        .any(|row| row.text.contains("A-PARTIAL"));
    let committed = app
        .world()
        .get::<Conversation>(agent)
        .is_some_and(|conversation| format!("{:?}", conversation.messages()).contains("A-PARTIAL"));
    assert!(streams && !committed);
    app.world_mut().resource_mut::<HoldAnswers>().0 = false;
    let holds: Vec<Entity> = {
        let world = app.world_mut();
        let mut holds = world.query_filtered::<Entity, With<AnswerHold>>();
        holds.iter(world).collect()
    };
    for hold in holds {
        app.world_mut().entity_mut(hold).despawn();
    }
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() >= 5
    });

    // `p` on B: B is the active line, and the model reads it.
    press(&mut app, "tab");
    press(&mut app, "p");
    settle(&mut app);
    let title = app
        .world()
        .get::<Conversation>(agent)
        .map(|c| c.line_title().to_owned());
    assert_eq!(title.as_deref(), Some("B"));
    assert!(panes(&mut app).is_none());
    ask(&mut app, &wakes, agent, "next", 6);
    let sent = model
        .requests()
        .last()
        .map(|request| format!("{request:?}"));
    let sent = sent.unwrap_or_default();
    assert!(
        sent.contains("B-MORE") && !sent.contains("A-DONE"),
        "{sent}"
    );
}
