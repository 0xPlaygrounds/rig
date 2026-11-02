use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, calls, collect, connect, first_agent, log_ended, reply,
    run_until, scratch_home,
};
use serde_json::json;

use super::*;
use crate::helpers::{HELPER, HelperCommand, HelpersPlugin};
use rig_ecs::inbox::Recalled;

/// A stand-in for `rig -p --json` that reports two calls of 15 tokens,
/// notes its pid in the directory it is given and then works on.
const SPENDER: &str = r#"d=$1
usage='{"v":3,"type":"usage","model":"test/model","usage":{"input_tokens":10,"output_tokens":5}}'
echo $$ > "$d/pid"; echo "$usage"; echo "$usage"; sleep 30"#;

/// The text of every notice written so far.
fn seen(app: &App) -> Vec<String> {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices.iter().map(|notice| notice.text.clone()).collect()
}

/// An app on `settings` with the budget and helpers, its helpers the
/// [`SPENDER`], its agent on `model`; the directory the helpers note in.
fn budgeted(
    name: &str,
    settings: &str,
    model: &MockCompletionModel,
) -> (App, Receiver<()>, PathBuf) {
    let dir = scratch_home(&format!("rig-basics-budget-{name}"))
        .root()
        .to_path_buf();
    std::fs::create_dir_all(&dir).ok();
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let command = HelperCommand {
        program: "sh".into(),
        args: vec![
            "-c".into(),
            SPENDER.into(),
            "child".into(),
            dir.clone().into(),
        ],
    };
    app.insert_resource(Settings::from_toml(settings))
        .insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, HelpersPlugin, BudgetPlugin))
        .insert_resource(command)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    collect::<Recalled>(&mut app);
    app.finish();
    app.cleanup();
    app.update();
    let world = app.world_mut();
    if let (Some(agent), Some(connection)) = (first_agent(world), connect(model)) {
        world.entity_mut(agent).insert(connection);
    }
    (app, wakes, dir)
}

/// Runs `/line` for the agent, and returns the notices it wrote.
fn command(app: &mut App, line: &str) -> Vec<String> {
    let before = seen(app).len();
    if let Some(agent) = first_agent(app.world_mut()) {
        let run = RunCommand::new(agent, line.to_owned());
        app.world_mut().trigger(run);
    }
    app.update();
    seen(app).into_iter().skip(before).collect()
}

/// Whether the process noted in `dir` still runs.
fn alive(dir: &Path) -> bool {
    let pid = std::fs::read_to_string(dir.join("pid")).unwrap_or_default();
    !pid.trim().is_empty() && Path::new("/proc").join(pid.trim()).exists()
}

#[test]
fn a_reached_budget_stops_the_running_helper_and_refuses_the_next_call() {
    let model = MockCompletionModel::from_stream_turns([
        calls("h0", HELPER, json!({ "task": "spend" })),
        reply("never sent"),
    ]);
    let (mut app, wakes, dir) = budgeted("stop", "[budget]\ntokens = 20", &model);
    if let Some(agent) = first_agent(app.world_mut()) {
        app.world_mut()
            .trigger(Deliver::new(agent, "Go.", DeliveryMode::Steer));
    }
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    let ended = format!("{:?}", app.world().resource::<Ended>().0);
    assert!(
        ended.contains("Stopped") && ended.contains("Raise it with /budget"),
        "{ended}"
    );
    let spent = app.world().resource::<SessionUsage>().0.total;
    assert_eq!(spent.total_tokens(), 30, "both reports are in");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while alive(&dir) && std::time::Instant::now() < deadline {
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(!alive(&dir), "the stop reached the helper");

    if let Some(agent) = first_agent(app.world_mut()) {
        app.world_mut()
            .trigger(Deliver::new(agent, "Again.", DeliveryMode::Steer));
    }
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    let ended = format!("{:?}", app.world().resource::<Ended>().0.get(1));
    assert!(
        ended.contains("Budget reached: 30 of 20 tokens spent. Raise it with /budget"),
        "{ended}"
    );
    assert_eq!(model.request_count(), 1, "the refused call was never sent");
}

#[test]
fn what_was_handed_down_holds_a_run_without_a_budget_of_its_own() {
    let model = MockCompletionModel::from_stream_turns([reply("never sent")]);
    let (mut app, wakes, _) = budgeted("handed", "", &model);
    // As a child left nothing, or a garbled variable, gets.
    app.insert_resource(HandedDown(Limit {
        usd: Some(0.0),
        tokens: None,
    }));
    assert_eq!(
        command(&mut app, "budget"),
        [
            "No budget: 0 tokens spent. The rig that started this one left it at most $0.00 of $0.00."
        ]
    );
    if let Some(agent) = first_agent(app.world_mut()) {
        app.world_mut()
            .trigger(Deliver::new(agent, "Go.", DeliveryMode::Steer));
    }
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    let ended = format!("{:?}", app.world().resource::<Ended>().0);
    assert!(ended.contains(USD_VARIABLE), "{ended}");
    assert_eq!(model.request_count(), 0);
}

#[test]
fn budget_sets_shows_and_removes_the_sessions_budget_and_lowering_it_stops_work() {
    let model = MockCompletionModel::from_stream_turns([reply("unused")]);
    let (mut app, _wakes, _) = budgeted("command", "[budget]\nusd = 5.0", &model);
    assert_eq!(
        command(&mut app, "budget"),
        ["Budget: $0.00 of $5.00 spent."]
    );
    assert_eq!(
        command(&mut app, "budget 1000 tokens"),
        ["Budget: $0.00 of $5.00 and 0 of 1.0k tokens spent."]
    );
    // A refused command comes back to the input with why.
    command(&mut app, "budget lots");
    let recalled = &app.world().resource::<Collected<Recalled>>().0;
    let why = recalled.last().map(|recalled| recalled.why.as_deref());
    assert!(
        why.flatten()
            .is_some_and(|why| why.starts_with("Not a budget: lots")),
        "{why:?}"
    );
    assert_eq!(
        command(&mut app, "budget off"),
        ["No budget: 0 tokens spent."]
    );
    assert_eq!(app.world().resource::<Budget>().limit, Limit::default());
}

#[test]
fn lowering_the_budget_below_what_was_spent_stops_the_running_helper() {
    let model = MockCompletionModel::from_stream_turns([
        calls("h0", HELPER, json!({ "task": "spend" })),
        reply("never sent"),
    ]);
    let (mut app, wakes, dir) = budgeted("lower", "[budget]\ntokens = 100", &model);
    if let Some(agent) = first_agent(app.world_mut()) {
        app.world_mut()
            .trigger(Deliver::new(agent, "Go.", DeliveryMode::Steer));
    }
    run_until(&mut app, &wakes, |world| {
        world.resource::<SessionUsage>().0.total.total_tokens() == 30
    });
    assert!(app.world().resource::<Ended>().0.is_empty(), "it works on");
    assert_eq!(
        command(&mut app, "budget 25 tokens"),
        ["Budget: 30 of 25 tokens spent."]
    );
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    let ended = format!("{:?}", app.world().resource::<Ended>().0);
    assert!(ended.contains("Raise it with /budget"), "{ended}");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while alive(&dir) && std::time::Instant::now() < deadline {
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(!alive(&dir), "the stop reached the helper");
}
