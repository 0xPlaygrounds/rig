use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_core::catalog::Catalog;
use rig_core::completion::Usage;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, calls, collect, connect, first_agent, log_ended, reply,
    run_until, scratch_home,
};
use serde_json::json;

use super::*;
use crate::helpers::{HELPER, HelperCommand, HelpersPlugin};
use crate::telemetry::usage::Spent;
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
    budgeted_on(&MemoryStore::default(), name, settings, model)
}

/// As [`budgeted`], on `store`.
fn budgeted_on(
    store: &MemoryStore,
    name: &str,
    settings: &str,
    model: &MockCompletionModel,
) -> (App, Receiver<()>, PathBuf) {
    let dir = scratch_home(&format!("rig-basics-budget-{name}"))
        .root()
        .to_path_buf();
    std::fs::create_dir_all(&dir).ok();
    let (mut app, wakes) = app_on(store);
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

/// Whether the process noted in `dir` still runs, where `/proc` shows it.
fn alive(dir: &Path) -> bool {
    let pid = std::fs::read_to_string(dir.join("pid")).unwrap_or_default();
    !pid.trim().is_empty() && Path::new("/proc").join(pid.trim()).exists()
}

/// Has the agent call `helper`, whose child reports 30 tokens and works
/// on, past a limit of 20: the turn stops with `why`, the child is
/// stopped, and the next turn's model call is refused.
fn stops_the_helper_and_refuses_the_next_call(
    name: &str,
    settings: &str,
    handed: Limit,
    why: &str,
) {
    let model = MockCompletionModel::from_stream_turns([
        calls("h0", HELPER, json!({ "task": "spend" })),
        reply("never sent"),
    ]);
    let (mut app, wakes, dir) = budgeted(name, settings, &model);
    app.insert_resource(HandedDown(handed));
    if let Some(agent) = first_agent(app.world_mut()) {
        app.world_mut()
            .trigger(Deliver::new(agent, "Go.", DeliveryMode::Steer));
    }
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    let ended = format!("{:?}", app.world().resource::<Ended>().0);
    assert!(ended.contains("Stopped") && ended.contains(why), "{ended}");
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
        ended.contains("Budget reached: 30 of 20 tokens spent") && ended.contains(why),
        "{ended}"
    );
    assert_eq!(model.request_count(), 1, "the refused call was never sent");
}

#[test]
fn a_reached_budget_stops_the_running_helper_and_refuses_the_next_call() {
    let why = "Raise it with /budget";
    let unlimited = Limit::default();
    stops_the_helper_and_refuses_the_next_call("stop", "[budget]\ntokens = 20", unlimited, why);
}

/// A child whose parent left it 20 tokens holds itself to them, with no
/// budget of its own: so it stops even when its parent is gone.
#[test]
fn a_child_holds_to_what_its_parent_left_it() {
    let left = Limit {
        usd: None,
        tokens: Some(20),
    };
    let why = "left it (RIG_BUDGET_USD, RIG_BUDGET_TOKENS)";
    stops_the_helper_and_refuses_the_next_call("handed-stop", "", left, why);
}

/// The end of a reply of a million output tokens, which the catalog's
/// DeepSeek price makes cost something.
fn costly() -> MockStreamEvent {
    MockStreamEvent::final_response(Usage::new().input_tokens(0).output_tokens(1_000_000))
}

#[test]
fn a_dollar_budget_refuses_once_spent_and_is_kept_with_the_session() {
    let spec = Catalog::builtin().resolve("deepseek/deepseek-flash");
    assert!(spec.is_ok(), "the catalog lists DeepSeek");
    let Ok(spec) = spec.map(|found| found.shared()) else {
        return;
    };
    let model = MockCompletionModel::from_stream_turns([
        vec![MockStreamEvent::text("one"), costly()],
        vec![MockStreamEvent::text("never sent"), costly()],
    ])
    .priced_as(spec);
    let store = MemoryStore::default();
    let (mut app, wakes, _) = budgeted_on(&store, "usd", "", &model);
    command(&mut app, "budget 0.01");
    if let Some(agent) = first_agent(app.world_mut()) {
        app.world_mut()
            .trigger(Deliver::new(agent, "Go.", DeliveryMode::Steer));
    }
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    let spent = app.world().resource::<SessionUsage>().0.total.cost;
    assert!(spent > 0.01, "the first call is priced past it: ${spent}");
    if let Some(agent) = first_agent(app.world_mut()) {
        app.world_mut()
            .trigger(Deliver::new(agent, "Again.", DeliveryMode::Steer));
    }
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    let ended = format!("{:?}", app.world().resource::<Ended>().0.get(1));
    assert!(
        ended.contains(&format!("Budget reached: ${spent:.2} of $0.01 spent")),
        "{ended}"
    );
    assert_eq!(model.request_count(), 1, "the refused call was never sent");

    // The session keeps its budget.
    app.update();
    drop(app);
    let (restarted, _, _) = budgeted_on(&store, "usd", "", &model);
    let kept = restarted.world().resource::<Budget>().0.usd;
    assert_eq!(kept, Some(0.01));
}

#[test]
fn what_was_handed_down_holds_a_run_without_a_budget_of_its_own() {
    let model = MockCompletionModel::from_stream_turns([reply("never sent")]);
    let (mut app, wakes, _) = budgeted("handed", "", &model);
    // Garbled variables allow nothing, as a child left nothing gets.
    let handed = handed_down(Some("NaN".to_owned()), Some("lots".to_owned()));
    assert_eq!(
        handed,
        Limit {
            usd: Some(0.0),
            tokens: Some(0)
        }
    );
    assert_eq!(handed_down(None, None), Limit::default());
    app.insert_resource(HandedDown(handed));
    assert_eq!(
        command(&mut app, "budget"),
        [
            "No budget: 0 tokens spent. The rig that started this one left it at most $0.00 of $0.00 and 0 of 0 tokens."
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
fn budget_sets_shows_and_removes_the_sessions_budget() {
    let model = MockCompletionModel::from_stream_turns([reply("unused")]);
    let (mut app, _wakes, _) = budgeted("command", "[budget]\nusd = 5.0", &model);
    assert_eq!(
        command(&mut app, "budget"),
        ["Budget: $0.00 of $5.00 spent."]
    );
    assert_eq!(
        command(&mut app, "budget 1k tokens"),
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
    assert_eq!(app.world().resource::<Budget>().0, Limit::default());

    // A call with no price is in no dollar budget, and both say so.
    let world = app.world_mut();
    let agent = first_agent(world);
    world.trigger(Spent {
        agent,
        turn: None,
        model: None,
        purpose: None,
        usage: Usage::new().input_tokens(900).output_tokens(100),
    });
    let said = command(&mut app, "budget 1.00");
    assert_eq!(
        said,
        [
            "Budget: $0.00 of $1.00 spent. 1 of 1 calls had no price, such as a subscription \
             model's, and are not in the dollars: /budget 500000 tokens counts them."
        ]
    );
    command(&mut app, "budget 1200 tokens");
    #[cfg(feature = "tui")]
    shows_in_the_status_line(&app);
}

/// The status line's budget: the dollars marked as leaving out a call,
/// red past 80% of the tokens.
#[cfg(feature = "tui")]
fn shows_in_the_status_line(app: &App) {
    let status = &app.world().resource::<rig_tui::AppStatus>().0;
    let item = status.iter().find(|item| item.name == "budget");
    let shown = item.map(|item| (item.text.as_str(), item.tone));
    assert_eq!(
        shown,
        Some(("$0.00+/$1.00 1.0k/1.2k tokens", rig_tui::Tone::Red)),
        "past 80% of the tokens"
    );
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
    // Where processes can be seen, as on Linux, the helper is seen working.
    let seen = Path::new("/proc/self").exists();
    assert!(!seen || alive(&dir), "the helper works");
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
