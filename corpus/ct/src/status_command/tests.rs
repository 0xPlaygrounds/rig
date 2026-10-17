use std::fs;
use std::os::unix::fs::PermissionsExt as _;
use std::path::Path;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use rig_basics::telemetry::UsagePlugin;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::watch::WatchPlugin;
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, first_agent, log_ended, reply, run_until, scratch_home,
    scripted_models,
};

use super::*;

/// Appends its input to `inputs.jsonl`, then, as `mode` says, sleeps,
/// fails, or prints the model and turn.
const SCRIPT: &str = r#"#!/bin/sh
dir=$(dirname "$0")
input=$(cat)
printf '%s\n' "$input" >> "$dir/inputs.jsonl"
case "$(cat "$dir/mode" 2>/dev/null)" in
  sleep) sleep 5 ;;
  fail) echo nope >&2; exit 1 ;;
esac
model=$(printf '%s' "$input" | sed -n 's/.*"model":"\([^"]*\)".*/\1/p')
turn=$(printf '%s' "$input" | sed -n 's/.*"turn":\([0-9]*\).*/\1/p')
printf '\033[32mM:%s T:%s\033[0m\nsecond line\n' "$model" "$turn"
"#;

fn write_script(path: &Path, text: &str) {
    assert!(fs::write(path, text).is_ok());
    assert!(fs::set_permissions(path, fs::Permissions::from_mode(0o755)).is_ok());
}

/// What the item says, if it shows.
fn item(world: &World) -> Option<(String, Tone)> {
    let status = world.get_resource::<AppStatus>()?;
    let item = status.0.iter().find(|item| item.name == ITEM.name)?;
    Some((item.text.clone(), item.tone))
}

fn text(world: &World) -> String {
    item(world).map(|(text, _)| text).unwrap_or_default()
}

/// The inputs the script read, one per run.
fn inputs(dir: &Path) -> Vec<String> {
    let text = fs::read_to_string(dir.join("inputs.jsonl")).unwrap_or_default();
    text.lines().map(str::to_owned).collect()
}

/// Runs frames for `wall` of wall-clock time.
fn run_for(app: &mut App, wakes: &Receiver<()>, wall: Duration) {
    let started = Instant::now();
    while started.elapsed() < wall {
        app.update();
        wakes.recv_timeout(Duration::from_millis(20)).ok();
    }
}

/// A change of the command's input: what the session spent.
fn spend(app: &mut App) {
    app.world_mut().resource_mut::<SessionUsage>().0.total.cost += 0.01;
    app.update();
}

fn failures(world: &World) -> usize {
    let notices = &world.resource::<Collected<Notice>>().0;
    notices
        .iter()
        .filter(|notice| notice.text.contains("nope"))
        .count()
}

#[test]
fn the_item_follows_the_session_through_slow_failing_and_replaced_scripts() {
    let home = scratch_home("corpus-ct-status-command");
    let dir = home.root().to_path_buf();
    assert!(fs::create_dir_all(&dir).is_ok());
    let script = dir.join("status.sh");
    write_script(&script, SCRIPT);

    let model = MockCompletionModel::from_stream_turns([reply("hello")]);
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    let setting = format!("{SECTION} = '{}'", script.display());
    app.insert_resource(Settings::from_toml(&setting))
        .insert_resource(scripted_models([("test/model", &model)]))
        // The terminal view's run, without rig-tui opening a terminal.
        .insert_resource(Invoked {
            terminal: true,
            ..Invoked::default()
        })
        .add_front(Front::new(FRONT).on_terminal())
        .insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, WatchPlugin))
        .add_plugins((
            rig_harness::clock::ClockPlugin,
            rig_harness::pick::PickPlugin,
            UsagePlugin,
            StatusCommandPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();

    // One turn: the item shows the model and the turn, in the script's
    // first colour, and the script read every listed field.
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let world = app.world_mut();
    world.trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
    });
    world.trigger(Deliver::new(agent, "Hi.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty() && text(world) == "M:test/model T:1"
    });
    let shown = item(app.world());
    assert_eq!(shown, Some(("M:test/model T:1".to_owned(), Tone::Green)));
    let read = inputs(&dir);
    let last = read.last().map(String::as_str).unwrap_or_default();
    let json: serde_json::Value = serde_json::from_str(last).unwrap_or_default();
    for field in [
        "session_id",
        "session_name",
        "project_dir",
        "agent",
        "model",
        "busy",
        "status",
        "context_percent",
        "session_input_tokens",
        "session_output_tokens",
        "session_cost_usd",
        "turn",
    ] {
        assert!(json.get(field).is_some(), "{field} missing from {last}");
    }
    assert_eq!(json.get("status"), Some(&serde_json::json!("idle")));
    assert_eq!(json.get("turn"), Some(&serde_json::json!(1)));

    // Ten changes within 100 ms: one run now, one after the pace.
    run_for(&mut app, &wakes, Duration::from_millis(500));
    let before = inputs(&dir).len();
    let started = Instant::now();
    for _ in 0..10 {
        spend(&mut app);
    }
    assert!(started.elapsed() < Duration::from_millis(100));
    run_for(&mut app, &wakes, Duration::from_millis(1200));
    let runs = inputs(&dir).len() - before;
    assert!((1..=2).contains(&runs), "{runs} runs");

    // A script that hangs: killed after 2 s, the last line kept, marked.
    assert!(fs::write(dir.join("mode"), "sleep").is_ok());
    let started = Instant::now();
    spend(&mut app);
    run_until(&mut app, &wakes, |world| {
        text(world).ends_with(STALE_MARKER)
    });
    let waited = started.elapsed();
    let marked = format!("M:test/model T:1 {STALE_MARKER}");
    assert_eq!(item(app.world()), Some((marked.clone(), Tone::Dim)));
    assert!(waited >= Duration::from_millis(1900), "{waited:?}");
    assert!(waited < Duration::from_secs(4), "{waited:?}");

    // A failing script: its error said once, the last line kept.
    assert!(fs::write(dir.join("mode"), "fail").is_ok());
    run_for(&mut app, &wakes, Duration::from_millis(400));
    spend(&mut app);
    run_until(&mut app, &wakes, |world| failures(world) > 0);
    run_for(&mut app, &wakes, Duration::from_millis(400));
    let before = inputs(&dir).len();
    spend(&mut app);
    run_until(&mut app, &wakes, |_| inputs(&dir).len() > before);
    run_for(&mut app, &wakes, Duration::from_millis(400));
    assert_eq!(failures(app.world()), 1);
    assert_eq!(text(app.world()), marked);

    // Another command in settings.toml: the next update uses it.
    assert!(fs::remove_file(dir.join("mode")).is_ok());
    let other = dir.join("other.sh");
    write_script(&other, "#!/bin/sh\ncat >/dev/null\necho other\n");
    let setting = format!("{SECTION} = '{}'\n", other.display());
    assert!(fs::write(home.settings(), setting).is_ok());
    run_until(&mut app, &wakes, |world| text(world) == "other");
    assert_eq!(text(app.world()), "other");
    fs::remove_dir_all(&dir).ok();
}
