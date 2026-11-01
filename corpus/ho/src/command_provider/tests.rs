use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::mpsc::Receiver;

use rig_harness::harness_protocol::SessionId;
use rig_harness::services::ServicesPlugin;
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, first_agent, log_ended, run_until, scratch_home,
};

use super::*;

/// The stand-in program: saves each request beside itself, then calls
/// `read` when the user asked for a tool and no result came back yet, and
/// otherwise answers "hel", "lo".
const ECHO: &str = r#"#!/bin/sh
dir=$(dirname "$0")
n=$(ls "$dir" | grep -c '^stdin-')
IFS= read -r request
printf '%s\n' "$request" > "$dir/stdin-$((n + 1)).json"
case "$request" in
  *tool_result*) ;;
  *'"use tool"'*)
    echo '{"tool_call":{"id":"t1","name":"read","arguments":{}}}'
    echo '{"done":true}'
    exit 0 ;;
esac
echo '{"text":"hel"}'
echo '{"text":"lo"}'
echo '{"usage":{"input_tokens":3,"output_tokens":2}}'
echo '{"done":true}'
"#;

/// A program that answers with a line that is not JSON.
const NOPE: &str = "#!/bin/sh\nread -r request\necho nope\n";

/// A program that fails, saying why on stderr.
const DIE: &str = "#!/bin/sh\necho 'no GPU found' >&2\nexit 3\n";

/// A program that asks to be tried again.
const BUSY: &str = "#!/bin/sh\nexit 75\n";

/// Whether a notice so far contains `text`.
fn noticed(world: &World, text: &str) -> bool {
    let notices = &world.resource::<Collected<Notice>>().0;
    notices.iter().any(|notice| notice.text.contains(text))
}

fn script(dir: &Path, name: &str, text: &str) {
    let path = dir.join(name);
    fs::write(&path, text).ok();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).ok();
}

/// `read`, answering with the file's made-up contents.
fn read(called: On<ToolCalled<Value>>, mut commands: Commands) {
    let read = Outcome::from(ToolResult::success("the notes say 42".into()));
    commands.entity(called.call).insert_if_new(read);
}

/// Sends `text` to `agent` and runs until its turn ended; how it ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) -> Option<TurnOutcome> {
    let before = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > before
    });
    app.world().resource::<Ended>().0.last().cloned()
}

fn answer(ended: Option<TurnOutcome>) -> String {
    match ended {
        Some(TurnOutcome::Answered(message)) => message.final_answer().unwrap_or_default(),
        other => format!("{other:?}"),
    }
}

/// A persistent program: answers "kept" with its pid, except its second
/// request, which it answers only once cancelled.
const KEPT: &str = r#"#!/bin/sh
dir=$(dirname "$0")
n=0
while IFS= read -r line; do
  case "$line" in
    *'"cancel"'*) echo "$line" >> "$dir/cancels"; echo '{"error":"cancelled"}' ;;
    *) n=$((n + 1))
       if [ "$n" -eq 2 ]; then echo '{"text":"waiting"}'
       else echo "{\"text\":\"kept $$\"}"; echo '{"done":true}'; fi ;;
  esac
done
"#;

/// A program that starts answering, then hangs in place of its shell.
const HANG: &str = r#"#!/bin/sh
echo $$ > "$(dirname "$0")/hang.pid"
echo '{"text":"waiting"}'
exec sleep 30
"#;

/// The app with the kernel, the services and the plugin reading the models
/// `config` names (`{dir}` is the scripts' directory), and its agent.
fn start(name: &str, scripts: &[(&str, &str)], config: &str) -> (App, Receiver<()>, Entity, Home) {
    let home = scratch_home(name);
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    for (name, text) in scripts {
        script(home.root(), name, text);
    }
    let config = config.replace("{dir}", &home.root().display().to_string());
    fs::write(home.root().join(FILE), config).ok();
    app.add_plugins((
        ServicesPlugin,
        AgentPlugin,
        SessionLogPlugin,
        CommandProviderPlugin,
    ))
    .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
    .init_resource::<Ended>()
    .add_observer(log_ended);
    collect::<StreamDelta>(&mut app);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    (app, wakes, agent, home)
}

fn use_model(app: &mut App, agent: Entity, model: &str) {
    let model = model.to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
}

/// Sends `text` to `agent`, interrupts its turn once the program started
/// answering, and runs until the turn ended.
fn interrupt(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let before = app.world().resource::<Collected<StreamDelta>>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Collected<StreamDelta>>().0.len() > before
    });
    app.world_mut().trigger(StopTurn::new(agent));
    app.update();
}

#[test]
fn a_program_answers_calls_tools_and_fails() {
    let config = "[[model]]\nid = \"cmd/echo\"\ncommand = [\"{dir}/echo.sh\"]\ntools = true\n\n\
                  [[model]]\nid = \"cmd/nope\"\ncommand = [\"{dir}/nope\"]\n\n\
                  [[model]]\nid = \"cmd/die\"\ncommand = [\"{dir}/die\"]\n\n\
                  [[model]]\nid = \"cmd/busy\"\ncommand = [\"{dir}/busy\"]\n";
    let scripts = [
        ("echo.sh", ECHO),
        ("nope", NOPE),
        ("die", DIE),
        ("busy", BUSY),
    ];
    let (mut app, wakes, agent, home) = start("ho-command-provider", &scripts, config);
    use_model(&mut app, agent, "cmd/echo");

    // Text streams as the program writes it.
    assert_eq!(answer(ask(&mut app, &wakes, agent, "say hi")), "hello");
    let deltas: Vec<&str> = app
        .world()
        .resource::<Collected<StreamDelta>>()
        .0
        .iter()
        .filter_map(|delta| match &delta.event {
            StreamEvent::Text { text, .. } | StreamEvent::Reasoning { text, .. } => {
                Some(text.as_str())
            }
            StreamEvent::Arguments { json, .. } => Some(json.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, ["hel", "lo"]);

    // The tool call runs, and the next request carries its result.
    assert_eq!(answer(ask(&mut app, &wakes, agent, "use tool")), "hello");
    let third = fs::read_to_string(home.root().join("stdin-3.json")).unwrap_or_default();
    assert!(third.contains(r#""type":"tool_result""#), "{third}");
    assert!(third.contains("the notes say 42"), "{third}");

    // A line that is not JSON fails the turn, naming the line.
    use_model(&mut app, agent, "cmd/nope");
    let ended = ask(&mut app, &wakes, agent, "say hi");
    assert!(matches!(ended, Some(TurnOutcome::Failed(_))), "{ended:?}");
    assert!(noticed(app.world(), "malformed line 1: nope"));

    // An exit before `done` fails it with the end of stderr.
    use_model(&mut app, agent, "cmd/die");
    let ended = ask(&mut app, &wakes, agent, "say hi");
    assert!(matches!(ended, Some(TurnOutcome::Failed(_))), "{ended:?}");
    assert!(noticed(app.world(), "code 3 before `done`: no GPU found"));

    // Exit code 75 is retried by the kernel.
    use_model(&mut app, agent, "cmd/busy");
    app.world_mut()
        .trigger(Deliver::new(agent, "say hi", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| noticed(world, "Retrying in"));
    assert!(noticed(app.world(), "asked to be tried again"));
    app.world_mut().trigger(StopTurn::new(agent));
    app.update();
}

#[test]
fn an_interrupt_kills_the_program_or_cancels_its_request() {
    let config = "[[model]]\nid = \"cmd/hang\"\ncommand = [\"{dir}/hang\"]\n\n\
                  [[model]]\nid = \"cmd/kept\"\ncommand = [\"{dir}/kept\"]\npersistent = true\n";
    let scripts = [("hang", HANG), ("kept", KEPT)];
    let (mut app, wakes, agent, home) = start("ho-command-provider-interrupt", &scripts, config);

    // A program per request is killed.
    use_model(&mut app, agent, "cmd/hang");
    interrupt(&mut app, &wakes, agent, "say hi");
    let pid = fs::read_to_string(home.root().join("hang.pid")).unwrap_or_default();
    let proc = Path::new("/proc").join(pid.trim());
    run_until(&mut app, &wakes, |_| {
        fs::read_to_string(proc.join("stat")).map_or(true, |stat| stat.contains(") Z "))
    });
    let stat = fs::read_to_string(proc.join("stat")).unwrap_or_default();
    assert!(
        stat.is_empty() || stat.contains(") Z "),
        "still running: {stat}"
    );

    // A persistent program is told to cancel, and serves the next request.
    use_model(&mut app, agent, "cmd/kept");
    let first = answer(ask(&mut app, &wakes, agent, "one"));
    interrupt(&mut app, &wakes, agent, "wait");
    let cancels = home.root().join("cancels");
    run_until(&mut app, &wakes, |_| cancels.exists());
    let cancelled = fs::read_to_string(&cancels).unwrap_or_default();
    assert!(cancelled.starts_with(r#"{"cancel":"#), "{cancelled}");
    let second = answer(ask(&mut app, &wakes, agent, "two"));
    assert!(first.starts_with("kept "), "{first}");
    assert_eq!(first, second, "one process serves both");
}
