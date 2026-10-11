//! Every default plugin can be left out: the app the generated `main.rs`
//! builds from the default `plugins.toml`, less any one of its plugins,
//! answers a `--print` prompt with a tool call on a scripted model, and so
//! does the app with the optional `rig-inspect` plugin on top. `/reload`
//! is there exactly when `rig-reload` is. The plugin `rig plugin new`
//! writes comes last, listed twice and added once. Each case is a child
//! process of this test with its own `RIG_HOME`, as a run is.

#[path = "../../../src/launcher/plugin/scaffold.rs"]
mod scaffold;

use std::path::Path;
use std::process::{Command, Stdio};

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::journal::SessionRestored;
use rig_harness::harness_protocol::Invocation;
use rig_harness::load;
use rig_harness::prelude::*;
use rig_harness_test_support::connect;

/// The plugin a child process leaves out.
const LEAVE_OUT: &str = "RIG_TEST_LEAVE_OUT";

/// The default `plugins.toml`'s plugins, in its order, each loaded from
/// its crate as the generated `main.rs` loads it.
macro_rules! defaults {
    ($($plugin:path),* $(,)?) => {
        [$((stringify!($plugin), |app: &mut App| {
            let krate = stringify!($plugin).split("::").next().unwrap_or_default();
            load::<$plugin>(app, &krate.replace('_', "-"), "")
        })),*]
    };
}

const DEFAULTS: [(&str, fn(&mut App)); 21] = defaults![
    rig_basics::ProjectContextPlugin,
    rig_models::ModelsPlugin,
    rig_login_chatgpt::ChatgptLoginPlugin,
    rig_models::DefaultsPlugin,
    rig_sessions::SessionsPlugin,
    rig_compaction::CompactionPlugin,
    rig_telemetry::UsagePlugin,
    rig_activity::ActivityPlugin,
    rig_telemetry::EffectLogPlugin,
    rig_coding_tools::ReadTool,
    rig_coding_tools::EditTool,
    rig_coding_tools::WriteTool,
    rig_coding_tools::SearchTool,
    rig_coding_tools::ShellTool,
    rig_coding_tools::AttachPlugin,
    rig_reload::ReloadPlugin,
    rig_basics::BasicCommandsPlugin,
    rig_subagents::SubagentsPlugin,
    rig_telemetry::DiagnosticsPlugin,
    rig_print::PrintPlugin,
    rig_tui::TuiPlugin,
];

/// The case that adds the optional `rig-inspect` plugin to the defaults.
const WITH_INSPECT: &str = "with rig_inspect::InspectPlugin";

/// Exits with 2 when the turn takes over 30 s.
fn give_up(mut exits: MessageWriter<AppExit>) {
    exits.write(AppExit::from_code(2));
}

/// The `--print` prompt.
const PROMPT: &str = "Read it.";

/// Connects the first agent to a scripted model that reads this crate's
/// manifest and then answers.
fn connect_first(added: On<Add<Agent>>, mut commands: Commands) {
    let read = serde_json::json!({ "path": format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR")) });
    let end = MockStreamEvent::final_response_with_default_usage;
    let model = MockCompletionModel::from_stream_turns([
        vec![MockStreamEvent::tool_call("call-1", "read", read), end()],
        vec![MockStreamEvent::text("read"), end()],
    ]);
    if let Some(connection) = connect(&model) {
        let choice = ModelChoice(connection.spec.reference());
        commands.entity(added.entity).insert((connection, choice));
    }
}

/// Sends the prompt as `--print` does, when it is left out.
fn ask(
    restored: Option<Res<SessionRestored>>,
    agents: PrimaryQuery,
    plugins: Query<&PluginSource>,
    mut asked: Local<bool>,
    mut commands: Commands,
) {
    let Some(agent) = primary(&agents).filter(|_| restored.is_some() && !*asked) else {
        return;
    };
    *asked = true;
    if !plugins.iter().any(|plugin| plugin.krate == "rig-print") {
        commands.trigger(Deliver::new(agent, PROMPT, DeliveryMode::Steer));
    }
}

fn answered(
    ended: On<TurnEnded>,
    provided: Query<(&Name, &ProvidedBy)>,
    plugins: Query<&PluginSource>,
    slash_commands: Query<&Name, With<SlashCommand>>,
    mut exits: MessageWriter<AppExit>,
) {
    let answered = matches!(&ended.outcome,
        TurnOutcome::Answered(message) if final_answer(message).as_deref() == Some("read"));
    let scaffold = |plugin: &PluginSource| plugin.krate == "rig-hello";
    let added_once = plugins.iter().filter(|plugin| scaffold(plugin)).count() == 1;
    let command = |(name, by): (&Name, &ProvidedBy)| {
        name.as_str() == "/__name__" && plugins.get(by.0).is_ok_and(scaffold)
    };
    let reload = plugins.iter().any(|plugin| plugin.krate == "rig-reload");
    let reloads = slash_commands.iter().any(|name| name.as_str() == "/reload");
    let ok = answered && added_once && provided.iter().any(command) && reload == reloads;
    exits.write(AppExit::from_code(u8::from(!ok)));
}

#[test]
fn every_default_plugin_can_be_left_out() {
    if let Ok(left_out) = std::env::var(LEAVE_OUT) {
        let mut app = App::new();
        // Any failing system or observer fails the run; `--print` keeps
        // the terminal view away.
        app.set_error_handler(rig_harness::error::panic)
            .insert_resource(Invoked {
                args: Invocation {
                    print: Some(PROMPT.to_owned()),
                    model: None,
                },
                terminal: false,
            })
            .add_plugins((HeadlessPlugins, RigHarnessPlugins));
        for (_, add) in DEFAULTS.iter().filter(|(name, _)| *name != left_out) {
            add(&mut app);
        }
        if left_out == WITH_INSPECT {
            load::<rig_inspect::InspectPlugin>(&mut app, "rig-inspect", "version 0.44.0");
        }
        for _ in 0..2 {
            load::<scaffold::ScaffoldPlugin>(&mut app, "rig-hello", "path plugins/hello");
        }
        app.add_systems(
            Update,
            (ask, give_up.run_if(on_real_timer(Duration::from_secs(30)))),
        )
        .add_observer(connect_first)
        .add_observer(answered);
        assert_eq!(app.run(), AppExit::Success, "without {left_out}");
        return;
    }
    let listed: Vec<&str> = include_str!("../../../src/launcher/plugins.toml")
        .lines()
        .filter_map(|line| line.strip_prefix("plugin = \"")?.strip_suffix('"'))
        .collect();
    let names: Vec<&str> = DEFAULTS.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, listed, "the default plugins.toml, in its order");
    let homes = Path::new(env!("CARGO_TARGET_TMPDIR")).join("removal");
    std::fs::remove_dir_all(&homes).ok();
    let test = std::env::current_exe().unwrap_or_default();
    let failed: Vec<&str> = ["nothing", WITH_INSPECT]
        .into_iter()
        .chain(names)
        .filter(|left_out| {
            let home = homes.join(left_out.rsplit("::").next().unwrap_or(left_out));
            let status = Command::new(&test)
                .args(["every_default_plugin_can_be_left_out", "--exact"])
                .env("RIG_HOME", home)
                .env(LEAVE_OUT, left_out)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .status();
            !status.is_ok_and(|status| status.success())
        })
        .collect();
    assert!(
        failed.is_empty(),
        "runs without {failed:?} failed; logs in {homes:?}"
    );
}
