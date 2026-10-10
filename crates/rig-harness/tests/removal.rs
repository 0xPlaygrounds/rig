//! Every default plugin can be left out: the app the generated `main.rs`
//! builds from the default `plugins.toml`, less any one of its plugins,
//! answers a turn with a tool call on a scripted model, and so does the
//! app with the optional `rig-inspect` plugin on top. The plugin `rig
//! plugin new` writes comes last, listed twice and added once. Each case
//! is a child process of this test with its own `RIG_HOME`, as a run is.
#![cfg(feature = "tui")]

#[path = "../../../src/launcher/plugin/scaffold.rs"]
mod scaffold;

use std::path::Path;
use std::process::{Command, Stdio};

use rig::harness_protocol::Invocation;
use rig_core::catalog::Catalog;
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::effects::Handler;
use rig_ecs::journal::SessionLog;
use rig_harness::load;
use rig_harness::prelude::*;

/// The plugin a child process leaves out.
const LEAVE_OUT: &str = "RIG_TEST_LEAVE_OUT";

/// The default `plugins.toml`'s plugins, in its order.
macro_rules! defaults {
    ($($plugin:path),* $(,)?) => {
        [$((stringify!($plugin), |app: &mut App| load::<$plugin>(app, "rig-harness", ""))),*]
    };
}

const DEFAULTS: [(&str, fn(&mut App)); 20] = defaults![
    rig_harness::plugins::project_context::ProjectContextPlugin,
    rig_harness::plugins::models::ModelsPlugin,
    rig_harness::plugins::login_chatgpt::ChatgptLoginPlugin,
    rig_harness::plugins::defaults::DefaultsPlugin,
    rig_harness::plugins::sessions::SessionsPlugin,
    rig_harness::plugins::compaction::CompactionPlugin,
    rig_harness::plugins::usage::UsagePlugin,
    rig_harness::plugins::activity::ActivityPlugin,
    rig_harness::plugins::effect_log::EffectLogPlugin,
    rig_harness::plugins::tools::ReadTool,
    rig_harness::plugins::tools::EditTool,
    rig_harness::plugins::tools::WriteTool,
    rig_harness::plugins::tools::SearchTool,
    rig_harness::plugins::tools::ShellTool,
    rig_harness::plugins::reload_tool::ReloadTool,
    rig_harness::plugins::basics::BasicCommandsPlugin,
    rig_harness::plugins::subagents::SubagentsPlugin,
    rig_harness::plugins::diagnostics::DiagnosticsPlugin,
    rig_harness::plugins::print::PrintPlugin,
    rig_harness::tui::TuiPlugin,
];

/// The case that adds the optional `rig-inspect` plugin to the defaults.
const WITH_INSPECT: &str = "with rig_inspect::InspectPlugin";

/// Exits with 2 when the turn takes over 30 s.
fn give_up(mut exits: MessageWriter<AppExit>) {
    exits.write(AppExit::from_code(2));
}

/// As a front does: connects the user's agent to a scripted model that
/// reads this crate's manifest and then answers, and sends the prompt.
fn ask(log: Res<SessionLog>, agents: PrimaryQuery, mut asked: Local<bool>, mut commands: Commands) {
    let Some(agent) = primary(&agents).filter(|_| log.is_live() && !*asked) else {
        return;
    };
    *asked = true;
    let read = serde_json::json!({ "path": format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR")) });
    let end = MockStreamEvent::final_response_with_default_usage;
    let model = MockCompletionModel::from_stream_turns([
        vec![MockStreamEvent::tool_call("call-1", "read", read), end()],
        vec![MockStreamEvent::text("read"), end()],
    ]);
    if let Ok(spec) = Catalog::builtin().resolve("deepseek/deepseek-flash") {
        let spec = spec.shared();
        let handler = ModelAdapter::<Completion>::new(spec.reference(), model);
        let handler = Handler(ErasedHandler::new(handler));
        commands.entity(agent).insert(Connection { spec, handler });
    }
    let prompt = "Read it.".to_owned();
    send_input(&mut commands, agent, prompt, DeliveryMode::Steer);
}

fn answered(
    ended: On<TurnEnded>,
    provided: Query<(&Name, &ProvidedBy)>,
    plugins: Query<&PluginSource>,
    mut exits: MessageWriter<AppExit>,
) {
    let answered = matches!(&ended.outcome,
        TurnOutcome::Answered(message) if final_answer(message).as_deref() == Some("read"));
    let scaffold = |plugin: &PluginSource| plugin.krate == "rig-hello";
    let added_once = plugins.iter().filter(|plugin| scaffold(plugin)).count() == 1;
    let command = |(name, by): (&Name, &ProvidedBy)| {
        name.as_str() == "/__name__" && plugins.get(by.0).is_ok_and(scaffold)
    };
    let ok = answered && added_once && provided.iter().any(command);
    exits.write(AppExit::from_code(u8::from(!ok)));
}

#[test]
fn every_default_plugin_can_be_left_out() {
    if let Ok(left_out) = std::env::var(LEAVE_OUT) {
        let mut app = App::new();
        // Any failing system or observer fails the run; `--print` keeps
        // the terminal view away, and this test is the front.
        app.set_error_handler(rig_harness::error::panic)
            .insert_resource(RunMode(Invocation {
                print: Some(String::new()),
                model: None,
            }))
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
        app.insert_resource(Front("test".to_owned()))
            .add_systems(
                Update,
                (ask, give_up.run_if(on_real_timer(Duration::from_secs(30)))),
            )
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
