//! Every default plugin can be left out: the app the generated `main.rs`
//! builds from the default `plugins.toml`, less any one of its plugins,
//! answers a `--print` prompt with a tool call on a scripted model, and so
//! does the app with every disabled entry of the default list on top
//! ([`OPTIONAL`]). `/reload` is there exactly when
//! `rig_basics::reload::ReloadPlugin` is. The plugin `rig plugin new`
//! writes comes last, listed twice and added once. No case leaves systems
//! of one agent-loop stage that write the same data unordered (the debug
//! build's ambiguity warning). Each case is a child process of this test
//! with its own `RIG_HOME`, as a run is.

#[path = "../../../src/launcher/plugin/scaffold.rs"]
mod scaffold;

use std::path::Path;
use std::process::{Command, Stdio};

use bevy_ecs::schedule::{ScheduleBuildWarning, ScheduleBuilt, ScheduleLabel};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_defaults_test_support::{DEFAULTS, plugins};
use rig_ecs::session_log::SessionRestored;
use rig_harness::harness_protocol::Invocation;
use rig_harness::load;
use rig_harness::prelude::*;
use rig_harness_test_support::connect;

/// The plugin a child process leaves out.
const LEAVE_OUT: &str = "RIG_TEST_LEAVE_OUT";

/// The default `plugins.toml`'s disabled entries, in its order, but
/// rig-steel's, whose steel-core build its own tests pay for.
const OPTIONAL: [(&str, fn(&mut App)); 7] = plugins![
    rig_basics::approval::ApprovalPlugin,
    rig_basics::preview::PreviewPlugin,
    rig_basics::checkpoint::CheckpointPlugin,
    rig_inspect::InspectPlugin,
    rig_basics::subagents::SubagentsPlugin,
    rig_basics::telemetry::EffectLogPlugin,
    rig_basics::telemetry::DiagnosticsPlugin,
];

/// The case that adds every [`OPTIONAL`] plugin to the defaults.
const WITH_EVERY: &str = "with every optional plugin";

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

/// Fails the run when the agent loop holds unordered systems writing the
/// same data in one stage.
fn stages_ordered(built: On<ScheduleBuilt>) {
    if built.label != AgentPass.intern() {
        return;
    }
    let unordered: Vec<String> = built
        .build_metadata
        .warnings
        .iter()
        .filter(|warning| matches!(warning, ScheduleBuildWarning::Ambiguity(_)))
        .map(ToString::to_string)
        .collect();
    assert!(unordered.is_empty(), "{unordered:?}");
}

/// Sends the prompt as `--print` does, when it is left out.
fn ask(
    restored: Option<Res<SessionRestored>>,
    agents: MainAgentQuery,
    plugins: Query<&Name, With<PluginSource>>,
    mut asked: Local<bool>,
    mut commands: Commands,
) {
    let Some(agent) = main_agent(&agents).filter(|_| restored.is_some() && !*asked) else {
        return;
    };
    *asked = true;
    if !plugins
        .iter()
        .any(|name| name.as_str() == "rig_basics::print::PrintPlugin")
    {
        commands.trigger(Deliver::new(agent, PROMPT, DeliveryMode::Steer));
    }
}

fn answered(
    settled: On<Add<Settled>>,
    turns: Query<&TurnOutcome>,
    provided: Query<(&Name, &ProvidedBy)>,
    (plugins, names): (Query<&PluginSource>, Query<&Name, With<PluginSource>>),
    slash_commands: Query<&Name, With<SlashCommand>>,
    mut exits: MessageWriter<AppExit>,
) {
    let Ok(outcome) = turns.get(settled.entity) else {
        return;
    };
    let answered = matches!(outcome,
        TurnOutcome::Answered(message) if message.final_answer().as_deref() == Some("read"));
    let scaffold = |plugin: &PluginSource| plugin.package == "rig-hello";
    let added_once = plugins.iter().filter(|plugin| scaffold(plugin)).count() == 1;
    let command = |(name, by): (&Name, &ProvidedBy)| {
        name.as_str() == "/__name__" && plugins.get(by.0).is_ok_and(scaffold)
    };
    let reload = names
        .iter()
        .any(|name| name.as_str() == "rig_basics::reload::ReloadPlugin");
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
                    ..Invocation::default()
                },
                terminal: false,
            })
            .add_plugins((HeadlessPlugins, RigHarnessPlugins));
        for (_, add) in DEFAULTS.iter().filter(|(name, _)| *name != left_out) {
            add(&mut app);
        }
        if left_out == WITH_EVERY {
            for (_, add) in OPTIONAL {
                add(&mut app);
            }
        }
        for _ in 0..2 {
            load::<scaffold::ScaffoldPlugin>(&mut app, "rig-hello", "path plugins/hello");
        }
        app.add_systems(
            Update,
            (ask, give_up.run_if(on_real_timer(Duration::from_secs(30)))),
        )
        .add_observer(connect_first)
        .add_observer(stages_ordered)
        .add_observer(answered);
        assert_eq!(app.run(), AppExit::Success, "without {left_out}");
        return;
    }
    let template = include_str!("../../../src/launcher/plugins.toml");
    let listed = |prefix: &str| -> Vec<&str> {
        template
            .lines()
            .filter_map(|line| line.strip_prefix(prefix)?.strip_suffix('"'))
            .filter(|plugin| *plugin != "rig_steel::SteelPlugin")
            .collect()
    };
    let names: Vec<&str> = DEFAULTS.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        names,
        listed("plugin = \""),
        "the default plugins.toml, in its order"
    );
    let optional: Vec<&str> = OPTIONAL.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        optional,
        listed("# plugin = \""),
        "its disabled entries, in its order"
    );
    let homes = Path::new(env!("CARGO_TARGET_TMPDIR")).join("removal");
    std::fs::remove_dir_all(&homes).ok();
    let test = std::env::current_exe().unwrap_or_default();
    let failed: Vec<&str> = ["nothing", WITH_EVERY]
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
