use rig_harness::commands::{AppCommandsExt, CommandArgs, RunCommand};
use rig_harness::prelude::*;

use super::BasicCommandsPlugin;

/// The `Workspace` methods of a test workspace that removes, renames, lists
/// and runs nothing, and takes each path as canonical.
macro_rules! workspace_stubs {
    () => {
        fn remove(&self, _: &::std::path::Path) -> ::std::io::Result<()> {
            Err(::std::io::ErrorKind::Unsupported.into())
        }

        fn rename(&self, _: &::std::path::Path, _: &::std::path::Path) -> ::std::io::Result<()> {
            Err(::std::io::ErrorKind::Unsupported.into())
        }

        fn canonical(&self, path: &::std::path::Path) -> ::std::path::PathBuf {
            path.to_path_buf()
        }

        fn files(
            &self,
            _: &::std::path::Path,
            _: Option<&str>,
            _: ::rig_tools::workspace::Listing,
        ) -> ::std::io::Result<::rig_tools::workspace::Listed> {
            Ok(::rig_tools::workspace::Listed::default())
        }

        fn spawn(
            &self,
            _: &::rig_tools::process::Launcher,
            _: Option<&::std::path::Path>,
            _: &::std::ffi::OsStr,
            _: &[::std::ffi::OsString],
            _: ::rig_tools::process::Spawn,
        ) -> ::std::io::Result<::rig_tools::process::Process> {
            Err(::std::io::ErrorKind::Unsupported.into())
        }
    };
}
pub(crate) use workspace_stubs;

#[test]
fn retry_is_an_unknown_command() {
    let mut app = App::new();
    app.add_plugins((AgentPlugin, BasicCommandsPlugin));
    let world = app.world_mut();
    let agent = spawn_root(world, ()).id();
    world.trigger(RunCommand::new(agent, "retry".to_owned()));
    world.flush();
    let why: Vec<Option<&str>> = app
        .world()
        .resource::<Messages<Recalled>>()
        .iter_current_update_messages()
        .map(|recalled| recalled.why.as_deref())
        .collect();
    assert_eq!(why, [Some("Unknown command /retry.")]);
}

#[test]
fn every_help_starts_two_columns_after_the_widest_name() {
    let mut app = App::new();
    app.add_plugins((AgentPlugin, BasicCommandsPlugin))
        .add_command("claude-cloak", "Cloak", |_: In<CommandArgs>| {})
        .add_command("skill:diagnose-crash", "Diagnose", |_: In<CommandArgs>| {})
        .add_command("日本語", "Wide", |_: In<CommandArgs>| {});
    let world = app.world_mut();
    let agent = spawn_root(world, ()).id();
    world.trigger(RunCommand::new(agent, "help".to_owned()));
    world.flush();
    let mut notices = app.world_mut().resource_mut::<Messages<Notice>>();
    let text: Vec<String> = notices.drain().map(|notice| notice.text).collect();
    // Columns on the screen: a wide name's characters take two each.
    let columns: Vec<Option<usize>> = text
        .iter()
        .flat_map(|text| text.lines())
        .map(|line| {
            ["Cloak", "Diagnose", "Wide", "List the commands", "Quit"]
                .iter()
                .find_map(|help| line.get(..line.find(help)?))
                .map(unicode_width::UnicodeWidthStr::width)
        })
        .collect();
    assert_eq!(columns.len(), 5, "{text:?}");
    assert!(
        columns
            .iter()
            .all(|column| *column == Some("/skill:diagnose-crash  ".len())),
        "{text:?}"
    );
}

#[test]
fn a_session_switch_refused_while_a_turn_runs_comes_back_in_the_input() {
    let home = rig_harness_test_support::scratch_home("rig-basics-switch-mid-turn");
    let mut app = App::new();
    app.insert_resource(RigHome(home.clone()))
        .add_plugins((AgentPlugin, crate::sessions::SessionsPlugin));
    let world = app.world_mut();
    let mut conversation = Conversation::default();
    conversation.push(rig_core::completion::Message::user("Keep this session"));
    let agent = spawn_root(world, conversation).id();
    world.spawn(TurnOf(agent));
    for line in ["new", "resume 1-1"] {
        world.trigger(RunCommand::new(agent, line));
        world.flush();
        let recalled: Vec<Recalled> = world.resource_mut::<Messages<Recalled>>().drain().collect();
        assert_eq!(recalled.len(), 1, "{line}");
        assert!(
            recalled.iter().all(|recalled| {
                recalled.agent == agent
                    && recalled.text == format!("/{line}")
                    && recalled
                        .why
                        .as_ref()
                        .is_some_and(|why| why.contains("A turn is running"))
            }),
            "{line}: {recalled:?}"
        );
    }
    std::fs::remove_dir_all(home.root()).ok();
}

#[test]
fn a_command_refused_while_a_turn_runs_comes_back_in_the_input() {
    let mut app = App::new();
    app.add_plugins((
        AgentPlugin,
        crate::models::ModelsPlugin,
        crate::compaction::CompactionPlugin,
    ));
    let world = app.world_mut();
    let agent = spawn_root(world, ()).id();
    world.spawn(TurnOf(agent));
    for line in ["model ollama/llama3", "compact"] {
        world.trigger(RunCommand::new(agent, line));
        world.flush();
        let recalled: Vec<Recalled> = world.resource_mut::<Messages<Recalled>>().drain().collect();
        assert_eq!(recalled.len(), 1, "{line}");
        assert!(
            recalled.iter().all(|recalled| {
                recalled.agent == agent
                    && recalled.text == format!("/{line}")
                    && recalled
                        .why
                        .as_ref()
                        .is_some_and(|why| why.contains("A turn is running"))
            }),
            "{line}: {recalled:?}"
        );
    }
}
