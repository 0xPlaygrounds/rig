use rig_harness::commands::{AppCommandsExt, CommandArgs, RunCommand};
use rig_harness::prelude::*;

use super::BasicCommandsPlugin;

#[test]
fn retry_is_an_unknown_command() {
    let mut app = App::new();
    app.add_plugins((AgentPlugin, BasicCommandsPlugin));
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
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
fn every_help_starts_at_one_column_after_the_widest_name() {
    let mut app = App::new();
    app.add_plugins((AgentPlugin, BasicCommandsPlugin))
        .add_command("claude-cloak", "Cloak", |_: In<CommandArgs>| {})
        .add_command("skill:diagnose-crash", "Diagnose", |_: In<CommandArgs>| {})
        .add_command("日本語", "Wide", |_: In<CommandArgs>| {});
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
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
            .all(|column| *column == Some("/skill:diagnose-crash ".len())),
        "{text:?}"
    );
}
