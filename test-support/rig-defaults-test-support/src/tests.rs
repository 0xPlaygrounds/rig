use rig_harness::prelude::*;

use super::*;

/// Data two systems write.
#[derive(Component)]
struct Tally;

fn first_writer(_: Query<&mut Tally>) {}

fn second_writer(_: Query<&mut Tally>) {}

#[test]
fn unordered_writers_in_one_stage_fail_by_name() {
    assert_stages_ordered(|_: &mut App| {});
    let unordered = |app: &mut App| {
        app.add_systems(AgentPass, (first_writer, second_writer).in_set(Stage::Gate));
    };
    let failure = std::panic::catch_unwind(|| assert_stages_ordered(unordered)).err();
    let report = failure
        .and_then(|payload| payload.downcast::<String>().ok())
        .unwrap_or_default();
    assert!(
        report.contains("first_writer") && report.contains("second_writer"),
        "{report}"
    );
}

/// The slash commands of a harness with the default plugins and `more`,
/// and whether Enter in the transcript does something (rewind).
fn commands_with(more: &[fn(&mut App)]) -> (Vec<String>, bool) {
    use rig_tui::crossterm::event::KeyCode;
    use rig_tui::{KeyBinding, KeyPress, KeyScope, Keys};
    let home = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("agent-commands");
    let mut app = App::new();
    app.insert_resource(RigHome(Home::at(home)))
        .insert_resource(Invoked {
            args: Invocation::default(),
            terminal: false,
        })
        .add_plugins((HeadlessPlugins, RigHarnessPlugins));
    add_defaults(&mut app);
    for add in more {
        add(&mut app);
    }
    // The session plugin sends panics to the session's log.
    drop(std::panic::take_hook());
    app.finish();
    app.cleanup();
    let world = app.world_mut();
    let mut names: Vec<String> = (world.query_filtered::<&Name, With<SlashCommand>>())
        .iter(world)
        .map(ToString::to_string)
        .collect();
    names.sort();
    let enter = Keys::from(KeyPress::key(KeyCode::Enter));
    let rewinds = (world.query::<&KeyBinding>())
        .iter(world)
        .any(|binding| binding.scope == KeyScope::Transcript && binding.keys == enter);
    (names, rewinds)
}

/// `/fork` and rewind come only with the forks plugin, which is off by
/// default; `/agents` comes once with it, with subagents, or with both;
/// `/resume` stays without them.
#[test]
fn agent_commands_come_with_the_plugins_that_make_agents() {
    use rig_basics::sessions::agents::AgentsPlugin;
    use rig_basics::subagents::SubagentsPlugin;
    let forks: fn(&mut App) = |app| rig_harness::load::<AgentsPlugin>(app, "rig-basics", "");
    let subagents: fn(&mut App) = |app| rig_harness::load::<SubagentsPlugin>(app, "rig-basics", "");
    let count = |names: &[String], name: &str| names.iter().filter(|it| *it == name).count();
    for (more, fork, agents) in [
        (&[][..], 0, 0),
        (&[subagents][..], 0, 1),
        (&[forks][..], 1, 1),
        (&[forks, subagents][..], 1, 1),
        (&[subagents, forks][..], 1, 1),
    ] {
        let (names, rewinds) = commands_with(more);
        let had = (count(&names, "/fork"), count(&names, "/agents"), rewinds);
        assert_eq!(had, (fork, agents, fork == 1), "{names:?}");
        assert_eq!(count(&names, "/resume"), 1, "{names:?}");
    }
}
