use bevy_app::TaskPoolPlugin;
use bevy_ecs::message::Messages;
use rig_ecs::AgentPlugin;
use rig_ecs::prelude::*;
use rig_harness_protocol::Invocation;

use super::*;
use crate::ask::AskPlugin;
use crate::commands::{AppCommandsExt, CommandArgs};
use crate::host::cli::{AppCliExt, Front};
use crate::host::launcher::Invoked;

/// The command lines run, with the agent each ran for, and what each
/// picker reported.
#[derive(Resource, Default)]
struct Seen {
    ran: Vec<(Entity, String)>,
    picked: Vec<Option<Chosen>>,
}

/// An app with the kernel, questions and pickers, a `note` command, and
/// its first agent, with no front.
fn app() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins((
        TaskPoolPlugin::default(),
        AgentPlugin,
        AskPlugin,
        PickPlugin,
    ))
    .init_resource::<Seen>()
    .add_command(
        "note",
        "Notes its arguments",
        |In(args): In<CommandArgs>, mut seen: ResMut<Seen>| {
            seen.ran.push((args.agent, args.args));
        },
    )
    .add_observer(|picked: On<Picked>, mut seen: ResMut<Seen>| {
        seen.picked.push(picked.chosen.clone());
    });
    app.update();
    app.update();
    let agent = app.world().resource::<ShownAgent>().0;
    (app, agent.unwrap_or(Entity::PLACEHOLDER))
}

#[test]
fn a_pick_runs_the_items_command_for_the_agent_shown_and_closes_the_picker() {
    let (mut app, agent) = app();
    let world = app.world_mut();
    assert!(
        world.get::<Agent>(agent).is_some(),
        "the primary agent is shown"
    );
    let size = Picker::new("Size")
        .item("small", "note small")
        .item("large", "note large");
    let picker = world.spawn(size).id();
    let closed = world.spawn(Picker::new("Other").item("x", "note x")).id();
    world.trigger(Pick::person(picker, Some(Chosen::item(1))));
    world.trigger(Pick::person(closed, None));
    app.update();
    let world = app.world();
    assert!(world.get_entity(picker).is_err() && world.get_entity(closed).is_err());
    let seen = world.resource::<Seen>();
    assert_eq!(seen.ran, [(agent, "large".to_owned())]);
    assert_eq!(seen.picked, [Some(Chosen::item(1)), None]);
}

#[test]
fn a_pick_of_several_items_runs_their_commands_in_list_order() {
    let (mut app, agent) = app();
    let world = app.world_mut();
    let files = Picker::new("Delete which files?")
        .multiple()
        .item("a.tmp", "note a.tmp")
        .checked(true)
        .item("b.log", "note b.log")
        .item("c.tmp", "note c.tmp");
    let picker = world.spawn(files).id();
    world.trigger(Pick::person(picker, Some(Chosen::items([2, 0]))));
    app.update();
    let seen = app.world().resource::<Seen>();
    let ran = [(agent, "a.tmp".to_owned()), (agent, "c.tmp".to_owned())];
    assert_eq!(seen.ran, ran);
    assert_eq!(seen.picked, [Some(Chosen::items([2, 0]))]);
}

#[test]
fn a_pick_answers_a_question_with_its_choices_and_a_note() {
    let (mut app, agent) = app();
    let world = app.world_mut();
    let text = world
        .spawn((Ask::yes_no("Why?").free_text(), ForAgent(agent)))
        .id();
    let noted = world
        .spawn((Ask::yes_no("Deploy?").free_text(), ForAgent(agent)))
        .id();
    let choice = world.spawn((Ask::yes_no("Run?"), ForAgent(agent))).id();
    let both = world.spawn((Ask::yes_no("Both?"), ForAgent(agent))).id();
    let closed = world.spawn((Ask::yes_no("Again?"), ForAgent(agent))).id();
    let several = Ask::yes_no("Which?").choice("maybe", "Maybe").multiple();
    let several = world.spawn((several, ForAgent(agent))).id();
    world.trigger(Pick::person(text, Some(Chosen::text("because"))));
    let note = Chosen::item(1).with_text("not on a Friday");
    world.trigger(Pick::person(noted, Some(note)));
    world.trigger(Pick::person(choice, Some(Chosen::item(1))));
    // A question that takes one choice is not answered with two.
    world.trigger(Pick::person(both, Some(Chosen::items([0, 1]))));
    world.trigger(Pick::person(closed, None));
    world.trigger(Pick::person(several, Some(Chosen::items([2, 0]))));
    app.update();
    let world = app.world();
    let answer = |ask: Entity| world.get::<Outcome>(ask).map(Answer::of);
    assert_eq!(
        [text, noted, choice, both, closed, several].map(answer),
        [
            Some(Some(Answer::text("because"))),
            Some(Some(Answer::choice("no").with_text("not on a Friday"))),
            Some(Some(Answer::choice("no"))),
            None,
            Some(None),
            Some(Some(Answer::choices(["yes", "maybe"]))),
        ]
    );
}

/// Why each line was refused, in the order refused.
fn refusals(app: &mut App) -> Vec<String> {
    let mut recalled = app.world_mut().resource_mut::<Messages<Recalled>>();
    recalled
        .drain()
        .filter_map(|recalled| recalled.why)
        .collect()
}

/// The app of [`app`] run by the front `front`, chosen for a run on a
/// terminal or not, with a `/size` command that opens a picker.
fn run_by(front: Front, terminal: bool) -> (App, Entity) {
    let (mut app, agent) = app();
    app.insert_resource(Invoked {
        args: Invocation::default(),
        terminal,
    })
    .add_front(front)
    .add_command(
        "size",
        "Picks a size",
        |In(args): In<CommandArgs>, mut commands: Commands| {
            let picker = Picker::new("Size").for_agent(args.agent);
            commands.spawn(
                picker
                    .item("small", "note small")
                    .item("large", "note large"),
            );
        },
    );
    let chosen = crate::host::cli::choose(app.world_mut());
    assert!(chosen.is_some());
    (app, agent)
}

#[test]
fn a_picker_the_front_cannot_show_refuses_its_command_saying_what_to_type() {
    let (mut app, agent) = run_by(Front::new("print").headless(), false);
    app.world_mut().trigger(RunCommand::new(agent, "size"));
    app.update();
    let why = "Size: rig cannot show a list to pick from here; type one of: /note small, \
               /note large";
    assert_eq!(refusals(&mut app), [why]);
    let world = app.world_mut();
    assert_eq!(world.query::<&Picker>().iter(world).count(), 0);
    assert_eq!(world.resource::<Seen>().picked, [None]);
    // A picker a plugin opens by itself, and one a plugin shows, stay.
    world.spawn(Picker::new("Status").item("Skip", "note skip"));
    world.spawn(ShowsPickers);
    world.trigger(RunCommand::new(agent, "size"));
    app.update();
    assert!(refusals(&mut app).is_empty());
    let world = app.world_mut();
    assert_eq!(world.query::<&Picker>().iter(world).count(), 2);

    let shows = Front::new("tui").on_terminal().shows_pickers();
    let (mut app, agent) = run_by(shows, true);
    app.world_mut().trigger(RunCommand::new(agent, "size"));
    app.update();
    assert!(refusals(&mut app).is_empty());
    let world = app.world_mut();
    assert_eq!(world.query::<&Picker>().iter(world).count(), 1);
}

#[test]
fn a_refused_picker_lists_ten_commands_and_never_asks_for_a_secret_here() {
    let models = (0..12).fold(Picker::new("Model"), |picker, n| {
        picker.item(format!("model {n}"), format!("model m{n}"))
    });
    let note = unshown(&models);
    assert!(
        note.contains("/model m0, ") && note.contains("/model m9, …"),
        "{note}"
    );
    assert!(!note.contains("/model m10"), "{note}");
    let reading = unshown(&Picker::new("Resume a session"));
    assert!(
        reading.contains("give the command what to pick"),
        "{reading}"
    );
    let key = unshown(&Picker::new("OpenAI API key").secret());
    assert!(key.contains("Type it in `rig` in a terminal"), "{key}");
}

#[test]
fn a_picker_a_command_opens_after_a_nested_command_ran_is_refused_too() {
    let (mut app, agent) = run_by(Front::new("print").headless(), false);
    app.add_command(
        "outer",
        "Runs /note, then opens a picker",
        |In(args): In<CommandArgs>, mut commands: Commands| {
            commands.queue(move |world: &mut World| {
                world.trigger(RunCommand::new(args.agent, "note first"));
                world.flush();
                world.spawn(Picker::new("Size").item("small", "note small"));
            });
        },
    );
    app.world_mut().trigger(RunCommand::new(agent, "outer"));
    app.update();
    let refused = refusals(&mut app);
    assert_eq!(refused.len(), 1, "{refused:?}");
    let world = app.world_mut();
    assert_eq!(world.query::<&Picker>().iter(world).count(), 0);
}
