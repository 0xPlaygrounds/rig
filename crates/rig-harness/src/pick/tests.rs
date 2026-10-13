use bevy_app::TaskPoolPlugin;
use rig_ecs::AgentPlugin;
use rig_ecs::prelude::*;

use super::*;
use crate::ask::AskPlugin;
use crate::commands::{AppCommandsExt, CommandArgs};

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
    let agent = app.world().resource::<Shown>().0;
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
    world.trigger(Pick::person(picker, Some(Chosen::Item(1))));
    world.trigger(Pick::person(closed, None));
    app.update();
    let world = app.world();
    assert!(world.get_entity(picker).is_err() && world.get_entity(closed).is_err());
    let seen = world.resource::<Seen>();
    assert_eq!(seen.ran, [(agent, "large".to_owned())]);
    assert_eq!(seen.picked, [Some(Chosen::Item(1)), None]);
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
    world.trigger(Pick::person(picker, Some(Chosen::Items(vec![2, 0]))));
    app.update();
    let seen = app.world().resource::<Seen>();
    let ran = [(agent, "a.tmp".to_owned()), (agent, "c.tmp".to_owned())];
    assert_eq!(seen.ran, ran);
    assert_eq!(seen.picked, [Some(Chosen::Items(vec![2, 0]))]);
}

#[test]
fn a_pick_answers_a_question() {
    let (mut app, agent) = app();
    let world = app.world_mut();
    let text = world
        .spawn((Ask::yes_no("Why?").free_text(), ForAgent(agent)))
        .id();
    let choice = world.spawn((Ask::yes_no("Run?"), ForAgent(agent))).id();
    let closed = world.spawn((Ask::yes_no("Again?"), ForAgent(agent))).id();
    let several = Ask::yes_no("Which?").choice("maybe", "Maybe").multiple();
    let several = world.spawn((several, ForAgent(agent))).id();
    world.trigger(Pick::person(text, Some(Chosen::Text("because".to_owned()))));
    world.trigger(Pick::person(choice, Some(Chosen::Item(1))));
    world.trigger(Pick::person(closed, None));
    world.trigger(Pick::person(several, Some(Chosen::Items(vec![2, 0]))));
    app.update();
    let world = app.world();
    let answer = |ask: Entity| world.get::<Outcome>(ask).map(Answer::of);
    assert_eq!(
        [
            answer(text),
            answer(choice),
            answer(closed),
            answer(several)
        ],
        [
            Some(Some(Answer::Text("because".to_owned()))),
            Some(Some(Answer::Choice("no".to_owned()))),
            Some(None),
            Some(Some(Answer::Choices(vec![
                "yes".to_owned(),
                "maybe".to_owned()
            ]))),
        ]
    );
}
