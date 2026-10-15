use bevy_app::TaskPoolPlugin;
use rig_core::completion::Message;
use rig_core::message::UserContent;
use rig_ecs::AgentPlugin;
use rig_ecs::prelude::*;

use super::*;

/// The arguments `/echo` ran with, and how each line was sent.
#[derive(Resource, Default)]
struct Echoed(Vec<(String, DeliveryMode)>);

/// An app with the kernel, an `/echo` command and an agent.
fn app() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin))
        .init_resource::<Echoed>()
        .add_command(
            "echo",
            "Echoes",
            |In(args): In<CommandArgs>, mut echoed: ResMut<Echoed>| {
                echoed.0.push((args.args, args.mode));
            },
        );
    let agent = app.world_mut().spawn(Agent).id();
    app.update();
    (app, agent)
}

/// The text of the user messages in `agent`'s conversation.
fn sent(app: &App, agent: Entity) -> Vec<String> {
    let messages = app
        .world()
        .get::<Conversation>(agent)
        .map(|conversation| conversation.messages().to_vec())
        .unwrap_or_default();
    messages
        .iter()
        .flat_map(|message| match &**message {
            Message::User { content } => content.to_vec(),
            _ => Vec::new(),
        })
        .filter_map(|content| match content {
            UserContent::Text(text) => Some(text.text),
            _ => None,
        })
        .collect()
}

#[test]
fn a_users_slash_line_runs_as_a_command_unless_taken_over_sent_literally_or_not_the_users() {
    let (mut app, agent) = app();
    // Prepare edits come first: this one makes a line a command.
    let edit = |mut typed: Query<&mut Incoming, Added<Admitted>>| {
        for mut typed in &mut typed {
            if typed.text() == "hi" {
                *typed.text_mut() = "/echo hello".to_owned();
            }
        }
    };
    // A Gate system ordered before the commands takes its own lines over.
    let take_over = |typed: Query<(Entity, &Incoming), ReadyToStart>, mut commands: Commands| {
        for (message, typed) in &typed {
            if typed.text().starts_with("/secret") {
                commands
                    .entity(message)
                    .insert_if_new(Outcome::custom("mine"));
            }
        }
    };
    app.add_systems(
        AgentPass,
        (
            edit.in_set(Stage::Prepare),
            take_over.in_set(Stage::Gate).before(CommandSystems),
        ),
    );
    let note = |text: &str| Deliver::new(agent, text, DeliveryMode::Inform);
    for typed in [
        note("hi"),
        Deliver::new(agent, "/echo typed", DeliveryMode::Queue),
        note("/secret x"),
        note("/echo sent").literal(),
        note("/echo reported").with_author(Author::plugin("p")),
    ] {
        app.world_mut().trigger(typed);
    }
    app.update();
    let echoed = &app.world().resource::<Echoed>().0;
    assert_eq!(
        echoed.as_slice(),
        [
            ("hello".to_owned(), DeliveryMode::Inform),
            ("typed".to_owned(), DeliveryMode::Queue)
        ]
    );
    assert_eq!(
        sent(&app, agent),
        [
            "/echo sent",
            "[Output of plugin p, not the user's words]\n/echo reported"
        ]
    );
}

#[test]
fn a_disabled_command_gives_way_to_its_replacement() {
    let (mut app, agent) = app();
    let say = |text: &'static str| {
        move |In(args): In<CommandArgs>, mut notices: MessageWriter<Notice>| {
            notices.write(Notice::info(args.agent, text));
        }
    };
    app.add_command("hi", "Says hi", say("old"));
    let world = app.world_mut();
    let mut names = world.query::<(Entity, &Name)>();
    let old = names
        .iter(world)
        .find(|(_, name)| name.as_str() == "/hi")
        .map(|(entity, _)| entity);
    assert!(old.is_some(), "the command is an entity named /hi");
    if let Some(old) = old {
        world.entity_mut(old).insert(Disabled);
    }
    app.add_command("hi", "Says hi", say("new"));
    app.world_mut().trigger(RunCommand::new(agent, "hi"));
    app.world_mut().flush();
    let notices = app.world().resource::<Messages<Notice>>();
    let said: Vec<&str> = notices
        .iter_current_update_messages()
        .map(|notice| notice.text.as_str())
        .collect();
    assert_eq!(said, ["new"]);
}
