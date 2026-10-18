use bevy_app::prelude::*;
use rig_harness::pick::PickPlugin;
use rig_harness::prelude::Invoked;
use rig_harness::rig_core::conversation::Conversation;
use rig_harness::rig_core::message::{
    AssistantContent, AssistantMessage, Message, ToolCall, ToolFunction, ToolName,
};
use rig_harness::view::protocol::{EventRecord, Seq, SessionId};
use rig_harness::view::{MirrorPlugin, ViewLink, mirror};
use rig_tui::{TuiPlugin, TuiTestScreen, add_view};

use bevy_ecs::prelude::*;
use rig_harness::rig_core::test_utils::MockCompletionModel;
use rig_harness::rig_ecs::prelude::Agent;
use rig_harness::rig_ecs::tasks::Wake;
use rig_harness::view::ViewTarget;
use rig_harness_test_support::{connect, reply};

use super::BasicsViewPlugin;

#[test]
fn a_mirrored_shell_call_is_drawn_with_the_shell_look() {
    let mut app = App::new();
    app.set_error_handler(rig_harness::error::panic)
        .insert_resource(Invoked::default())
        .add_plugins((PickPlugin, MirrorPlugin, TuiPlugin, BasicsViewPlugin));
    add_view(&mut app);
    let (link, client, _, _) = ViewLink::detached();
    app.insert_resource(link)
        .insert_resource(client)
        .insert_resource(TuiTestScreen::new(60, 12));
    let Ok(name) = ToolName::new("shell") else {
        return;
    };
    let args = serde_json::json!({ "command": "ls -la" });
    let call = ToolCall::from_wire("c1", ToolFunction::new(name.clone(), args));
    let mut conversation = Conversation::default();
    conversation.push(Message::user("what is here?"));
    let reply = AssistantMessage::new(vec![AssistantContent::ToolCall(call.clone())]);
    conversation.push(Message::Assistant(reply));
    conversation.push(Message::tool_result(call.id.clone(), name, "Cargo.toml"));
    let records = conversation.take_unsaved().into_iter().zip(1..);
    let records = records.map(|(record, seq)| EventRecord {
        session: SessionId("s-1".into()),
        seq: Seq(seq),
        at_ms: 0,
        source: None,
        kind: "conversation.change".into(),
        data: serde_json::json!({ "agentId": "a-1", "log": "a-1", "index": seq, "record": record }),
    });
    mirror::apply(app.world_mut(), records);
    app.update();
    let rows = app.world().resource::<TuiTestScreen>().rows();
    assert!(rows.iter().any(|row| row.contains("$ ls -la")), "{rows:#?}");
}

/// Not a test: one process of a whole stack on a scripted model, for a
/// person or an agent to drive by hand at a terminal. Run as `RIG_ROLE=server`
/// (rigd, whose workers are this same test as `worker`), or `view` (the
/// terminal view, which never starts rigd itself), each with `RIG_HOME`:
///
/// ```text
/// RIG_HOME=$H RIG_ROLE=server $TESTS --exact view::tests::stack_role --ignored &
/// RIG_HOME=$H RIG_ROLE=view $TESTS --exact view::tests::stack_role --ignored
/// ```
#[test]
#[ignore = "a process of a hand-driven stack, not a test"]
fn stack_role() {
    let role = std::env::var(rig_harness::harness_protocol::env::ROLE).unwrap_or_default();
    let this = std::env::current_exe().unwrap_or_default();
    let args = [
        "--exact",
        "view::tests::stack_role",
        "--ignored",
        "--nocapture",
    ];
    match role.as_str() {
        "server" => {
            let home = rig_harness::harness_protocol::Home::from_env();
            let Ok((lock, token, listener)) = rig_harness::server::bind(&home) else {
                return;
            };
            let options = rig_harness::server::Options {
                idle_exit: std::time::Duration::from_secs(600),
                worker: Some((this, args.iter().map(Into::into).collect())),
                ..rig_harness::server::Options::default()
            };
            let mut app = rig_harness::server::server_app();
            rig_harness::server::serve(&mut app, home, token, listener, options);
            app.run();
            drop(lock);
        }
        "worker" => {
            let turns =
                (1..=50).map(|n| reply(&format!("Reply {n}: hello from the scripted agent.")));
            let model = MockCompletionModel::from_stream_turns(turns);
            // A worker as `worker_app` makes it, on no command line: this
            // test's own words are no agent options.
            let mut app = App::new();
            app.insert_resource(Invoked::default())
                .set_error_handler(rig_harness::worker::fail)
                .add_plugins((
                    rig_harness::HeadlessPlugins,
                    rig_harness::RigHarnessPlugins,
                    rig_harness::worker::WorkerPlugin,
                ));
            app.add_plugins((
                crate::activity::ActivityPlugin,
                crate::coding_tools::ShellTool,
            ));
            app.add_systems(
                PostStartup,
                move |agents: Query<Entity, With<Agent>>, mut commands: Commands| {
                    for agent in &agents {
                        if let Some(connection) = connect(&model) {
                            commands.entity(agent).insert(connection);
                        }
                    }
                },
            );
            app.run();
        }
        "view" => {
            let mut app = rig_harness::view_app();
            app.add_plugins((rig_tui::TuiPlugin, super::BasicsViewPlugin));
            let target = ViewTarget {
                autostart: false,
                ..ViewTarget::from_env()
            };
            let wake = app
                .world()
                .get_resource::<Wake>()
                .cloned()
                .unwrap_or_default();
            let (link, client) = ViewLink::connect(target, wake);
            app.insert_resource(link).insert_resource(client);
            app.run();
        }
        _ => {}
    }
}
