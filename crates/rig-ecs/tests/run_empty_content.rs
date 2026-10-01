//! A run whose history holds an empty message or an empty tool result fails
//! as invalid content before any completion is dispatched.

use rig_core::message::{AssistantContent, CallId, ToolName, ToolResult, UserContent};
use rig_ecs::agent::content::parts::ContentError;
use rig_ecs::agent::{Failed, Failure, MessageParts};
use rig_ecs::systems::RunCommands;

use crate::run_support::*;

fn empty_histories() -> Vec<MessageParts> {
    vec![
        MessageParts::User {
            content: Vec::<UserContent>::new(),
        },
        MessageParts::Assistant {
            id: None,
            content: Vec::<AssistantContent>::new(),
        },
        MessageParts::User {
            content: vec![UserContent::ToolResult(ToolResult {
                call: CallId::from_wire("call_1"),
                name: ToolName::new("lookup").expect("tool name"),
                content: Vec::new(),
            })],
        },
    ]
}

#[test]
fn an_empty_message_in_history_fails_the_run_before_dispatch() {
    for history in empty_histories() {
        let mut app = app();
        let (handler, requests) = Capturing::new("model", "ok");
        let model = register(&mut app, "model", handler);
        let agent = spawn_agent(app.world_mut(), "test", model);
        let run =
            app.world_mut()
                .spawn_run(agent, std::slice::from_ref(&history), "go", false, None);
        tick_until(&mut app, "the run fails", |world| {
            world.get::<Failed>(run).is_some()
        });
        let failed = app.world().get::<Failed>(run).expect("failed");
        assert!(
            matches!(failed.0, Failure::Content(ContentError::Shape)),
            "{history:?}: {:?}",
            failed.0
        );
        assert!(requests.lock().unwrap().is_empty(), "{history:?}");
    }
}

#[test]
fn message_parts_refuse_an_empty_list() {
    assert_eq!(MessageParts::user(Vec::new()), Err(ContentError::Shape));
    assert_eq!(
        MessageParts::assistant(None, Vec::new()),
        Err(ContentError::Shape)
    );
}
