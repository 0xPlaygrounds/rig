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
