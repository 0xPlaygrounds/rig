//! Synthetic transcript tests cover identities that cannot be requested reliably
//! from a live provider. Adapter request-boundary tests cover their use.

use super::*;
use crate::completion::{Accepts, ReplayTarget};
use crate::message::{Api, ToolCall, ToolFunction, ToolResult};

#[derive(Debug)]
struct Target;

impl ReplayTarget for Target {
    fn api(&self) -> Api {
        Api::from_static("test.api")
    }
    fn provider(&self) -> &str {
        "test"
    }
    fn model(&self) -> &str {
        "model"
    }
    fn accepts(&self, _model: &str) -> Accepts {
        Accepts::ALL
    }
}

fn call(id: CallId) -> Message {
    Message::Assistant(crate::message::AssistantMessage::new(vec![
        AssistantContent::ToolCall(ToolCall::new(
            id,
            ToolFunction::new(
                crate::message::ToolName::new("test").expect("tool name"),
                serde_json::json!({}),
            ),
        )),
    ]))
}

fn result(id: CallId) -> Message {
    Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            is_error: false,
            call: id,
            name: crate::message::ToolName::new("possibly_repaired").expect("tool name"),
            content: vec![crate::message::ToolResultContent::text("")],
        })],
    }
}

#[test]
fn a_rig_issued_id_is_one_alias_that_no_provider_id_takes() {
    let issued = CallId::from_wire("");
    let later = CallId::from_wire("");
    let history = vec![
        call(CallId::from_wire("tool-0")),
        call(issued.clone()),
        result(issued.clone()),
        call(later.clone()),
    ];
    let ids = WireIds::for_target(&history, &Target, "model");
    assert_eq!(ids.of(&CallId::from_wire("tool-0")), Some("tool-0"));
    assert_eq!(ids.of(&issued), Some("tool-1"));
    assert_eq!(ids.of(&later), Some("tool-2"));
}
