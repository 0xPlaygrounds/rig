//! OpenCode Go's Messages route, which names a session on every request.

use rig_core::providers::anthropic::wire::OPENCODE_GO;

use super::anthropic::MessagesHistory;

pub const OPENCODE_GO_MESSAGES_HISTORY: MessagesHistory = MessagesHistory {
    dialect: &OPENCODE_GO,
    model: "claude-haiku-5-5",
    other_model: "qwen3.8-max",
    text_only_model: None,
    signature: "sig_1",
    hosted: false,
};

rig_history_conformance::history_conformance_suite! {
    wire: "anthropic_opencode-go",
    fixture: OPENCODE_GO_MESSAGES_HISTORY,
}
