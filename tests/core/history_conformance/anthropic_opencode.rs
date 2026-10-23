//! The OpenCode Zen dialect of the Messages wire's history suite, on the
//! Claude models it relays.

use rig_core::providers::anthropic::wire::OPENCODE;

use super::anthropic::MessagesHistory;

pub const OPENCODE_MESSAGES_HISTORY: MessagesHistory = MessagesHistory {
    dialect: &OPENCODE,
    model: "claude-haiku-5-5",
    other_model: "claude-sonnet-5-5",
    text_only_model: None,
    signature: "sig_1",
    hosted: false,
};

rig_history_conformance::history_conformance_suite! {
    wire: "anthropic_opencode",
    fixture: OPENCODE_MESSAGES_HISTORY,
}
