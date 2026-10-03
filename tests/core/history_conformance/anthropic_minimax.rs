//! The MiniMax dialect of the Messages wire's history suite. M2 models
//! read no images.

use rig_core::providers::anthropic::wire::MINIMAX;

use super::anthropic::MessagesHistory;

pub const MINIMAX_MESSAGES_HISTORY: MessagesHistory = MessagesHistory {
    dialect: &MINIMAX,
    model: "MiniMax-M3",
    other_model: rig_core::providers::minimax::MINIMAX_M2_7,
    text_only_model: Some(rig_core::providers::minimax::MINIMAX_M2_7),
    signature: "sig_1",
    hosted: false,
};

rig_history_conformance::history_conformance_suite! {
    wire: "anthropic_minimax",
    fixture: MINIMAX_MESSAGES_HISTORY,
}
