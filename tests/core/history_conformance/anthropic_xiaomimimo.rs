//! The Xiaomi MiMo dialect of the Messages wire's history suite. V2 Flash
//! and Pro read no images.

use rig_core::providers::anthropic::wire::XIAOMIMIMO;

use super::anthropic::MessagesHistory;

pub const XIAOMIMIMO_MESSAGES_HISTORY: MessagesHistory = MessagesHistory {
    dialect: &XIAOMIMIMO,
    model: rig_core::providers::xiaomimimo::MIMO_V2_5,
    other_model: rig_core::providers::xiaomimimo::MIMO_V2_OMNI,
    text_only_model: Some(rig_core::providers::xiaomimimo::MIMO_V2_FLASH),
    signature: "sig_1",
    hosted: false,
};

rig_history_conformance::history_conformance_suite! {
    wire: "anthropic_xiaomimimo",
    fixture: XIAOMIMIMO_MESSAGES_HISTORY,
}
