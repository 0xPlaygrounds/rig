//! The Z.AI dialect of the Messages wire's history suite. GLM vision
//! models put `v` after the version; the rest read no images.

use rig_core::providers::anthropic::wire::ZAI;

use super::anthropic::MessagesHistory;

pub const ZAI_MESSAGES_HISTORY: MessagesHistory = MessagesHistory {
    dialect: &ZAI,
    model: rig_core::providers::zai::GLM_4_5V,
    other_model: rig_core::providers::zai::GLM_4_6,
    text_only_model: Some(rig_core::providers::zai::GLM_4_6),
    signature: "sig_1",
    hosted: false,
};

rig_history_conformance::history_conformance_suite! {
    wire: "anthropic_zai",
    fixture: ZAI_MESSAGES_HISTORY,
}
