//! The Moonshot dialect of the Messages wire's history suite. Kimi sends
//! its thinking unsigned, so this is where a current unsigned thinking
//! item must replay verbatim (#1315), and Kimi K2 before K2.5 reads no
//! images.

use rig_core::providers::anthropic::wire::MOONSHOT;
use rig_core::providers::moonshot::{KIMI_K2_6, KIMI_K3};

use super::anthropic::MessagesHistory;

pub const MOONSHOT_HISTORY: MessagesHistory = MessagesHistory {
    dialect: &MOONSHOT,
    model: KIMI_K2_6,
    other_model: KIMI_K3,
    text_only_model: Some("kimi-k2-thinking"),
    signature: "",
    hosted: false,
};

rig_history_conformance::history_conformance_suite! {
    wire: "anthropic_moonshot",
    fixture: MOONSHOT_HISTORY,
}
