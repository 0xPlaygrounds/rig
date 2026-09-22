//! ChatGPT subscription authentication and Responses-format dialect configuration.
//!
//! Use an exchanged access token from the environment, or sign in through [`auth`]
//! and pass the resolved token to [`OpenAI::with_key`](crate::providers::openai::OpenAI::with_key).
//!
//! ```no_run
//! use rig_core::providers::chatgpt;
//! use rig_core::providers::openai::OpenAI;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAI::from_env_with(&chatgpt::DIALECT)?.completion(chatgpt::GPT_5_3_CODEX);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

pub mod auth;

use crate::providers::openai::responses_api::SystemInstructionsPlacement;
use crate::providers::openai::wire::{
    Dialect, Identity, OutputCap, Quirks, ResponsesContract, ResponsesQuirks, Route,
};

const CHATGPT_API_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const DEFAULT_ORIGINATOR: &str = "rig";
const DEFAULT_INSTRUCTIONS: &str = "You are ChatGPT, a helpful AI assistant.";

/// `gpt-5.4`
pub const GPT_5_4: &str = "gpt-5.4";
/// `gpt-5.4-pro`
pub const GPT_5_4_PRO: &str = "gpt-5.4-pro";
/// `gpt-5.3-codex`
pub const GPT_5_3_CODEX: &str = "gpt-5.3-codex";
/// `gpt-5.3-codex-spark`
pub const GPT_5_3_CODEX_SPARK: &str = "gpt-5.3-codex-spark";
/// `gpt-5.3-instant`
pub const GPT_5_3_INSTANT: &str = "gpt-5.3-instant";
/// `gpt-5.3-chat-latest`
pub const GPT_5_3_CHAT_LATEST: &str = "gpt-5.3-chat-latest";

/// Stable descriptor name reported on normalized ChatGPT responses.
pub const PROVIDER_NAME: &str = "chatgpt";

/// Subscription-backend dialect with SSE replies, Codex parameters, and caller identity.
/// All system messages become top-level instructions; replayed frames omit envelope metadata.
pub const DIALECT: Dialect = Dialect {
    base_url_env: Some("CHATGPT_API_BASE"),
    request_id_header: Some("x-request-id"),
    quirks: Quirks {
        completion_route: Route::Responses,
        output_cap: OutputCap::OpenAiReasoningFamilies,
        base_url_env_alias: Some("OPENAI_CHATGPT_API_BASE"),
        account_id_env: Some("CHATGPT_ACCOUNT_ID"),
        default_instructions: Some(DEFAULT_INSTRUCTIONS),
        instructions_env: Some("CHATGPT_DEFAULT_INSTRUCTIONS"),
        identity: Some(Identity {
            originator: DEFAULT_ORIGINATOR,
            originator_env: "CHATGPT_ORIGINATOR",
            user_agent_env: "CHATGPT_USER_AGENT",
            session_ids: true,
        }),
        responses: ResponsesQuirks {
            system_instructions: SystemInstructionsPlacement::AllInstructions,
            contract: ResponsesContract::Codex,
            ..ResponsesQuirks::openai()
        },
        ..Quirks::openai()
    },
    ..Dialect::gateway(PROVIDER_NAME, CHATGPT_API_BASE_URL, "CHATGPT_ACCESS_TOKEN")
};

/// Generate a per-request session correlator for transport headers only.
pub(crate) fn session_id() -> String {
    crate::id::generate()
}
