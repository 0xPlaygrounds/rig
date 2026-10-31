//! ChatGPT subscription authentication and Responses-format dialect configuration.
//!
//! [`from_env`] reads an exchanged access token from the environment. To sign
//! in through [`auth`], serve a catalog model with an [`auth::Authenticator`]
//! as a [`SignedInModel`](crate::client::signed_in::SignedInModel).
//!
//! ```no_run
//! use rig_core::providers::chatgpt;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let model = chatgpt::from_env()?.completion(chatgpt::DEFAULT_MODEL);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

pub mod auth;
pub mod extension;

use crate::providers::openai::responses_api::SystemInstructionsPlacement;
use crate::providers::openai::wire::{
    Dialect, Identity, OutputCap, Quirks, ResponsesContract, ResponsesQuirks, Route,
};

const CHATGPT_API_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const DEFAULT_ORIGINATOR: &str = "rig";
const DEFAULT_INSTRUCTIONS: &str = "You are ChatGPT, a helpful AI assistant.";

/// Stable descriptor name reported on normalized ChatGPT responses.
pub const PROVIDER_NAME: &str = "chatgpt";

/// The name users know the subscription plan by.
pub const PLAN_TITLE: &str = "ChatGPT";

/// The model a fresh sign-in switches to: the plan's latest frontier model.
pub const DEFAULT_MODEL: &str = crate::providers::openai::GPT_6_1_SOL;

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
        // The subscription backend keeps no files, so no file id resolves
        // there.
        accepts_file_ids: false,
        // ChatGPT bills a subscription, not the catalog's API rates.
        catalog_priced: false,
        identity: Some(Identity {
            originator: DEFAULT_ORIGINATOR,
            originator_env: "CHATGPT_ORIGINATOR",
            user_agent_env: "CHATGPT_USER_AGENT",
        }),
        // The backend routes its prompt cache by this header, over
        // `prompt_cache_key`.
        session_header: Some("session-id"),
        responses: ResponsesQuirks {
            system_instructions: SystemInstructionsPlacement::AllInstructions,
            contract: ResponsesContract::Codex,
            ..ResponsesQuirks::openai()
        },
        ..Quirks::openai()
    },
    ..Dialect::gateway(PROVIDER_NAME, CHATGPT_API_BASE_URL, "CHATGPT_ACCESS_TOKEN")
};

crate::client::macros::openai_vendor!(DIALECT, "ChatGPT");

/// The dialect's environment settings with the signed-in credential,
/// account, and endpoint taking precedence.
pub(crate) fn signed_in_config(
    context: &auth::AuthContext,
) -> Result<crate::providers::openai::OpenAIConfig, crate::client::env::EnvError> {
    let mut config = crate::providers::openai::OpenAIConfig::from_env_with_credential(
        &DIALECT,
        context.api_key.expose().to_owned(),
        DIALECT.quirks.auth,
    )?;
    if let Some(account_id) = &context.account_id {
        config = config.with_account_id(account_id);
    }
    if let Some(base) = &context.api_base {
        config = config.with_base_url(base);
    }
    Ok(config)
}
