//! Official Sign in with ChatGPT inference using a caller-managed OAuth token.
//!
//! Obtain and refresh the token through OpenAI's [open-source sign-in flow](https://developers.openai.com/siwc/token-sharing-open-source/sign-in),
//! including ID-token verification and the `chatgpt.tokens.use.direct` grant.
//! Requests use HTTP/SSE and group local function/custom tools in the `rig` namespace.
//! Tool choice accepts `auto`, `none`, or `required`.
//!
//! ```no_run
//! use rig_core::providers::chatgpt::siwc;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let model = siwc::from_env()?.completion("gpt-6.1-sol");
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

use crate::error::EncodeError;
use crate::providers::openai::responses_api::SystemInstructionsPlacement;
use crate::providers::openai::wire::{
    Dialect, DialectHooks, Quirks, ResponsesContract, ResponsesQuirks, Route,
};
use serde_json::{Map, Value, json};

/// The ChatGPT plan Responses endpoint. Reads `CHATGPT_SIWC_ACCESS_TOKEN`,
/// authorized for `https://api.openai.com/v1`, with no endpoint environment override.
pub const DIALECT: Dialect = Dialect {
    request_id_header: Some("x-request-id"),
    quirks: Quirks {
        hooks: Some(&HOOKS),
        completion_route: Route::Responses,
        accepts_file_ids: false,
        catalog_priced: false,
        responses: ResponsesQuirks {
            system_instructions: SystemInstructionsPlacement::AllInstructions,
            system_role: "developer",
            contract: ResponsesContract::Siwc,
            ..ResponsesQuirks::openai()
        },
        ..Quirks::openai()
    },
    ..Dialect::gateway(
        "chatgpt-siwc",
        "https://api.openai.com/v1",
        "CHATGPT_SIWC_ACCESS_TOKEN",
    )
};

crate::client::macros::openai_vendor!(DIALECT, "Sign in with ChatGPT");

static HOOKS: DialectHooks = DialectHooks {
    default_endpoint: None,
    model_route: None,
    completion_envelope: None,
    modality_envelope: None,
    model_listing: Some(catalogue),
};

fn catalogue(event: Value) -> crate::model::ModelList {
    let models = event
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|model| model.get("visibility").and_then(Value::as_str) == Some("list"))
        .filter_map(|model| {
            let mut info = crate::model::ModelInfo::from_id(model.get("slug")?.as_str()?);
            info.name = model
                .get("display_name")
                .and_then(Value::as_str)
                .map(str::to_owned);
            Some(info)
        })
        .collect();
    crate::model::ModelList::new(models)
}

pub(crate) fn rewrite(body: &mut Map<String, Value>) -> Result<(), EncodeError> {
    for field in [
        "background",
        "conversation",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "moderation",
        "multi_agent",
        "prompt",
        "prompt_cache_retention",
        "safety_identifier",
        "temperature",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
        "previous_response_id",
    ] {
        crate::json_utils::remove_preserving_order(body, field);
    }
    body.insert("store".into(), json!(false));
    body.insert("stream".into(), json!(true));
    if body
        .get("input")
        .and_then(Value::as_array)
        .is_some_and(|items| {
            items
                .iter()
                .any(|item| item.get("role").and_then(Value::as_str) == Some("system"))
        })
    {
        return Err(EncodeError::request(
            "SIWC input must use developer messages, not system messages",
        ));
    }
    let Some(tools) = body.get_mut("tools") else {
        return Ok(());
    };
    let tools = tools
        .as_array_mut()
        .ok_or_else(|| EncodeError::request("SIWC tools must be an array"))?;
    if tools.is_empty() {
        return Ok(());
    }
    if tools.iter().any(|tool| {
        !matches!(
            tool.get("type").and_then(Value::as_str),
            Some("function" | "custom")
        )
    }) {
        return Err(EncodeError::request(
            "The SIWC integration supports only local function/custom tools",
        ));
    }
    let local_tools = std::mem::take(tools);
    tools.push(json!({"type": "namespace", "name": "rig", "tools": local_tools}));
    if body.get("tool_choice").is_some_and(Value::is_object) {
        return Err(EncodeError::request(
            "SIWC namespaced tools require auto, none, or required tool choice",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
