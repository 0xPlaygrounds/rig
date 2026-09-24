//! The mock provider's wire documents: what [`MockCompletionModel`] reports as
//! a response's `raw`.
//!
//! The mock is a provider, so its documents have a layout of their own
//! rather than following rig's types: every field is spelled, an absent value
//! as `null`, in a fixed order.
//!
//! [`MockCompletionModel`]: super::MockCompletionModel

use serde_json::{Map, Value, json};

use super::streaming::MockFinal;
use crate::completion::{AssistantContent, FinishReason, Usage};

/// A unary turn's document.
pub(super) fn turn(
    choice: &[AssistantContent],
    usage: &Usage,
    message_id: Option<&str>,
    response_id: Option<&str>,
    provider_request_id: Option<&str>,
    finish_reason: Option<&FinishReason>,
) -> Result<Value, serde_json::Error> {
    let choice = choice.iter().map(content).collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "choice": choice,
        "usage": usage,
        "message_id": message_id,
        "response_id": response_id,
        "provider_request_id": provider_request_id,
        "finish_reason": finish_reason,
    }))
}

/// The document the mock reports as the `raw` of a streamed turn's terminal
/// `record`: the record in the mock's own layout.
pub fn mock_terminal_document(record: &MockFinal) -> Result<Value, serde_json::Error> {
    let mut document = Map::new();
    document.insert("usage".into(), serde_json::to_value(record.usage)?);
    document.insert(
        "finish_reason".into(),
        serde_json::to_value(&record.finish_reason)?,
    );
    document.insert("message_id".into(), json!(record.message_id));
    document.insert("response_id".into(), json!(record.response_id));
    if let Some(id) = &record.provider_request_id {
        document.insert("provider_request_id".into(), json!(id));
    }
    document.insert("provider".into(), json!(record.provider));
    if let Some(issuer) = &record.reasoning_issuer {
        document.insert("reasoning_issuer".into(), json!(issuer));
    }
    document.insert("model".into(), json!(record.model));
    document.insert("raw".into(), record.raw.clone());
    Ok(Value::Object(document))
}

/// One content part: a tool call spells its signature and parameters last,
/// a reasoning part its id after its type.
fn content(part: &AssistantContent) -> Result<Value, serde_json::Error> {
    let Value::Object(mut fields) = serde_json::to_value(part)? else {
        return serde_json::to_value(part);
    };
    let mut spelled = |key: &str| (key.to_owned(), fields.remove(key).unwrap_or(Value::Null));
    let document: Map<String, Value> = match part {
        AssistantContent::ToolCall(_) => {
            let signature = spelled("signature");
            let additional_params = spelled("additional_params");
            fields
                .into_iter()
                .chain([signature, additional_params])
                .collect()
        }
        AssistantContent::Reasoning(_) => {
            let id = spelled("id");
            let tag = spelled("type");
            [tag, id].into_iter().chain(fields).collect()
        }
        AssistantContent::Text(_) | AssistantContent::Image(_) => fields,
    };
    Ok(Value::Object(document))
}
