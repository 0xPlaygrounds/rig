//! The cache markers a request cannot ask for itself: the end of the tool
//! definitions and of the system prompt, placed in the Anthropic Messages
//! body on its way out.

use rig_harness::rig_core::http_client::{self, Bytes, HeaderMap, HttpMiddleware, Method, Uri};
use rig_harness::rig_core::wasm_compat::WasmBoxedFuture;
use serde_json::{Map, Value, json};

/// The most `cache_control` markers an Anthropic request takes.
const BUDGET: usize = 4;

/// Places cache markers at the end of the tools and of the system prompt
/// of every Anthropic Messages request, where the encoder left none and
/// the budget allows.
pub struct PlaceMarkers;

impl HttpMiddleware for PlaceMarkers {
    fn before_request_body<'a>(
        &'a self,
        _: &'a Method,
        _: &'a Uri,
        headers: &'a HeaderMap,
        body: Bytes,
    ) -> WasmBoxedFuture<'a, http_client::Result<Bytes>> {
        let body = if headers.contains_key("anthropic-version") {
            place_markers(&body).map_or(body, Bytes::from)
        } else {
            body
        };
        Box::pin(async move { Ok(body) })
    }
}

/// `body`, an Anthropic Messages request, with a marker on its last tool
/// and on its system prompt's last block, unless one is there or the
/// budget of four is spent; none when it is not such a body or nothing
/// changed. The markers take the one-hour lifetime when the request has a
/// one-hour marker already, since those must come first.
pub fn place_markers(body: &[u8]) -> Option<Vec<u8>> {
    let mut json: Value = serde_json::from_slice(body).ok()?;
    let request = json.as_object_mut()?;
    request.get("messages")?.as_array()?;
    let mut markers: Vec<Value> = request
        .get("cache_control")
        .filter(|marker| !marker.is_null())
        .cloned()
        .into_iter()
        .collect();
    let blocks = ["tools", "system"]
        .into_iter()
        .filter_map(|key| request.get(key)?.as_array())
        .flatten()
        .chain(
            request
                .get("messages")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|message| message.get("content")?.as_array())
                .flatten(),
        );
    markers.extend(blocks.filter_map(|block| block.get("cache_control").cloned()));
    let long = markers
        .iter()
        .any(|marker| marker.get("ttl") == Some(&json!("1h")));
    let marker = if long {
        json!({ "type": "ephemeral", "ttl": "1h" })
    } else {
        json!({ "type": "ephemeral" })
    };
    let mut left = BUDGET.saturating_sub(markers.len());
    let mut changed = false;

    let last_tool = request
        .get_mut("tools")
        .and_then(Value::as_array_mut)
        .and_then(|tools| {
            tools
                .iter_mut()
                .filter_map(Value::as_object_mut)
                .rfind(|tool| tool.get("defer_loading") != Some(&Value::Bool(true)))
        });
    if let Some(tool) = last_tool
        && left > 0
        && mark(tool, &marker)
    {
        left -= 1;
        changed = true;
    }

    if let Some(Value::String(text)) = request.get("system") {
        let block = json!([{ "type": "text", "text": text }]);
        request.insert("system".to_owned(), block);
    }
    let last_block = request
        .get_mut("system")
        .and_then(Value::as_array_mut)
        .and_then(|system| system.last_mut())
        .and_then(Value::as_object_mut);
    if let Some(block) = last_block
        && left > 0
        && mark(block, &marker)
    {
        changed = true;
    }
    changed.then(|| serde_json::to_vec(&json).ok()).flatten()
}

/// Puts `marker` on `block` unless it has one; whether it did.
fn mark(block: &mut Map<String, Value>, marker: &Value) -> bool {
    if block.contains_key("cache_control") {
        return false;
    }
    block.insert("cache_control".to_owned(), marker.clone());
    true
}
