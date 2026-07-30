//! Browser chat example backed by SmolLM2 embedded in WebAssembly.

use std::cell::RefCell;

use rig::{
    agent::{AgentConfig, ToolCatalog, prepare_request},
    candle::{CandleModel, GgufModelData},
    message::Message,
};
use wasm_bindgen::prelude::*;

#[derive(Debug, thiserror::Error)]
enum BrowserModelError {
    #[error("no model is embedded; run ./build.sh before opening the example")]
    MissingEmbeddedModel,
    #[error("embedded model initialization failed: {0}")]
    Initialization(String),
    #[error("the model has not been initialized")]
    NotInitialized,
    #[error("message must not be empty")]
    EmptyMessage,
    #[error("message is too large; use at most {max_bytes} UTF-8 bytes")]
    MessageTooLarge { max_bytes: usize },
    #[error("local inference failed: {0}")]
    Inference(String),
}

const CONFIG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/config.json"));
const TOKENIZER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tokenizer.json"));
const WEIGHTS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.gguf"));
const MAX_MESSAGE_BYTES: usize = 1024;
const MAX_HISTORY_MESSAGES: usize = 12;
const MAX_HISTORY_JSON_BYTES: usize = 2048;

// Candle has no `ProviderConfig` arm (model tensors are not expressible as
// plain configuration), so the chat drives the sans-IO agent protocol
// directly: `prepare_request` builds each turn's request and the loaded
// model executes it through its inherent `complete` method (there is no
// model trait any more).
struct ChatState {
    model: CandleModel,
    config: AgentConfig,
    history: Vec<Message>,
}

thread_local! {
    static CHAT_STATE: RefCell<Option<ChatState>> = const { RefCell::new(None) };
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

/// Reports whether the build found and embedded all three model artifacts.
#[wasm_bindgen]
pub fn model_is_embedded() -> bool {
    env!("RIG_CANDLE_WASM_MODEL_EMBEDDED") == "1"
}

/// Returns the total number of model artifact bytes embedded in the module.
#[wasm_bindgen]
pub fn embedded_model_size() -> usize {
    CONFIG.len() + TOKENIZER.len() + WEIGHTS.len()
}

/// Loads the embedded model and creates the in-browser Rig agent.
#[wasm_bindgen]
pub fn initialize() -> Result<(), JsValue> {
    if !model_is_embedded() {
        return Err(js_error(BrowserModelError::MissingEmbeddedModel));
    }
    if CHAT_STATE.with(|state| state.borrow().is_some()) {
        return Ok(());
    }

    let model = CandleModel::from_gguf_bytes(GgufModelData {
        config: CONFIG,
        tokenizer: TOKENIZER,
        weights: WEIGHTS,
    })
    .map_err(|error| js_error(BrowserModelError::Initialization(error.to_string())))?;
    let mut config = AgentConfig::new().with_preamble(
        "Repeat facts the user asks you to remember. Use those facts in later answers. \
         Never claim you cannot remember conversation history.",
    );
    config.temperature = Some(0.0);
    config.max_tokens = Some(32);
    config.additional_params = Some(serde_json::json!({"repeat_penalty": 1.0}));
    CHAT_STATE.with(|state| {
        state.replace(Some(ChatState {
            model,
            config,
            history: Vec::new(),
        }));
    });
    Ok(())
}

/// Sends one conversational turn through Rig and the embedded Candle model.
#[wasm_bindgen]
pub async fn chat(message: String) -> Result<String, JsValue> {
    if message.trim().is_empty() {
        return Err(js_error(BrowserModelError::EmptyMessage));
    }
    if message.len() > MAX_MESSAGE_BYTES {
        return Err(js_error(BrowserModelError::MessageTooLarge {
            max_bytes: MAX_MESSAGE_BYTES,
        }));
    }

    let mut state = CHAT_STATE
        .with(|slot| slot.take())
        .ok_or_else(|| js_error(BrowserModelError::NotInitialized))?;
    trim_history(&mut state.history);
    let result = run_turn(&mut state, message).await;
    CHAT_STATE.with(|slot| slot.replace(Some(state)));
    result.map_err(js_error)
}

/// One tool-less conversational turn through the sans-IO agent protocol:
/// prepare the request, run the loaded model, then append the user prompt and
/// assistant reply to the history (the same history bookkeeping `Agent::chat`
/// performs for an agent without tools).
async fn run_turn(state: &mut ChatState, message: String) -> Result<String, BrowserModelError> {
    let prompt = Message::user(message);
    let prepared = prepare_request(
        &state.config,
        &ToolCatalog::default(),
        false,
        prompt.clone(),
        &state.history,
        None,
        None,
    )
    .map_err(|error| BrowserModelError::Inference(error.to_string()))?;
    let response = state
        .model
        .complete(prepared.request)
        .await
        .map_err(|error| BrowserModelError::Inference(error.to_string()))?;
    let reply: String = response
        .choice
        .iter()
        .filter_map(|content| match content {
            rig::completion::AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    state.history.push(prompt);
    state.history.push(Message::assistant(reply.clone()));
    Ok(reply)
}

fn trim_history(history: &mut Vec<Message>) {
    while history.len() > MAX_HISTORY_MESSAGES
        || serde_json::to_vec(&*history).map_or(usize::MAX, |serialized| serialized.len())
            > MAX_HISTORY_JSON_BYTES
    {
        let next_turn = history
            .iter()
            .enumerate()
            .skip(1)
            .find_map(|(index, message)| matches!(message, Message::User { .. }).then_some(index))
            .unwrap_or(history.len());
        history.drain(..next_turn);
    }
}

/// Returns the number of messages currently retained by the browser chat.
#[wasm_bindgen]
pub fn history_len() -> Result<usize, JsValue> {
    CHAT_STATE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|state| state.history.len())
            .ok_or_else(|| js_error(BrowserModelError::NotInitialized))
    })
}

/// Clears the conversation while retaining the loaded model.
#[wasm_bindgen]
pub fn clear_history() -> Result<(), JsValue> {
    CHAT_STATE.with(|slot| {
        let mut state = slot.borrow_mut();
        let state = state
            .as_mut()
            .ok_or_else(|| js_error(BrowserModelError::NotInitialized))?;
        state.history.clear();
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_window_keeps_recent_complete_turns_within_bounds() {
        let mut history = Vec::new();
        for turn in 0..10 {
            history.push(Message::user(format!("turn-{turn}-{}", "x".repeat(400))));
            history.push(Message::assistant(format!(
                "answer-{turn}-{}",
                "y".repeat(100)
            )));
        }

        trim_history(&mut history);

        assert!(history.len() <= MAX_HISTORY_MESSAGES);
        assert!(
            serde_json::to_vec(&history)
                .is_ok_and(|serialized| serialized.len() <= MAX_HISTORY_JSON_BYTES)
        );
        assert!(matches!(history.first(), Some(Message::User { .. })));
        assert!(history.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|item| matches!(
                        item,
                        rig::message::UserContent::Text(text) if text.text.starts_with("turn-9-")
                    ))
            )
        }));
    }
}
