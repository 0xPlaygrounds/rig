//! Browser chat example backed by a Llama model embedded in WebAssembly.

use std::cell::RefCell;

use rig::{
    agent::{Agent, AgentBuilder},
    candle::{CandleModel, GgufModelData},
    completion::Chat,
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
    #[error("local inference failed: {0}")]
    Inference(String),
}

const CONFIG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/config.json"));
const TOKENIZER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tokenizer.json"));
const WEIGHTS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.gguf"));

struct ChatState {
    agent: Agent<CandleModel>,
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

    let model = CandleModel::from_gguf_bytes(GgufModelData {
        config: CONFIG,
        tokenizer: TOKENIZER,
        weights: WEIGHTS,
    })
    .map_err(|error| js_error(BrowserModelError::Initialization(error.to_string())))?;
    let agent = AgentBuilder::new(model)
        .preamble(
            "Repeat facts the user asks you to remember. Use those facts in later answers. \
             Never claim you cannot remember conversation history.",
        )
        .temperature(0.0)
        .max_tokens(32)
        .additional_params(serde_json::json!({"repeat_penalty": 1.0}))
        .build();
    CHAT_STATE.with(|state| {
        state.replace(Some(ChatState {
            agent,
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

    let mut state = CHAT_STATE
        .with(|slot| slot.take())
        .ok_or_else(|| js_error(BrowserModelError::NotInitialized))?;
    let result = state.agent.chat(message, &mut state.history).await;
    CHAT_STATE.with(|slot| slot.replace(Some(state)));
    result.map_err(|error| js_error(BrowserModelError::Inference(error.to_string())))
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
