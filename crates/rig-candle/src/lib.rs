//! Local CPU inference from caller-supplied, validated model artifacts.
//! Supports selected Llama, SmolLM2, and Qwen3 checkpoints without performing
//! filesystem or network access. WASM inference is synchronous; use a worker.
//!
//! ```
//! use rig_candle::ConversationProtocol;
//!
//! let protocol = ConversationProtocol::Qwen3;
//! ```

mod artifacts;
mod generation;
mod loader;
mod model;
mod profile;
mod protocol;
mod runtime;
mod types;
mod validation;

pub use artifacts::{GgufModelData, ModelArtifacts, ModelData};
pub use generation::{GenerationConfig, GenerationEvent};
pub use model::{CandleModel, CandleModelBuilder, stream_from_events};
pub use profile::{ConversationProtocol, ModelArchitecture, Quantization};
pub use types::{CandleCompletionResponse, CandleError, FinishReason};

pub(crate) use profile::{
    BEGIN_OF_TEXT, END_HEADER, END_OF_TURN, IM_END, IM_START, SMOLLM2_DEFAULT_SYSTEM_PROMPT,
    START_HEADER,
};
