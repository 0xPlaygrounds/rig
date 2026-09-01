#![doc = include_str!("../README.md")]

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
pub use generation::GenerationConfig;
pub use model::{CandleModel, CandleModelBuilder, LlamaModel, stream_from_events};
pub use profile::{ConversationProtocol, ModelArchitecture, ModelFamily, Quantization};
pub use types::{CandleCompletionResponse, CandleError, FinishReason};

pub(crate) use profile::{
    BEGIN_OF_TEXT, END_HEADER, END_OF_TURN, IM_END, IM_START, SMOLLM2_DEFAULT_SYSTEM_PROMPT,
    START_HEADER,
};
