//! Anthropic API client and Rig integration

pub mod client;
pub mod completion;
pub mod model_listing;
pub mod observation;
pub mod streaming;

pub use client::{
    ANTHROPIC, Anthropic, AnthropicConfig, AnthropicKey, Client, ClientBuilder, Dialect, Verify,
    compatible,
};
pub use completion::{
    CLAUDE_FABLE_5, CLAUDE_FABLE_5_1, CLAUDE_HAIKU_4_5, CLAUDE_OPUS_4_6, CLAUDE_OPUS_4_7,
    CLAUDE_OPUS_4_8, CLAUDE_OPUS_5, CLAUDE_SONNET_4_6, CLAUDE_SONNET_5, CacheTtl, CompletionModel,
    Messages,
};
pub use model_listing::{AnthropicModelLister, Models};
