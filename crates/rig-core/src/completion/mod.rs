//! Provider-agnostic completion requests and responses. A provider's wire
//! translates [`CompletionRequest`] into its own request and normalizes the
//! reply as [`CompletionResponse`]; a [`Model`](crate::Model) sends it.
//!
//! ```no_run
//! use rig_core::{Model, completion::CompletionRequest, providers::openai::OpenAIConfig};
//!
//! # async fn run(http: rig_core::http_client::DynHttpClient) -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAIConfig::from_env()?.connect(http).completion("gpt-4o");
//! let request = CompletionRequest::new("What is Rig?");
//! let response = model.call(request).await?;
//! println!("{:?}", response.choice);
//! # Ok(())
//! # }
//! ```

pub mod cache_cost;
pub mod handle;
pub mod history;
pub mod message;
pub mod options;
pub mod prompt;
pub mod provider_options;
pub mod request;

pub use cache_cost::{CacheCost, CacheRates};
pub use handle::ModelRef;
pub use history::{Accepts, LaterSystem, Media, Pairing, Place, Replay, ReplayTarget, adapt};
pub use message::{AssistantContent, AssistantMessage, Message, MessageError};
pub use options::{
    CacheRetention, CheckError, Effort, GenerationOptions, OnUnsupported, Reasoning, ServiceTier,
    UnsupportedOption, Verbosity,
};
pub use prompt::{Section, SystemPrompt};
pub use provider_options::{
    ExtensionOptions, OptionsError, ProviderExtension, ProviderOptions, ReplyExtras, SHARED,
};
pub use request::*;
