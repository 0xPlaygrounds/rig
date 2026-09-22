//! Provider-agnostic completion requests, responses, and model traits.
//! Providers translate [`CompletionRequest`] into wire requests and normalize
//! replies as [`CompletionResponse`].
//!
//! ```no_run
//! use rig_core::completion::CompletionModel;
//!
//! # async fn run(model: &(impl CompletionModel + Clone)) -> Result<(), Box<dyn std::error::Error>> {
//! let request = model.completion_request("What is Rig?").build();
//! let response = model.completion(request).await?;
//! println!("{:?}", response.choice);
//! # Ok(())
//! # }
//! ```

pub mod handle;
pub mod message;
pub mod request;

pub use handle::ModelRef;
pub use message::{AssistantContent, Message, MessageError};
pub use request::*;
