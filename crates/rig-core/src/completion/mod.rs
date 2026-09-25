//! Provider-agnostic completion requests and responses. A provider's wire
//! translates [`CompletionRequest`] into its own request and normalizes the
//! reply as [`CompletionResponse`]; a [`Model`](crate::Model) sends it.
//!
//! ```no_run
//! use rig_core::{Model, completion::CompletionRequestBuilder, providers::openai::OpenAI};
//!
//! # async fn run(http: rig_core::http_client::BoxedHttpClient) -> Result<(), Box<dyn std::error::Error>> {
//! let model = Model::new(OpenAI::from_env()?.completion("gpt-4o"), http);
//! let request = CompletionRequestBuilder::new("What is Rig?").build();
//! let response = model.call(request, None).await?;
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
