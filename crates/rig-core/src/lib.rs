#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! Provider-agnostic model, message, tool, memory, and vector-store contracts.
//! Provider configurations build endpoint wires, and a [`Model`] binds a
//! wire to a [`driver::Transport`]; a [`BoxedModel`] is a model erased to
//! its operation, for consumers that store one. Companion crates supply
//! transports, agent runtimes, and external storage integrations.
//!
//! ```no_run
//! use rig_core::BoxedModel;
//! use rig_core::completion::{CompletionRequestBuilder, CompletionResponse};
//! use rig_core::error::ProviderError;
//! use rig_core::operation::Completion;
//!
//! async fn ask(model: &BoxedModel<Completion>) -> Result<CompletionResponse, ProviderError> {
//!     let request = CompletionRequestBuilder::new("Who are you?").build();
//!     model.call(request).await
//! }
//! ```

extern crate self as rig;

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;
pub mod client;
pub mod completion;
pub mod driver;
pub mod effect;
pub mod embeddings;
pub mod error;
pub mod http_client;
pub mod id;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
/// Internal JSON helpers shared with sibling runtime crates (e.g. `rig-agent`).
/// Not part of rig-core's stable public API.
#[doc(hidden)]
pub mod json_utils;
pub mod loaders;
pub mod markers;
pub mod memory;
pub mod model;
pub mod observe;
pub mod operation;
pub mod prelude;
pub(crate) mod provider_response;
pub mod providers;
pub mod rerank;
pub mod serve;

pub mod streaming;
#[cfg(any(test, feature = "test-utils"))]
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub mod test_utils;
pub mod tool;
pub mod transcript;
pub mod transcription;
pub mod vector_store;
pub mod wasm_compat;
pub mod wire;
#[cfg(feature = "websocket")]
#[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
pub mod ws_client;

pub use completion::message;
pub use driver::{BoxedModel, Model};
pub use embeddings::Embed;
pub use error::{ErrorKind, ErrorReport, ProviderError};
pub use provider_response::ProviderResponseError;
// `schemars`, `serde`, and `serde_json` are re-exported so macro-generated
// code (and downstream crates) can resolve them through Rig instead of
// requiring a direct dependency on each.
pub use schemars;
pub use serde;
pub use serde_json;

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use rig_derive::ContextValue;
#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use rig_derive::Embed;

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use rig_derive::rig_tool;

pub mod telemetry;

// Native runtime values must retain their thread-safety bounds.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    fn assert_send_static<T: Send + 'static>() {}
    assert_send_sync_static::<tool::ManagedToolToken>();
    assert_send_sync_static::<streaming::StreamEvent>();
    // The serializable identity a typed view is resolved from.
    assert_send_sync_static::<completion::ModelRef>();
    assert_send_sync_static::<tool::DynamicTool>();
    // One erased transport, shared by every provider a host binds.
    assert_send_sync_static::<http_client::BoxedHttpClient>();
    // A live stream is owned by one poller: `Send` so it can move to a worker,
    // not `Sync`.
    assert_send_static::<streaming::CompletionStream>();
};
