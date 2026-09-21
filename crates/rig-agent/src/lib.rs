#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
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
//! Rig's classic agent runtime: builders, serializable run state, hooks, tools,
//! memory orchestration, and typed extraction.
//!
//! Portable contracts are available through [`core`]; recording accepts
//! [`rig_core::serve::Recorder`], with concrete replay adapters in `rig-cassette`.
//! Native and browser WASM targets are supported; WASI is not. MCP integration
//! is provided by the native-only `rig-rmcp` crate.
//!
//! ```
//! use rig_agent::{Agent, AgentBuilder, core::completion::CompletionModel};
//! fn assistant(model: impl CompletionModel + 'static) -> Agent {
//!     AgentBuilder::new(model).preamble("Be concise.").build()
//! }
//! ```

extern crate self as rig;

/// Portable provider, data, memory, and tool contracts, also used as the expansion
/// root for portable `#[rig_tool]` functions. These exports are not forwarded to
/// the `rig_agent` crate root.
///
/// ```
/// use rig_agent::core::Embed;
/// fn accepts_embeddings<T: Embed>() {}
/// ```
pub mod core {
    pub use rig_core::*;
}

pub mod agent;
pub mod bus;
pub mod client;
pub mod completion;
pub mod extractor;
/// Ready-made integrations: the CLI chatbot.
pub mod integrations;
pub(crate) use rig_core::json_utils;
pub mod prelude;
pub mod run;
pub mod streaming;
pub(crate) mod sync;
#[cfg(any(test, feature = "test-utils"))]
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub mod test_utils;
pub mod tool;

pub use agent::TypedPromptResponse;
pub use agent::{
    Agent, AgentBuilder, AgentHook, AgentRun, AgentRunner, HookContext, ModelHandle, ModelRef,
    ModelSelection, ModelSelectionAction,
};

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use rig_derive::rig_tool;

// Compile-time thread-safety contract: the agent surface must be safe to hold
// in shared host state (worker pools, ECS resources) on native targets.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<Agent>();
    assert_send_sync_static::<AgentRunner>();
    assert_send_sync_static::<ModelHandle>();
    assert_send_sync_static::<agent::MultiTurnStreamItem>();
    assert_send_sync_static::<agent::RunEvents>();
    assert_send_sync_static::<agent::PromptResponse>();
    assert_send_sync_static::<tool::server::ToolServerHandle>();
    assert_send_sync_static::<tool::ToolSet>();
    assert_send_sync_static::<tool::ToolCatalog>();
};
