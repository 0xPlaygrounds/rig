//! Shared provider infrastructure: the wire-adapter contract, its
//! single-policy-site driver, and the decode-then-validate classify layer.
//!
//! [`adapter`], [`wire`], and [`tool_call_bridge`] are public so out-of-tree
//! providers implement [`adapter::WireAdapter`] and inherit the shared driver,
//! frame-triage policy, and index→identity tool-call bridging instead of
//! hand-rolling per-provider assemblers; the remaining helpers are
//! crate-private.

pub mod adapter;
pub(crate) mod auth;
pub(crate) mod chunk_lifecycle;
pub(crate) mod openai_chat_completions_compatible;
#[cfg(any(test, debug_assertions))]
pub(crate) mod sequence_law;
pub mod tool_call_bridge;
pub mod wire;

pub(crate) fn completion_usage(
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    cached_input_tokens: u64,
) -> crate::completion::Usage {
    crate::completion::Usage {
        input_tokens,
        output_tokens,
        total_tokens,
        cached_input_tokens,
        cache_creation_input_tokens: 0,
        tool_use_prompt_tokens: 0,
        reasoning_tokens: 0,
    }
}
