//! llama.cpp's conventional model identifier. A server's timings stay in the
//! response's `raw` value.
//!
//! ```
//! assert_eq!(rig_core::providers::llamacpp::LLAMA_CPP, "LLaMA_CPP");
//! ```

/// Conventional identifier for a single-model server. For a multi-model router,
/// use an identifier from its model listing instead.
pub const LLAMA_CPP: &str = "LLaMA_CPP";
