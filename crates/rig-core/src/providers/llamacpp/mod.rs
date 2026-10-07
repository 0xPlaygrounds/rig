//! llama.cpp model identifiers.
//!
//! [`from_env`] and [`new`] build a client on the
//! [`LLAMACPP`](crate::providers::openai::wire::LLAMACPP) dialect, which defaults to
//! `http://localhost:8080/v1`. `LLAMACPP_API_BASE_URL` overrides the endpoint;
//! `LLAMACPP_API_KEY` supplies optional bearer authentication for secured servers.
//!
//! ```no_run
//! use rig_core::providers::llamacpp;
//! let model = llamacpp::from_env()?.chat(llamacpp::LLAMA_CPP);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod completion;
pub mod extension;

pub use completion::LLAMA_CPP;

/// The provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "llamacpp";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::LLAMACPP, "llama.cpp");
