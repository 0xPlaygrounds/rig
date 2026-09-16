//! llama.cpp (`llama-server`) as an OpenAI chat-completions dialect.
//!
//! [llama.cpp](https://github.com/ggml-org/llama.cpp) ships `llama-server`, an
//! OpenAI-compatible HTTP server that serves a GGUF model from local hardware.
//! Started with no arguments beyond a model it listens on
//! `http://localhost:8080` and exposes `/v1/chat/completions`, `/v1/embeddings`,
//! `/v1/models`, `/v1/rerank` and more. Rig reaches all of them through
//! [`openai::wire::LLAMACPP`](crate::providers::openai::wire::LLAMACPP), so
//! this module has no client and no models of its own — what lives here is
//! [`LLAMA_CPP`] and the typed read of llama.cpp's own reply,
//! [`CompletionResponse`] (see [`completion`]).
//!
//! # This module replaces `providers::llamafile`
//!
//! Rig used to reach the same server through a provider named after Mozilla's
//! [llamafile](https://github.com/Mozilla-Ocho/llamafile) distribution. A
//! `.llamafile` bundles the *same* llama.cpp server into a single executable
//! and serves the *same* OpenAI-compatible API, so one dialect covers both:
//! point the base URL at a running `.llamafile` and everything works exactly
//! as it does against `llama-server`. See `MIGRATING.md`.
//!
//! # Base URL
//!
//! The dialect's default base URL is `http://localhost:8080/v1`, and
//! `LLAMACPP_API_BASE_URL` or
//! [`OpenAI::with_base_url`](crate::providers::openai::wire::OpenAI::with_base_url)
//! overrides it. llama.cpp's operational routes (`/props`, `/health`,
//! `/slots`, `/metrics`, `/tokenize`, …) are served at the server *root* and
//! answer 404 under `/v1`, which the dialect records rather than guesses.
//!
//! # Authentication
//!
//! A local `llama-server` needs no credential and *rejects* a request that
//! carries an `Authorization` header, so the dialect's credential is an
//! optional bearer: absent unless `LLAMACPP_API_KEY` (or an explicit key) is
//! set, which is what a server started with `--api-key <key>` requires.
//!
//! # Example
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model.
//! ```no_run
//! use rig_core::providers::llamacpp;
//! use rig_core::providers::openai::wire::{LLAMACPP, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // A local server, no credential.
//! let model = OpenAI::from_env_with(&LLAMACPP)?.chat(llamacpp::LLAMA_CPP);
//!
//! // `llama-server --api-key hunter2`, on another host.
//! let secured = OpenAI::with_key(&LLAMACPP, "hunter2")
//!     .with_base_url("http://gpu.local:8080")
//!     .chat(llamacpp::LLAMA_CPP);
//! # let _ = (model, secured);
//! # Ok(())
//! # }
//! ```

pub mod completion;

pub use completion::{CompletionResponse, LLAMA_CPP, Timings};
