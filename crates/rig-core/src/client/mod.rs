//! What is left of provider construction now that a provider is data.
//!
//! A provider is plain configuration (`openai::wire::OpenAI`,
//! `anthropic::wire::Anthropic`, `cohere::Cohere`, …) plus one
//! [`Wire`](crate::wire::Wire) per API endpoint; binding that configuration to
//! a transport with [`Bound`](crate::driver::Bound) is what produces a model,
//! and [`Bound`](crate::driver::Bound) is where `completion(model)`,
//! `embedding(model, ndims)`, `verify()` and their siblings live. Nothing
//! generic sits between a provider and its transport any more.
//!
//! Two things outlive that move, and this module is exactly those two:
//!
//! - [`mod@env`]: reading a provider's configuration out of the process
//!   environment, with [`EnvError`] naming a variable that is absent or
//!   unusable. This is the only construction step that can fail before a
//!   request is sent — building a transport fails as
//!   [`http_client::Error`](crate::http_client::Error), which is the error the
//!   transport layer already has.
//! - [`VerifyError`]: the `Verify` operation's error. Verification is the one
//!   operation whose reply *status* is the entire answer, so the 401/403
//!   reading lives in its error type instead of being restated by every wire.

pub mod env;
pub mod verify;

pub use env::EnvError;
pub use verify::VerifyError;
