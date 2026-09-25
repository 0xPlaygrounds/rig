//! Environment configuration helpers.
//! A provider configuration builds wires; a [`Model`](crate::driver::Model)
//! binds one to a transport.
//!
//! ```no_run
//! use rig_core::client::env;
//!
//! let credential = env::required("OPENAI_API_KEY")?;
//! # let _ = credential;
//! # Ok::<(), env::EnvError>(())
//! ```

mod cached_content;
pub mod env;
pub(crate) mod verify;

pub use env::EnvError;
