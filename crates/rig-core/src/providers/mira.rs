//! The Mira gateway: [`from_env`] and [`new`] build a client on the
//! [`MIRA`](crate::providers::openai::wire::MIRA) dialect. Model identifiers
//! come from its model listing rather than constants.
//!
//! ```no_run
//! use rig_core::providers::mira;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let models = mira::from_env()?.list_models().await?;
//! # let _ = models;
//! # Ok(())
//! # }
//! ```

crate::client::macros::openai_vendor!(crate::providers::openai::wire::MIRA, "Mira");
