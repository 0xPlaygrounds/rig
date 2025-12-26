//! z.ai Provider
//!
//! This module contains the client and models for the z.ai provider.

pub mod client;
pub mod completion;

pub use client::Client;
pub use completion::{
    CompletionModel, GLM_4, GLM_4_0520, GLM_4_AIR, GLM_4_AIRX, GLM_4_FLASH, GLM_4_PLUS,
};
