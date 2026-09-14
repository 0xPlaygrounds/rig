//! Ordered content entities and content-addressed binary payloads.

pub mod binary;

pub mod cache;

pub mod parts;

#[cfg(feature = "reflect")]
pub mod reflect;
