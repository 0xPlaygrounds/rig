//! Ordered content entities and content-addressed binary payloads.
//!
//! ```
//! use rig_ecs::agent::content::binary::BinaryAssets;
//! let assets = BinaryAssets::default();
//! assert!(assets.is_empty());
//! ```

pub mod binary;

pub mod parts;

pub mod reflect;
