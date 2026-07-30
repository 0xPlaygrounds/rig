//! Marker struct for type-safe builders.
//!
//! The `Provided<T>` half of this module was removed together with the modality and
//! vector-search request typestate builders (single-architecture R6); those requests
//! now use plain constructors plus `with_*` setters.
//!
//! `Missing` survives only as the default type parameter of the `ClientBuilder`
//! typestate in [`crate::client`] and the per-provider `ClientBuilder` aliases that
//! re-export it. R7 deletes `Client`/`ClientBuilder`, which removes this module's last
//! consumer — delete `markers.rs` then.

use serde::{Deserialize, Serialize};

/// Marker struct representing missing data in a builder.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Missing;
