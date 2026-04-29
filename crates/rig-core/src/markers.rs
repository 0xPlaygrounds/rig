//! Common marker traits and structs for type-safe builders.

use serde::{Deserialize, Serialize};

/// Marker struct representing missing data in a request builder.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Missing;

/// Marker struct representing provided data in a request builder.
///
/// The generic type `T` represents the type of the provided data.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Provided<T>(pub T);
