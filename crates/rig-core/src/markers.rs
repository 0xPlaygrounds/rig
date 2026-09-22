//! Missing and provided states for type-safe builders.
//!
//! ```
//! use rig_core::markers::Provided;
//!
//! let value = Provided("ready");
//! assert_eq!(value.0, "ready");
//! ```

use serde::{Deserialize, Serialize};

/// Marker struct representing missing data in a request builder.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Missing;

/// Builder state containing a supplied value of type `T`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Provided<T>(pub T);
