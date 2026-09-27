//! Model metadata returned by providers with model listing support.
//!
//! Use [`ModelList`] for provider responses and [`ModelInfo`] for each
//! advertised model entry. A provider with a model-listing wire returns it
//! from `models()`; call it through a [`Model`](crate::driver::Model).
//!
//! ```
//! use rig_core::model::ModelInfo;
//!
//! let model = ModelInfo::from_id("example");
//! assert_eq!(model.display_name(), "example");
//! ```

pub mod listing;

pub use listing::{ModelInfo, ModelList};
