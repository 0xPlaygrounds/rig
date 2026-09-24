//! Model metadata returned by providers with model listing support.
//!
//! Use [`ModelList`] for provider responses and [`ModelInfo`] for each advertised
//! model entry. A provider with a model-listing endpoint builds its wire with
//! `models()`; a [`Model`](crate::driver::Model) over that wire implements
//! [`ModelLister`].
//!
//! ```
//! use rig_core::model::ModelInfo;
//!
//! let model = ModelInfo::from_id("example");
//! assert_eq!(model.display_name(), "example");
//! ```

pub mod listing;

pub use listing::{BoxedModelLister, ModelInfo, ModelList, ModelLister, ModelPage};
