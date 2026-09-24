//! Model metadata returned by providers with model listing support.
//!
//! Use [`ModelList`] for provider responses and [`Model`] for each advertised
//! model entry. A provider with a model-listing endpoint builds its wire with
//! `models()`; a [`Model`](crate::driver::Model) over that wire implements
//! [`ModelLister`].
//!
//! ```
//! use rig_core::model::Model;
//!
//! let model = Model::from_id("example");
//! assert_eq!(model.display_name(), "example");
//! ```

pub mod listing;

pub use listing::{BoxedModelLister, Model, ModelList, ModelLister, ModelPage};
