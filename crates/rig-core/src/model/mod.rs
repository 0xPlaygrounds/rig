//! Model metadata returned by providers with model listing support.
//!
//! Use [`ModelList`] for provider responses and [`Model`] for each advertised
//! model entry. A provider that declares a model-listing wire
//! ([`HasModelListing`](crate::driver::HasModelListing)) reaches the catalogue
//! through `models()` on its [`Bound`](crate::driver::Bound); a
//! provider without one has no such method.
//!
//! ```
//! use rig_core::model::Model;
//!
//! let model = Model::from_id("example");
//! assert_eq!(model.display_name(), "example");
//! ```

pub mod listing;

pub use listing::{Model, ModelList, ModelLister};
