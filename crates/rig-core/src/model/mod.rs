//! Model metadata returned by providers with model listing support.
//!
//! Use [`ModelList`] for provider responses and [`Model`] for each advertised
//! model entry. Provider clients expose listing through
//! [`ModelListingClient`](crate::client::ModelListingClient) when their
//! capabilities declare support.

pub mod listing;

pub use listing::{Model, ModelList, ModelListingError};
