//! Model metadata returned by providers with model listing support.
//!
//! Use [`ModelList`] for provider responses and [`Model`] for each advertised
//! model entry. Providers that support listing expose it as a
//! `functions::list_models` free function returning these types.

pub mod listing;

pub use listing::{Model, ModelList, ModelListingError};
