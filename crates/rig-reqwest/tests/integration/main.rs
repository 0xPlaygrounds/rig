//! The integration tests of rig-reqwest, one module per suite, linked into one
//! binary.

mod cancellation;
mod middleware_features;
mod middleware_runtime;
mod multipart;
mod no_system_roots;
mod off_runtime;
mod retry_class;
