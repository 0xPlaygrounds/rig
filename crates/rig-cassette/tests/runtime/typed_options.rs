//! The acceptance tests of typed options, the model catalog, provider
//! extras, the one `raw` shape, citations and cost, as
//! `crates/rig-core/TYPED_OPTIONS.md` specifies them. Each group is
//! compiled out until the phase named on its gate lands, which deletes the
//! gate; the tests must then pass unchanged.

#[path = "typed_options/tests.rs"]
mod tests;
