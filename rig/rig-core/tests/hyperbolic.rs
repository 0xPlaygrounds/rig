//! Hyperbolic integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test hyperbolic`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test hyperbolic agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "hyperbolic/agent.rs"]
mod agent;
#[cfg(feature = "audio")]
#[path = "hyperbolic/audio_generation.rs"]
mod audio_generation;
#[cfg(feature = "image")]
#[path = "hyperbolic/image_generation.rs"]
mod image_generation;
