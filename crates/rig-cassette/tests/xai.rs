#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]

use rig_test_support::cache_conformance;
#[path = "common/cassette_safety.rs"]
mod cassette_safety;
use rig_test_support::cassettes;
use rig_test_support::raw_capture;
use rig_test_support::reasoning;
use rig_test_support::support;

#[path = "providers/xai/mod.rs"]
mod xai;

use rig_test_support::matrix;

#[path = "common/image_inputs.rs"]
#[allow(dead_code)]
mod image_inputs;

#[path = "common/request_identity.rs"]
mod request_identity;
