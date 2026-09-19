#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use rig_test_support::cache_conformance;
#[path = "common/cassette_safety.rs"]
mod cassette_safety;
use rig_test_support::cassettes;
use rig_test_support::raw_capture;
use rig_test_support::support;

#[path = "providers/groq/mod.rs"]
mod groq;

use rig_test_support::matrix;
