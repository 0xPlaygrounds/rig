//! The root package's integration tests, one module per suite, linked into one
//! binary. The live-only provider suites live in `tests/providers/` and the
//! Bedrock integrations in `tests/integrations/`.

use rig_test_support::{goldens, reasoning, support};

mod api_surface;
#[allow(clippy::expect_used)]
#[path = "../providers/azure/mod.rs"]
mod azure;
mod connect_credentials;
#[allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]
#[path = "../core/mod.rs"]
mod core;
#[allow(clippy::expect_used)]
#[path = "../providers/huggingface/mod.rs"]
mod huggingface;
#[allow(clippy::expect_used)]
#[path = "../providers/hyperbolic/mod.rs"]
mod hyperbolic;
mod integrations;
mod launcher_plugins;
#[allow(clippy::expect_used, clippy::panic)]
#[path = "../providers/minimax/mod.rs"]
mod minimax;
#[allow(clippy::expect_used)]
#[path = "../providers/mira/mod.rs"]
mod mira;
#[allow(clippy::expect_used, clippy::panic)]
#[path = "../providers/moonshot/mod.rs"]
mod moonshot;
#[allow(clippy::expect_used)]
#[path = "../providers/together/mod.rs"]
mod together;
mod tool_facade_traits;
#[allow(clippy::expect_used, clippy::indexing_slicing)]
#[path = "../providers/voyageai/mod.rs"]
mod voyageai;
#[allow(clippy::expect_used, clippy::panic)]
#[path = "../providers/xiaomimimo/mod.rs"]
mod xiaomimimo;
#[allow(clippy::expect_used)]
#[path = "../providers/zai/mod.rs"]
mod zai;
