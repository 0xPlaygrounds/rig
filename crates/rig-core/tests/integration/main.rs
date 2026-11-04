//! The integration tests of rig-core, one module per suite, linked into one
//! binary. The websocket suites need the `tungstenite` feature.

mod custom_token_store;
mod driver_adoption;
mod macro_hygiene;
mod model_listing_warnings;
mod openai_responses_websocket;
mod out_of_tree_decoder;
mod provider_config_base_url_env;
mod pull_protocol;
#[cfg(feature = "tungstenite")]
mod streaming_conformance_websocket;
