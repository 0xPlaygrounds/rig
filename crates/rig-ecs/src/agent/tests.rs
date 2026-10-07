//! The agent components' own helpers.

use rig_core::providers::openrouter::extension::{OpenRouterExt, OpenRouterOptions};
use rig_core::providers::xai::extension::{XaiExt, XaiOptions};

use super::ProviderOptions;

#[test]
fn set_on_the_component_equals_the_long_form() {
    let xai = XaiOptions::new().prompt_cache_key("k");
    let openrouter = OpenRouterOptions::new().session_id("s-1");
    let short = ProviderOptions::default()
        .set(xai.clone())
        .set(OpenRouterOptions::new().session_id("s-0"))
        .set(openrouter.clone());
    let long = rig_core::completion::ProviderOptions::new()
        .with::<XaiExt>(&xai)
        .and_then(|options| options.with::<OpenRouterExt>(&openrouter))
        .map(ProviderOptions);
    assert_eq!(Ok(short), long.map_err(|error| error.to_string()));
}
