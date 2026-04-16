mod auth;
mod embeddings;
mod routing;

use rig::providers::copilot;

pub(crate) fn live_builder() -> copilot::ClientBuilder {
    let mut builder = copilot::Client::builder();

    if let Ok(base_url) =
        std::env::var("GITHUB_COPILOT_API_BASE").or_else(|_| std::env::var("COPILOT_BASE_URL"))
    {
        builder = builder.base_url(base_url);
    }

    if let Some(api_key) = ["GITHUB_COPILOT_API_KEY", "COPILOT_API_KEY", "GITHUB_TOKEN"]
        .into_iter()
        .find_map(|name| {
            std::env::var(name)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
    {
        builder.api_key(api_key)
    } else {
        builder.oauth()
    }
}

pub(crate) fn live_client() -> copilot::Client {
    live_builder().build().expect("Copilot client should build")
}
