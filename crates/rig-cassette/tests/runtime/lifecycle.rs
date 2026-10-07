//! The lifecycle matrix (`lifecycle_matrix`) once: HTTP middleware phases,
//! run-start rewrites and settle hooks on rig-agent, which every wire runs
//! the same. Anthropic's rows run over
//! their own pinned replies (the rewrite cells read the marker the rewritten
//! prompt asked for), through the same middleware stack, instead of their
//! cassettes. Gemini's and OpenAI's copies repeat the same runtime.

/// The cassette wrapper the lifecycle files call, over the bank instead.
mod support {
    use rig::http_client::DynHttpClient;
    use rig::providers::anthropic::wire::AnthropicConfig;
    use rig_test_support::bank;
    use rig_test_support::cassette_models::AnthropicModels;
    use std::future::Future;

    pub(super) async fn with_anthropic_lifecycle_cassette<M, F, Fut>(
        scenario: &str,
        middleware: M,
        test_body: F,
    ) where
        M: rig::http_client::HttpMiddleware + 'static,
        F: FnOnce(AnthropicModels) -> Fut,
        Fut: Future<Output = ()>,
    {
        let replies = bank::recorded("anthropic", scenario);
        let http =
            DynHttpClient::new(bank::BankHttpClient::new(&replies)).with_middleware(middleware);
        let config = AnthropicConfig::new("bank").with_base_url("http://bank.invalid");
        test_body(AnthropicModels::new(config, http)).await;
    }
}

// The files reach the wrapper as `super::super::support`, as they do under
// their provider; the module's own directory is this file's.
#[path = "."]
mod cassette {
    #[path = "../providers/anthropic/cassette/lifecycle_matrix.rs"]
    mod lifecycle_matrix;
}
