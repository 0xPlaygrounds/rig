//! The agent tool sessions (`agent_tool_sessions`) once: Groq's rows, the
//! wire whose recordings of the shared session cells all replay, over its
//! own pinned replies instead of its cassette. A session asserts exact tool
//! arguments, the pairing of every call with its result in history, and
//! answer tokens, so it runs over the replies its scenario recorded
//! (`bank::recorded`), not over replies of the same shape. The other wires'
//! copies repeat the same runtime; their direct rows read wire-specific
//! metadata and stay with their providers.

/// The cassette wrapper the session file calls, over the bank instead.
mod support {
    use rig::providers::openai::wire::{GROQ, OpenAIConfig};
    use rig_test_support::bank;
    use rig_test_support::cassette_models::OpenAiModels;
    use std::future::Future;

    pub(super) async fn with_groq_cassette_result<F, Fut, E>(
        scenario: &str,
        test_body: F,
    ) -> Result<(), E>
    where
        F: FnOnce(OpenAiModels) -> Fut,
        Fut: Future<Output = Result<(), E>>,
    {
        let replies = bank::recorded("groq", scenario);
        let config = OpenAIConfig::with_key(&GROQ, "bank").with_base_url("http://bank.invalid");
        test_body(OpenAiModels::new(config, bank::client(&replies))).await
    }
}

// The session file names helpers only its own target uses.
#[allow(unused_imports)]
#[path = "../providers/groq/agent_tool_sessions.rs"]
mod agent_tool_sessions;
