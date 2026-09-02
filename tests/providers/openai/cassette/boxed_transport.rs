//! OpenAI scenarios replayed through the erased transport.
//!
//! `BoxedHttpClient` must be byte-transparent: the same recorded exchanges the
//! generic `Client<Ext, ReqwestClient>` produced must match when the client is
//! `Client<Ext, BoxedHttpClient>` over the same transport. The replay server
//! matches on method, path, allowlisted headers and body bytes, so a boxed
//! request that differed in any of them would not find its interaction.

use rig::prelude::*;
use rig::providers::openai;

use super::super::support::with_openai_boxed_cassette;
use crate::support::{
    BASIC_PREAMBLE, BASIC_PROMPT, STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response,
    collect_stream_final_response_and_provider_final,
};

#[tokio::test]
async fn completion_smoke_through_boxed_transport() {
    with_openai_boxed_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(BASIC_PREAMBLE)
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed")
            .output;

        assert_nonempty_response(&response);
    })
    .await;
}

#[tokio::test]
async fn streaming_smoke_through_boxed_transport() {
    with_openai_boxed_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(STREAMING_PREAMBLE)
            .build();

        let mut stream = agent.stream_prompt(STREAMING_PROMPT).stream().await;
        let (response, provider_final): (_, rig::streaming::StreamFinal) =
            collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
        assert_eq!(provider_final.provider, "openai");
        assert!(provider_final.usage.total_tokens > 0);
    })
    .await;
}
