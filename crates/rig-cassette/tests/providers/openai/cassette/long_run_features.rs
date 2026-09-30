//! Prompt caching over a long live run of four conversations side by side on
//! gpt-6-luna, over Responses, sharing the preamble and one
//! `prompt_cache_key`, recorded and replayed.
//!
//! Record with `RIG_PROVIDER_TEST_MODE=record` and `OPENAI_API_KEY`; see
//! `tests/README.md`.

use rig::agent::Agent;
use rig::providers::openai::GPT_6_LUNA;
use rig_test_support::cache_longrun::{self, Limits, workloads};

use crate::cassettes::CassetteSpec;

use super::super::support::with_openai_long_run_cassette;
use super::long_run_workloads::{LUNA, check, support_agent};

/// Four conversations sharing the preamble and one `prompt_cache_key`,
/// concurrent from their second turn. Their requests all differ, so the
/// cassette replays them unordered, as the Gemini sub-agent run does.
#[tokio::test]
async fn fan_out_4x25() {
    let log = with_openai_long_run_cassette(
        CassetteSpec::new("long_run_caching/fan_out_4x25").unordered(),
        |client, clock| async move {
            let agents: Vec<Agent> = (0..4)
                .map(|_| support_agent(client.openai.completion(GPT_6_LUNA).into()))
                .collect();
            workloads::fan_out(
                &agents,
                &clock,
                25,
                &["FAN OUT B", "FAN OUT C", "FAN OUT D", "FAN OUT E"],
                &["B", "C", "D", "E"],
            )
            .await
        },
    )
    .await;
    let (recording, figures) = check(
        "long_run_caching/fan_out_4x25",
        GPT_6_LUNA,
        LUNA,
        Some(Limits {
            min_saving: Some(0.50),
            min_call_share: Some(0.50),
            max_writes_share: Some(0.25),
        }),
        &log,
    );
    // What each conversation's first call read: after the first, a read is
    // the shared preamble another conversation wrote.
    let mut first_reads = Vec::new();
    for marker in ["FAN OUT B", "FAN OUT C", "FAN OUT D", "FAN OUT E"] {
        let first = recording
            .calls
            .iter()
            .find(|call| call.conversation.contains(marker))
            .expect("each conversation made calls");
        first_reads.push((marker, first.usage.cached_input_tokens.unwrap_or(0)));
    }
    cache_longrun::print_limit(
        &figures,
        &format!("fan-out first-call reads per conversation {first_reads:?}"),
    );
    assert!(
        first_reads.iter().skip(1).all(|(_, reads)| *reads > 0),
        "every later conversation reads the shared prefix on its first call: {first_reads:?}"
    );
}
