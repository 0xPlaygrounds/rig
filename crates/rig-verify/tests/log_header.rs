//! The log an agent records carries the run spec it ran under and the keys
//! it performed effects on; a named agent mints the same keys in every
//! process, and an agent refuses a log it cannot replay before the first
//! dispatch.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use std::time::Duration;

use rig_agent::AgentBuilder;
use rig_core::test_utils::MockCompletionModel;

async fn within<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("a run over the bus never hangs")
}

#[test]
fn a_named_agent_mints_the_same_keys_every_time() {
    let keys = |name: Option<&str>| {
        let mut builder = AgentBuilder::new(MockCompletionModel::text("hi"));
        if let Some(name) = name {
            builder = builder.name(name);
        }
        let agent = builder.build();
        let mut keys = agent.tool_server_handle().owner();
        keys.push('|');
        keys.push_str(agent.model_key().as_str());
        keys
    };
    assert_eq!(
        keys(Some("planner")),
        keys(Some("planner")),
        "a named agent's keys do not depend on how many agents came before"
    );
    assert!(keys(Some("planner")).starts_with("planner|planner/model:default"));
    assert_ne!(
        keys(None),
        keys(None),
        "an anonymous agent's keys come from the process counter"
    );
}

#[tokio::test]
async fn an_agents_log_carries_its_spec_and_the_agent_checks_it() {
    let agent = AgentBuilder::new(MockCompletionModel::text("hi"))
        .name("planner")
        .record_effects()
        .build();
    within(agent.prompt("go").run()).await.expect("run");
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(log.header.run_spec, Some(agent.run_spec_hash()));
    assert!(
        log.header.signature.contains_key(agent.model_key().raw()),
        "the signature names the model key"
    );
    agent.check_replayable(&log).expect("its own log replays");

    // Another agent, another spec (max turns), same name: refused up front,
    // with both hashes in the message.
    let other = AgentBuilder::new(MockCompletionModel::text("hi"))
        .name("planner")
        .default_max_turns(7)
        .build();
    let report = other.check_replayable(&log).expect_err("a different spec");
    assert!(
        report.message.contains("run spec") && report.message.contains("this agent runs under"),
        "{}",
        report.message
    );

    // Same spec, a different name: the key the log needs is not on this bus.
    let renamed = AgentBuilder::new(MockCompletionModel::text("hi"))
        .name("reviewer")
        .build();
    let report = renamed
        .check_replayable(&log)
        .expect_err("the key is absent");
    assert!(
        report.message.contains("nothing serves"),
        "{}",
        report.message
    );
}
