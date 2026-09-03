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

/// The header names the program: its hook stack, its required effect row
/// and its bus policy. Each is a refusal with both sides in the message.
#[tokio::test]
async fn the_header_names_the_program_and_the_agent_refuses_another() {
    use rig_agent::agent::{AgentHook, HookContext, RunStart, RunStartAction};
    use rig_core::effect::EffectFamily;

    #[derive(serde::Deserialize)]
    struct NoArgs {}
    struct Add;
    impl rig_agent::tool::Tool for Add {
        const NAME: &'static str = "add";
        type Args = NoArgs;
        type Output = i64;
        type Error = rig_agent::tool::ToolExecutionError;
        fn description(&self) -> String {
            "adds".into()
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn call(
            &self,
            _context: &mut rig_agent::tool::ToolContext,
            _args: NoArgs,
        ) -> Result<i64, Self::Error> {
            Ok(0)
        }
    }

    struct Tagger;
    impl AgentHook for Tagger {
        async fn on_run_start(&self, _ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
            RunStartAction::Continue
        }
    }

    let recorded = AgentBuilder::new(MockCompletionModel::text("hi"))
        .name("planner")
        .add_hook(Tagger)
        .record_effects()
        .build();
    within(recorded.prompt("go").run()).await.expect("run");
    let log = recorded.take_effect_log().expect("recording");
    assert_eq!(log.header.hooks.len(), 1);
    assert!(
        log.header.hooks[0].ends_with("Tagger"),
        "{:?}",
        log.header.hooks
    );
    assert_eq!(
        log.header.required.get(recorded.model_key().raw()),
        Some(&EffectFamily::Completion),
        "the required row names the model"
    );
    assert_eq!(log.header.bus, recorded.bus_config());
    recorded
        .check_replayable(&log)
        .expect("its own log replays");

    // Another hook stack, same spec: another program.
    let other_hooks = AgentBuilder::new(MockCompletionModel::text("hi"))
        .name("planner")
        .build();
    let refusal = other_hooks
        .check_replayable(&log)
        .expect_err("a different hook stack is a different program");
    assert!(
        refusal.message.contains("hook stack"),
        "{}",
        refusal.message
    );
    assert!(
        refusal.message.contains("Tagger"),
        "both stacks: {}",
        refusal.message
    );

    // A program that needs a tool the log never served.
    let needs_a_tool = AgentBuilder::new(MockCompletionModel::text("hi"))
        .name("planner")
        .add_hook(Tagger)
        .tool(Add)
        .build();
    let refusal = needs_a_tool
        .check_replayable(&log)
        .expect_err("a tool the log never served");
    assert!(
        refusal.message.contains("never served"),
        "{}",
        refusal.message
    );
    assert!(refusal.message.contains("tool:add"), "{}", refusal.message);

    // Another bus policy.
    let serial = AgentBuilder::new(MockCompletionModel::text("hi"))
        .name("planner")
        .add_hook(Tagger)
        .configure_bus(rig_bus::BusConfig {
            serial_per_handler: true,
            ..rig_bus::BusConfig::default()
        })
        .build();
    let refusal = serial
        .check_replayable(&log)
        .expect_err("a different serving policy");
    assert!(
        refusal.message.contains("bus policy"),
        "{}",
        refusal.message
    );
}
