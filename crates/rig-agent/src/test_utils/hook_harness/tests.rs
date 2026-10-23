use rig_core::conversation::{Aside, Author};
use rig_core::effect::{EffectId, EffectKind};

use super::HookTestHarness;
use crate::agent::{AgentHook, DispatchAction, DispatchEvent, HookContext};

/// Allows `limit` tool calls per run, counted in the scratchpad.
struct CallLimit {
    limit: usize,
}

impl AgentHook for CallLimit {
    async fn on_dispatch(&self, ctx: &HookContext, _: DispatchEvent<'_>) -> DispatchAction {
        let calls = ctx.scratchpad().update(|calls: &mut usize| {
            *calls += 1;
            *calls
        });
        if calls > self.limit {
            DispatchAction::skip("call limit reached")
        } else {
            DispatchAction::Proceed
        }
    }
}

fn dispatch(kind: &EffectKind, turn: usize) -> DispatchEvent<'_> {
    DispatchEvent {
        id: EffectId::from_raw(turn as u64),
        kind,
        turn,
        call_id: None,
        context: None,
    }
}

#[tokio::test]
async fn run_state_persists_across_hook_calls() {
    let harness = HookTestHarness::new()
        .with_agent_name("budgeted")
        .with_streaming(true)
        .with_hook(CallLimit { limit: 2 })
        .with_entries(&[Aside::new("note", Author::user(), serde_json::json!(1))]);
    let kind = EffectKind::ToolCall {
        name: "add".into(),
        args: "{}".into(),
    };

    let mut actions = Vec::new();
    for turn in 1..=3 {
        harness.set_turn(turn);
        actions.push(
            harness
                .hooks()
                .on_dispatch(harness.context(), dispatch(&kind, turn))
                .await,
        );
    }

    let context = harness.context();
    assert!(matches!(
        actions.as_slice(),
        [
            DispatchAction::Proceed,
            DispatchAction::Proceed,
            DispatchAction::Deny(_)
        ]
    ));
    assert_eq!(context.scratchpad().get::<usize>(), Some(3));
    assert_eq!(context.turn(), 3);
    assert_eq!(context.agent_name(), Some("budgeted"));
    assert!(context.is_streaming());
    assert_eq!(context.entries("note").len(), 1);
}
