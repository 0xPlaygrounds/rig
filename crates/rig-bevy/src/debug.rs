//! Redacted explanation views derived from authoritative ECS facts.

use bevy_ecs::world::World;
use serde::{Deserialize, Serialize};

use crate::{
    components::{
        CallBudget, CommittedTranscript, RunNode, TerminalState, ToolCallNode, UsageLedger,
    },
    topology::RunId,
};

/// Safe-by-default explanation of one run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunExplanation {
    /// Stable run ID.
    pub run: RunId,
    /// Lifecycle phase.
    pub phase: String,
    /// Model calls dispatched versus configured budget.
    pub calls: String,
    /// Count of committed canonical messages; content is redacted.
    pub committed_messages: usize,
    /// Aggregate token count.
    pub total_tokens: u64,
    /// Terminal reason rendered without prompt/tool arguments.
    pub terminal: Option<String>,
    /// Tool call facts with arguments and results omitted.
    pub tool_calls: Vec<RedactedToolCall>,
}

/// Redacted tool-call fact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedactedToolCall {
    /// Provider-facing name.
    pub name: String,
    /// Pinned revision.
    pub revision: u64,
    /// Model-call order.
    pub ordinal: usize,
    /// Whether policy suppressed execution.
    pub suppressed: bool,
    /// Whether ingress committed a result.
    pub committed: bool,
}

/// Build a redacted explanation from authoritative components.
pub fn explain(world: &mut World, run_id: RunId) -> Option<RunExplanation> {
    let run = world
        .query::<(
            &RunNode,
            &CallBudget,
            &CommittedTranscript,
            &UsageLedger,
            Option<&TerminalState>,
        )>()
        .iter(world)
        .find(|(run, _, _, _, _)| run.id == run_id)
        .map(|(run, budget, transcript, usage, terminal)| {
            (
                run.clone(),
                *budget,
                transcript.0.len(),
                usage.0.total_tokens,
                terminal.map(|terminal| terminal_label(&terminal.reason).to_string()),
            )
        })?;

    let mut tool_calls = world
        .query::<&ToolCallNode>()
        .iter(world)
        .filter(|call| call.run == run_id)
        .map(|call| RedactedToolCall {
            name: call.name.clone(),
            revision: call.revision,
            ordinal: call.ordinal,
            suppressed: call.suppressed,
            committed: call.committed,
        })
        .collect::<Vec<_>>();
    tool_calls.sort_by_key(|call| call.ordinal);

    Some(RunExplanation {
        run: run.0.id,
        phase: format!("{:?}", run.0.phase),
        calls: format!("{}/{}", run.1.dispatched, run.1.limit),
        committed_messages: run.2,
        total_tokens: run.3,
        terminal: run.4,
        tool_calls,
    })
}

fn terminal_label(reason: &crate::components::TerminalReason) -> &'static str {
    use crate::components::TerminalReason;

    match reason {
        TerminalReason::Completed => "completed",
        TerminalReason::BudgetExhausted => "budget_exhausted",
        TerminalReason::Cancelled(_) => "cancelled",
        TerminalReason::Stopped(_) => "stopped",
        TerminalReason::ProviderFailure(_) => "provider_failure",
        TerminalReason::ToolFailure(_) => "tool_failure",
        TerminalReason::OutputFailure(_) => "output_failure",
        TerminalReason::StoreFailure(_) => "store_failure",
        TerminalReason::Livelock => "livelock",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        components::{ProgressState, RunPhase, TerminalReason},
        topology::{AgentId, Generation, TenantId},
    };
    use rig_core::{completion::Usage, message::Message};

    #[test]
    fn explanation_omits_prompt_and_arguments() {
        let mut world = World::new();
        let run = RunId::allocate();
        world.spawn((
            RunNode {
                id: run,
                agent: AgentId::allocate(),
                tenant: TenantId(8),
                generation: Generation(0),
                phase: RunPhase::Terminal,
            },
            CallBudget {
                limit: 1,
                dispatched: 1,
            },
            CommittedTranscript(vec![Message::user("SECRET PROMPT")]),
            UsageLedger(Usage::new()),
            ProgressState {
                changes: 1,
                idle_passes: 0,
                max_idle_passes: 8,
            },
            TerminalState {
                reason: TerminalReason::Completed,
                committed_tick: 1,
            },
        ));
        let explanation = explain(&mut world, run).expect("explanation");
        let json = serde_json::to_string(&explanation).expect("serialize");
        assert!(!json.contains("SECRET PROMPT"));
        assert_eq!(explanation.committed_messages, 1);
    }
}
