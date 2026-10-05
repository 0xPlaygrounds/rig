//! The turn-termination matrix (`turn_termination_matrix`, rig#2184) once
//! per reply shape that matters to it. A hook's `ModelTurnFinished` must
//! carry the normalized reason the provider stopped and the cap that
//! attempt ran under, on both surfaces. What differs per provider is only
//! how its wire spells the reason, so each cell runs over every bank reply,
//! of every provider and encoder, that decodes to the cell's ending: a
//! truncated answer, a finished answer, or a turn that calls `add`. The
//! escalating retry runs once per provider and reply mode that holds both a
//! truncated and a finished answer. Every cell runs on rig-agent's runner
//! and in a world (`ecs_termination`), which reads the same facts off its
//! turns.
//!
//! The per-provider copies also re-read the cap each recorded request
//! carried; that is what an encoder sends, pinned by the request snapshots.

use rig::completion::FinishReason;
use rig_test_support::bank::{self, Entry};

use rig_ecs::agent::{MaxTokens, Temperature};

use crate::decode::{decode, with_model};
use crate::ecs_agent::EcsAgent;
use crate::ecs_termination::{self, NativeEscalation, NativeProbe};
use crate::support::{
    Adder, EscalateCapOnTruncation, ObservedTermination, TurnTerminationProbe,
    collect_stream_final_response,
};

/// Where a cell runs: rig-agent's runner, or an agent graph in a world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Surface {
    Agent,
    World,
}

const TINY_CAP: u64 = 16;
const ROOMY_CAP: u64 = 512;
const TRUNCATING_PROMPT: &str = "Write two sentences about maple trees.";
const SHORT_PROMPT: &str = "Reply with exactly the word: cedar.";
const TOOL_PROMPT: &str = "Calculate 2 + 3.";
const CONCISE_PREAMBLE: &str = "You are a concise assistant. Answer directly in plain text.";
const TOOL_PREAMBLE: &str = "You are a calculator. Use the add tool to answer.";

/// The turn a cell's reply must decode to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Turn {
    Truncated,
    Finished,
    Tool,
}

impl Turn {
    fn reason(self) -> FinishReason {
        match self {
            Self::Truncated => FinishReason::Length,
            Self::Finished => FinishReason::Stop,
            Self::Tool => FinishReason::ToolCalls,
        }
    }
}

/// Every bank reply of the mode that decodes to `turn`: an answer with text
/// for a truncated or finished turn, a call to `add` alone for a tool turn.
async fn replies(turn: Turn, streamed: bool) -> Vec<Entry> {
    let mut found = Vec::new();
    for provider in bank::providers() {
        for entry in bank::entries(&provider).iter() {
            if entry.then.status != 200 || entry.streamed() != streamed {
                continue;
            }
            let Some(decoded) = decode(entry).await else {
                continue;
            };
            let fits = match turn {
                Turn::Truncated | Turn::Finished => {
                    decoded.calls.is_empty() && !decoded.text.trim().is_empty()
                }
                Turn::Tool => decoded.calls == ["add"],
            };
            if fits && decoded.error.is_none() && decoded.finish == Some(turn.reason()) {
                found.push(entry.clone());
            }
        }
    }
    found
}

/// One attempt over `entry` on `surface`, the run answered after it by
/// `then` where given: the reasons and caps observed.
async fn attempt(
    entry: &Entry,
    then: Option<&Entry>,
    turn: Turn,
    streamed: bool,
    surface: Surface,
) -> Vec<ObservedTermination> {
    let cap = if turn == Turn::Truncated {
        TINY_CAP
    } else {
        ROOMY_CAP
    };
    let (preamble, prompt) = match turn {
        Turn::Truncated => (CONCISE_PREAMBLE, TRUNCATING_PROMPT),
        Turn::Finished => (CONCISE_PREAMBLE, SHORT_PROMPT),
        Turn::Tool => (TOOL_PREAMBLE, TOOL_PROMPT),
    };
    // A tool turn's run goes on to an answer; only the first turn is the
    // cell's.
    let max_turns = (turn == Turn::Tool).then_some(3);
    let script: Vec<Entry> = std::iter::once(entry).chain(then).cloned().collect();
    let http = bank::client(&script);
    if surface == Surface::World {
        let probe = NativeProbe::default();
        let ran = with_model!(entry, http, |model| {
            let mut ecs = EcsAgent::new(model, preamble, 1);
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert((Temperature(Some(0.0)), MaxTokens(Some(cap))));
            if turn == Turn::Tool {
                ecs.tool(Adder);
            }
            ecs_termination::install(&mut ecs, probe.clone(), None);
            ecs.prompt_with_max_turns(prompt, streamed, max_turns).await;
        });
        assert!(ran.is_some(), "{}: the bank decodes it", entry.source);
        probe.observations()
    } else {
        let probe = TurnTerminationProbe::default();
        let hook = probe.clone();
        let ran = with_model!(entry, http, |model| {
            let builder = rig::AgentBuilder::new(model)
                .preamble(preamble)
                .temperature(0.0)
                .max_tokens(cap);
            if turn == Turn::Tool {
                let agent = builder.tool(Adder).build();
                let runner = agent.prompt(prompt).add_hook(hook).max_turns(3);
                if streamed {
                    let mut stream = runner.stream();
                    let _ = collect_stream_final_response(&mut stream).await;
                } else {
                    let _ = runner.run().await;
                }
            } else {
                let agent = builder.build();
                let runner = agent.prompt(prompt).add_hook(hook);
                if streamed {
                    let mut stream = runner.stream();
                    collect_stream_final_response(&mut stream)
                        .await
                        .unwrap_or_else(|error| panic!("{}: {error}", entry.source));
                } else {
                    runner
                        .run()
                        .await
                        .unwrap_or_else(|error| panic!("{}: {error}", entry.source));
                }
            }
        });
        assert!(ran.is_some(), "{}: the bank decodes it", entry.source);
        probe.observations()
    }
}

async fn cell(turn: Turn, streamed: bool, surface: Surface) {
    let replies = replies(turn, streamed).await;
    assert!(!replies.is_empty(), "the bank holds a {turn:?} reply");
    // A tool turn is answered by a finished answer of the same wire.
    let answers = if turn == Turn::Tool {
        self::replies(Turn::Finished, streamed).await
    } else {
        Vec::new()
    };
    let mut ran = 0;
    let cap = if turn == Turn::Truncated {
        TINY_CAP
    } else {
        ROOMY_CAP
    };
    for entry in &replies {
        let then = answers
            .iter()
            .find(|answer| answer.provider == entry.provider && answer.encoder == entry.encoder);
        if turn == Turn::Tool && then.is_none() {
            continue;
        }
        ran += 1;
        let observed = attempt(entry, then, turn, streamed, surface).await;
        let (reason, max_tokens) = observed
            .first()
            .cloned()
            .unwrap_or_else(|| panic!("{}: {surface:?} observed no turn", entry.source));
        assert_eq!(
            reason,
            Some(turn.reason()),
            "{} on {surface:?}: the wire's reason reaches the hook normalized",
            entry.source
        );
        assert_eq!(
            max_tokens,
            Some(cap),
            "{} on {surface:?}: the hook reports the cap this attempt ran under",
            entry.source
        );
        assert_eq!(
            reason.is_some_and(|reason| reason.truncated_output()),
            turn == Turn::Truncated,
            "{} on {surface:?}: only a truncated turn satisfies the retry predicate",
            entry.source
        );
    }
    assert!(ran > 0, "a {turn:?} reply ran on {surface:?}");
}

#[tokio::test]
async fn blocking_truncated_turn_reports_length_and_cap() {
    cell(Turn::Truncated, false, Surface::Agent).await;
}

#[tokio::test]
async fn world_blocking_truncated_turn_reports_length_and_cap() {
    cell(Turn::Truncated, false, Surface::World).await;
}

#[tokio::test]
async fn streaming_truncated_turn_reports_length_and_cap() {
    cell(Turn::Truncated, true, Surface::Agent).await;
}

#[tokio::test]
async fn world_streaming_truncated_turn_reports_length_and_cap() {
    cell(Turn::Truncated, true, Surface::World).await;
}

#[tokio::test]
async fn blocking_completed_turn_reports_stop_and_cap() {
    cell(Turn::Finished, false, Surface::Agent).await;
}

#[tokio::test]
async fn world_blocking_completed_turn_reports_stop_and_cap() {
    cell(Turn::Finished, false, Surface::World).await;
}

#[tokio::test]
async fn streaming_completed_turn_reports_stop_and_cap() {
    cell(Turn::Finished, true, Surface::Agent).await;
}

#[tokio::test]
async fn world_streaming_completed_turn_reports_stop_and_cap() {
    cell(Turn::Finished, true, Surface::World).await;
}

#[tokio::test]
async fn blocking_tool_turn_reports_tool_calls() {
    cell(Turn::Tool, false, Surface::Agent).await;
}

#[tokio::test]
async fn world_blocking_tool_turn_reports_tool_calls() {
    cell(Turn::Tool, false, Surface::World).await;
}

#[tokio::test]
async fn streaming_tool_turn_reports_tool_calls() {
    cell(Turn::Tool, true, Surface::Agent).await;
}

#[tokio::test]
async fn world_streaming_tool_turn_reports_tool_calls() {
    cell(Turn::Tool, true, Surface::World).await;
}

/// A truncated answer and then a finished one, both of one provider,
/// encoder and mode: the first attempt truncates under the tiny cap, the
/// escalation hook retries with the roomy one, and each attempt reports its
/// own cap rather than the agent's baseline.
async fn escalation(streamed: bool, surface: Surface) {
    let truncated = replies(Turn::Truncated, streamed).await;
    let finished = replies(Turn::Finished, streamed).await;
    let mut paired = std::collections::BTreeSet::new();
    for first in &truncated {
        let Some(second) = finished
            .iter()
            .find(|entry| entry.provider == first.provider && entry.encoder == first.encoder)
        else {
            continue;
        };
        if !paired.insert((first.provider.clone(), first.encoder.clone())) {
            continue;
        }
        let script = [first.clone(), second.clone()];
        let http = bank::client(&script);
        let (observed, escalations, retries) = if surface == Surface::World {
            let probe = NativeProbe::default();
            let escalate = NativeEscalation::new(TINY_CAP, ROOMY_CAP);
            let ran = with_model!(first, http, |model| {
                let mut ecs = EcsAgent::new(model, CONCISE_PREAMBLE, 1);
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert((Temperature(Some(0.0)), MaxTokens(Some(64))));
                ecs_termination::install(&mut ecs, probe.clone(), Some(escalate.clone()));
                ecs.prompt_with_max_turns(TRUNCATING_PROMPT, streamed, Some(2))
                    .await;
            });
            assert!(ran.is_some());
            (
                probe.observations(),
                escalate.escalations(),
                escalate.retries(),
            )
        } else {
            let probe = TurnTerminationProbe::default();
            let escalate = EscalateCapOnTruncation::new(TINY_CAP, ROOMY_CAP);
            let (hook, escalating) = (probe.clone(), escalate.clone());
            let ran = with_model!(first, http, |model| {
                let agent = rig::AgentBuilder::new(model)
                    .preamble(CONCISE_PREAMBLE)
                    .temperature(0.0)
                    .max_tokens(64)
                    .build();
                let runner = agent
                    .prompt(TRUNCATING_PROMPT)
                    .add_hook(hook)
                    .add_hook(escalating)
                    .max_turns(2);
                if streamed {
                    let mut stream = runner.stream();
                    collect_stream_final_response(&mut stream)
                        .await
                        .unwrap_or_else(|error| panic!("{}: {error}", first.source));
                } else {
                    runner
                        .run()
                        .await
                        .unwrap_or_else(|error| panic!("{}: {error}", first.source));
                }
            });
            assert!(ran.is_some());
            (
                probe.observations(),
                escalate.escalations(),
                escalate.retries(),
            )
        };
        assert_eq!(
            observed,
            vec![
                (Some(FinishReason::Length), Some(TINY_CAP)),
                (Some(FinishReason::Stop), Some(ROOMY_CAP)),
            ],
            "{} then {} on {surface:?}: each attempt reports its own cap",
            first.source,
            second.source
        );
        assert_eq!(escalations, vec![ROOMY_CAP]);
        assert_eq!(retries, 1);
    }
    assert!(
        !paired.is_empty(),
        "the bank holds a truncated and a finished reply of one wire"
    );
}

#[tokio::test]
async fn blocking_escalating_retry_reports_each_attempts_own_cap() {
    escalation(false, Surface::Agent).await;
}

#[tokio::test]
async fn streaming_escalating_retry_reports_each_attempts_own_cap() {
    escalation(true, Surface::Agent).await;
}

#[tokio::test]
async fn world_blocking_escalating_retry_reports_each_attempts_own_cap() {
    escalation(false, Surface::World).await;
}

#[tokio::test]
async fn world_streaming_escalating_retry_reports_each_attempts_own_cap() {
    escalation(true, Surface::World).await;
}
