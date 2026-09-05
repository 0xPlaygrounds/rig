//! Matrix B: the hook surface.
//!
//! Hooks are program. The header names the stack by type, a different
//! stack is refused before the first dispatch, and within the same stack
//! every decision is re-made on replay and must land where it landed at
//! record time: in the record (a patched call, a hook's own dispatch), in
//! the transcript only (a replaced result, a denied call's reason), in the
//! request (an overridden preamble), or in the run's shape (a retried
//! turn, a retried invalid call). The hand driver makes each decision
//! itself, so a row proves the decision is a function of the event.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | `observes` | default · every family |
//! | `on_dispatch` | Proceed · Patch (tool args) · Deny (skip) |
//! | `on_outcome` | Proceed · Replace (a tool result) · Replace (a completion) |
//! | `on_completion_call` | Continue · Patch (preamble) |
//! | `on_model_turn_finished` | Continue · Retry with feedback |
//! | `on_invalid_tool_call` | (none) · Retry, twice |
//! | a hook's own dispatch | none · `on_run_start` calls a tool |
//! | stack | one hook · two hooks |
//! | transport | unary · streamed with events (for Patch and Deny) |
//!
//! Full cross-product: 2 × 3 × 3 × 2 × 2 × 2 × 2 × 2 × 2 = 1152. Recorded:
//! the 12 cells below, one per decision plus the streamed twins of the
//! two dispatch decisions and one two-hook stack. Pruned: every other
//! combination, because the hook methods fire at distinct points and
//! compose by the stack's documented rules (`HookStack` docs), which
//! `rig-agent`'s unit tests cover; the corpus pins where each decision
//! lands on the record, and a stack of two shows the order the header
//! names. `on_model_select` is Matrix C's (routing); `on_text_delta`,
//! `on_reasoning_delta`, `on_tool_call_delta`, `on_run_settled` observe
//! and cannot change a record.
//!
//! # Cells
//!
//! | golden | producer | shape | decision lands in |
//! |---|---|---|---|
//! | `anthropic_hooks_observe_everything` | `corpus_hooks.rs` `observe_everything_…` | `[Memory, Completion, Tool, Completion, Memory]` | nowhere; the header names the hook |
//! | `anthropic_hooks_patch_tool_args` | `patch_tool_args_…` | `[Completion, Tool, Completion]` | the record: `add` runs with `{"x":40,"y":2}`; history keeps `{17, 25}` |
//! | `anthropic_hooks_patch_tool_args_streamed` | `patch_tool_args_streamed_…` | the same, events kept | the record |
//! | `anthropic_hooks_deny_tool` | `deny_tool_…` | `[Completion, Completion]` | the transcript: the reason as the tool's result; no tool record |
//! | `anthropic_hooks_deny_tool_streamed` | `deny_tool_streamed_…` | the same, events kept | the transcript |
//! | `anthropic_hooks_replace_tool_result` | `replace_tool_result_…` | `[Completion, Tool, Completion]` | the transcript: `99`; the record holds `42` |
//! | `anthropic_hooks_replace_answer` | `replace_answer_…` | `[Completion]` | the run's output: `REPLACED`; the record holds the model's text |
//! | `anthropic_hooks_preamble_override` | `preamble_override_…` | `[Completion]` | the request: the pirate preamble; the spec keeps the base |
//! | `anthropic_hooks_demand_done` | `demand_done_…` | `[Completion, Completion]` | the run's shape: a second completion after feedback |
//! | `anthropic_hooks_lookup_before_run` | `lookup_before_run_…` | `[Tool, Completion, Tool, Completion]` | the record: the hook's `add(1, 2)` under the tool's key, first |
//! | `anthropic_hooks_two_hooks` | `two_hooks_…` | `[Completion, Tool, Completion]` | both: the patched call in the record, `99` in the transcript; header `[PatchAddArgs, ReplaceAddResult]` |
//! | `mock_hooks_retry_twice` | `tests/core/golden_hooks.rs` `hooks_retry_twice_…` | `[Completion, Completion, Completion, Tool, Completion]` | the run's shape: two retries (mock-scripted: no live model calls an unadvertised tool) |
//!
//! Every Anthropic cell is recorded on the wire (`CLAUDE_SONNET_4_6`,
//! temperature 0) into `tests/cassettes/anthropic/corpus_hooks/`.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{Hook, Program, REPLACED_ANSWER};

const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const BASIC_PROMPT: &str = "In one or two sentences, explain what Rust programming language is and why memory safety matters.";
const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";

const TOOLS: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};
const BASIC: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: BASIC_PROMPT,
    temperature: Some(0.0),
    ..Program::DEFAULT
};

const OBSERVE_EVERYTHING: Program = Program {
    fixture: "anthropic_hooks_observe_everything",
    conversation: Some("golden-conversation"),
    hooks: &[Hook::ObserveEverything],
    ..TOOLS
};
const PATCH_TOOL_ARGS: Program = Program {
    fixture: "anthropic_hooks_patch_tool_args",
    hooks: &[Hook::PatchAddArgs],
    ..TOOLS
};
const PATCH_TOOL_ARGS_STREAMED: Program = Program {
    fixture: "anthropic_hooks_patch_tool_args_streamed",
    hooks: &[Hook::PatchAddArgs],
    streamed: true,
    ..TOOLS
};
const DENY_TOOL: Program = Program {
    fixture: "anthropic_hooks_deny_tool",
    hooks: &[Hook::DenyAdd],
    ..TOOLS
};
const DENY_TOOL_STREAMED: Program = Program {
    fixture: "anthropic_hooks_deny_tool_streamed",
    hooks: &[Hook::DenyAdd],
    streamed: true,
    ..TOOLS
};
const REPLACE_TOOL_RESULT: Program = Program {
    fixture: "anthropic_hooks_replace_tool_result",
    hooks: &[Hook::ReplaceAddResult],
    ..TOOLS
};
const REPLACE_ANSWER: Program = Program {
    fixture: "anthropic_hooks_replace_answer",
    hooks: &[Hook::ReplaceAnswer],
    expected_output: Some(REPLACED_ANSWER),
    ..BASIC
};
const PREAMBLE_OVERRIDE: Program = Program {
    fixture: "anthropic_hooks_preamble_override",
    hooks: &[Hook::PreambleOverride],
    ..BASIC
};
const DEMAND_DONE: Program = Program {
    fixture: "anthropic_hooks_demand_done",
    hooks: &[Hook::DemandDone],
    max_turns: Some(3),
    ..BASIC
};
const LOOKUP_BEFORE_RUN: Program = Program {
    fixture: "anthropic_hooks_lookup_before_run",
    hooks: &[Hook::LookupBeforeRun],
    ..TOOLS
};
const TWO_HOOKS: Program = Program {
    fixture: "anthropic_hooks_two_hooks",
    hooks: &[Hook::PatchAddArgs, Hook::ReplaceAddResult],
    ..TOOLS
};
const RETRY_TWICE: Program = Program {
    fixture: "mock_hooks_retry_twice",
    preamble: Some("Use the add tool."),
    prompt: "What is 2 + 3?",
    max_turns: Some(5),
    hooks: &[Hook::RetryUnknownTool],
    invalid_retries: 2,
    ..Program::DEFAULT
};

both_interpreters! {
    observe_everything: OBSERVE_EVERYTHING,
    patch_tool_args: PATCH_TOOL_ARGS,
    patch_tool_args_streamed: PATCH_TOOL_ARGS_STREAMED,
    deny_tool: DENY_TOOL,
    deny_tool_streamed: DENY_TOOL_STREAMED,
    replace_tool_result: REPLACE_TOOL_RESULT,
    replace_answer: REPLACE_ANSWER,
    preamble_override: PREAMBLE_OVERRIDE,
    demand_done: DEMAND_DONE,
    lookup_before_run: LOOKUP_BEFORE_RUN,
    two_hooks: TWO_HOOKS,
    retry_twice: RETRY_TWICE,
}

/// The header names the stack in registration order, so the same two
/// hooks in the other order are another program: the golden refuses it
/// with both stacks shown.
#[tokio::test]
async fn the_two_hook_golden_refuses_the_reversed_stack() {
    let reversed = Program {
        hooks: &[Hook::ReplaceAddResult, Hook::PatchAddArgs],
        ..TWO_HOOKS
    };
    let replay = corpus::Replay::open(&reversed);
    let server = replay.tool_server();
    let agent = rig_agent::AgentBuilder::over_bus(
        replay.dispatcher.clone(),
        replay.registrar.clone(),
        reversed.owner,
        replay.model_key.clone(),
    )
    .name(reversed.owner)
    .preamble(TOOLS_PREAMBLE)
    .temperature(0.0)
    .tool_server_handle(server);
    let agent = corpus::with_hooks(agent, reversed.hooks).build();
    let refusal = agent
        .check_replayable(&replay.log)
        .expect_err("a reversed stack is another program")
        .to_string();
    assert!(
        refusal.contains("[\"PatchAddArgs\", \"ReplaceAddResult\"]")
            && refusal.contains("[\"ReplaceAddResult\", \"PatchAddArgs\"]"),
        "{refusal}"
    );
    drop(agent);
    replay.close().await;
}
