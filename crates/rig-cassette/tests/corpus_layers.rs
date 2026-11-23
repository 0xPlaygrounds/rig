//! Matrix P: layers.
//!
//! Interception is handler composition: a `Layer` wraps a handler in an `Intercept` that decides
//! before (`Proceed`, `Patch`, `Deny`) and after (`Keep`, `Replace`), and registers under the
//! inner descriptor with its name in `layers`. Decisions are program, never record, because the
//! recorder taps the innermost hop. A denial leaves no record, a replacement leaves the
//! handler's real answer in it, and a replay re-makes the decision with the same layer
//! registered (`Program::layers`). The header's `hooks` list names the hooks, then the layers
//! in handler-table order, outermost first within a key (`Agent::program_names`), so a replay
//! under another stack is refused by the hook-stack check.
//!
//! The equivalence cells are the evidence that `Layer`'s decisions are expressive enough:
//! hand-written layers reproduce four hook goldens' records byte for byte, without rig-agent's
//! stack. The other cells cover a host's own policy over its bus, a layer beneath the agent's
//! hook patch, a suspended decision that is approved, denied or cancelled, a wrong-family
//! patch, and a streamed replacement that ends cancelled. A wrong-family patch does not fail
//! the run: the engine turns it into a failed tool result the model sees.

use crate::corpus;
use rig_cassette::agent::AgentReplayExt;

use corpus::{Answer, Ending, Hook, LayerAt, LayerKind, LayerSpec, Program};
use rig_core::effect::EffectFamily;
use rig_core::error::ErrorKind;

const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const PROMPT: &str = "Reply with the single word: ready.";

const fn at_tool(layer: LayerKind) -> LayerSpec {
    LayerSpec {
        at: LayerAt::Tool,
        layer,
    }
}

const TOOLS: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};

const DENY_TOOL: Program = Program {
    fixture: "anthropic_layers_deny_tool",
    layers: &[at_tool(LayerKind::DenyAdd)],
    ..TOOLS
};
const PATCH_TOOL_ARGS: Program = Program {
    fixture: "anthropic_layers_patch_tool_args",
    layers: &[at_tool(LayerKind::PatchAddArgs)],
    ..TOOLS
};
const REPLACE_TOOL_RESULT: Program = Program {
    fixture: "anthropic_layers_replace_tool_result",
    layers: &[at_tool(LayerKind::ReplaceAddResult)],
    ..TOOLS
};
const TWO_LAYERS: Program = Program {
    fixture: "anthropic_layers_two_layers",
    layers: &[
        at_tool(LayerKind::PatchAddArgs),
        at_tool(LayerKind::ReplaceAddResult),
    ],
    ..TOOLS
};
const HOST_DENY_OVER_HOST_BUS: Program = Program {
    fixture: "anthropic_layers_host_deny_over_host_bus",
    layers: &[at_tool(LayerKind::DenyAdd)],
    ..TOOLS
};
const PATCH_BENEATH_HOOK_PATCH: Program = Program {
    fixture: "anthropic_layers_patch_beneath_hook_patch",
    hooks: &[Hook::PatchAddArgs],
    layers: &[at_tool(LayerKind::PatchAgain)],
    ..TOOLS
};
const SUSPEND_APPROVE: Program = Program {
    fixture: "mock_layers_suspend_approve",
    layers: &[at_tool(LayerKind::Approval(Answer::Approve))],
    ..TOOLS
};
const SUSPEND_DENY: Program = Program {
    fixture: "mock_layers_suspend_deny",
    layers: &[at_tool(LayerKind::Approval(Answer::Deny))],
    ..TOOLS
};
const SUSPEND_CANCELLED: Program = Program {
    fixture: "mock_layers_suspend_cancelled",
    layers: &[at_tool(LayerKind::Approval(Answer::Never))],
    cancel_when_reached: true,
    ..TOOLS
};
const WRONG_FAMILY_PATCH: Program = Program {
    fixture: "mock_layers_wrong_family_patch",
    layers: &[at_tool(LayerKind::WrongFamily)],
    ..TOOLS
};
const REPLACE_STREAMED_CANCELLED: Program = Program {
    fixture: "mock_layers_replace_streamed_cancelled",
    preamble: Some(BASIC_PREAMBLE),
    prompt: PROMPT,
    temperature: Some(0.0),
    streamed: true,
    ending: Ending::Failed(ErrorKind::Cancelled),
    layers: &[LayerSpec {
        at: LayerAt::Model,
        layer: LayerKind::CancelStream,
    }],
    ..Program::DEFAULT
};

both_interpreters! {
    deny_tool: DENY_TOOL,
    patch_tool_args: PATCH_TOOL_ARGS,
    replace_tool_result: REPLACE_TOOL_RESULT,
    two_layers: TWO_LAYERS,
    host_deny_over_host_bus: HOST_DENY_OVER_HOST_BUS,
    patch_beneath_hook_patch: PATCH_BENEATH_HOOK_PATCH,
    suspend_approve: SUSPEND_APPROVE,
    suspend_deny: SUSPEND_DENY,
    suspend_cancelled: SUSPEND_CANCELLED,
    wrong_family_patch: WRONG_FAMILY_PATCH,
    replace_streamed_cancelled: REPLACE_STREAMED_CANCELLED,
}

/// The equivalence row: the same decisions, made by layers, record the
/// same bytes as the hooks did — every record, and every header field but
/// `hooks` (the names) and the handler table's `layers`.
#[test]
fn hand_written_layers_reproduce_the_hook_goldens_byte_for_byte() {
    let pairs = [
        (
            &DENY_TOOL,
            "anthropic_hooks_deny_tool",
            vec!["DenyAddLayer"],
        ),
        (
            &PATCH_TOOL_ARGS,
            "anthropic_hooks_patch_tool_args",
            vec!["PatchAddArgsLayer"],
        ),
        (
            &REPLACE_TOOL_RESULT,
            "anthropic_hooks_replace_tool_result",
            vec!["ReplaceAddResultLayer"],
        ),
        (
            &TWO_LAYERS,
            "anthropic_hooks_two_hooks",
            vec!["PatchAddArgsLayer", "ReplaceAddResultLayer"],
        ),
    ];
    for (cell, hook_golden, names) in pairs {
        let layered = corpus::golden(cell.fixture);
        let hooked = corpus::golden(hook_golden);
        let records = |log: &rig_cassette::effect_log::EffectLog| {
            log.iter().map(corpus::as_data).collect::<Vec<_>>()
        };
        assert_eq!(records(&layered), records(&hooked), "{}", cell.fixture);
        assert_eq!(layered.header.hooks, names, "{}", cell.fixture);
        assert_eq!(layered.header.run_spec, hooked.header.run_spec);
        assert_eq!(layered.header.required, hooked.header.required);
        assert_eq!(layered.header.signature, hooked.header.signature);
        let without_layers = |log: &rig_cassette::effect_log::EffectLog| {
            let mut handlers = log.header.handlers.clone();
            for handler in &mut handlers {
                handler.layers.clear();
            }
            handlers
        };
        assert_eq!(without_layers(&layered), without_layers(&hooked));
        let add = layered
            .header
            .handlers
            .iter()
            .find(|handler| handler.key.as_str().ends_with("/tool:add#0"))
            .expect("the tool's descriptor");
        assert_eq!(add.layers, names, "the handler table names the layers");
    }
}

/// A replay under another layer stack is refused, as under another hook
/// stack: the header's `hooks` names the layers.
#[tokio::test]
async fn a_replay_under_another_layer_stack_is_refused() {
    let replay = corpus::Replay::open(&DENY_TOOL);
    let server = replay.tool_server_for(&PATCH_TOOL_ARGS);
    let agent = corpus::build_agent_unchecked(&replay, &PATCH_TOOL_ARGS, server, &replay.log);
    let refusal = agent
        .check_replayable(&replay.log)
        .expect_err("another layer is another program")
        .to_string();
    assert!(
        refusal.contains("DenyAddLayer") && refusal.contains("PatchAddArgsLayer"),
        "{refusal}"
    );
    drop(agent);
    replay.close().await;
}

/// The record is what the innermost handler answered, never a layer's
/// verdict; a denial leaves none.
#[test]
fn the_record_is_the_handlers_answer() {
    let replaced = corpus::golden(REPLACE_TOOL_RESULT.fixture);
    let Ok(rig_core::effect::Answered::ToolResult { result, .. }) = &replaced[1].outcome else {
        panic!("a tool result: {:?}", replaced[1].outcome);
    };
    assert_eq!(
        result.output().render(),
        "42",
        "the tool's, not the layer's 99"
    );
    let patched = corpus::golden(PATCH_BENEATH_HOOK_PATCH.fixture);
    let rig_core::effect::EffectKind::ToolCall { args, .. } = &patched[1].kind else {
        panic!("a tool call: {:?}", patched[1].kind);
    };
    assert_eq!(
        args,
        corpus::PATCHED_AGAIN_ARGS,
        "the innermost patch is what was served"
    );
    for cell in [
        &DENY_TOOL,
        &HOST_DENY_OVER_HOST_BUS,
        &SUSPEND_DENY,
        &WRONG_FAMILY_PATCH,
    ] {
        let log = corpus::golden(cell.fixture);
        assert!(
            log.iter()
                .all(|record| record.kind.family() != EffectFamily::Tool),
            "{}: no record for a decision before the handler",
            cell.fixture
        );
    }
    let suspended = corpus::golden(SUSPEND_CANCELLED.fixture);
    assert!(
        matches!(&suspended[1].outcome, Err(report) if report.kind == ErrorKind::Cancelled),
        "cancelled by the consumer's drop, not the layer"
    );
    let streamed = corpus::golden(REPLACE_STREAMED_CANCELLED.fixture);
    assert!(streamed[0].outcome.is_ok() && streamed[0].events.is_some());
}
