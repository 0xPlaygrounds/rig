//! Matrix T: the leftovers — the `Denied` cells.
//!
//! `ErrorKind::Denied` is what a layer's `Decision::deny` resolves, and
//! every consumer maps it its own way: on a tool the model sees the
//! skipped result (as it does for a hook's denial); on a completion the
//! run fails with the report; on a memory `Load` the run fails at the
//! record (`MemoryError::Policy`); on a custom effect from a hook, the
//! hook sees `Denied` and decides. None of them is a record: the layer
//! decided before any handler served the dispatch.
//!
//! The matrix's L2–L4 cells (the recorder's slot search, an
//! unserializable custom effect from a hook, required keys of every family
//! described from the handler table or refused by name) are the module's
//! next commit.
//!
//! # Cells
//!
//! | golden | producer | shape |
//! |---|---|---|
//! | `mock_leftovers_denied_tool` | `tests/core/golden_leftovers.rs` `denied_tool_…` | `[Completion, Completion]`: the skipped result in the second request |
//! | `mock_leftovers_denied_completion` | `denied_completion_…` | `[]`: the run fails with `Denied` before any record |
//! | `mock_leftovers_denied_memory_load` | `denied_memory_load_…` | `[]`: the run fails at the load, `MemoryError::Policy` |
//! | `mock_leftovers_denied_custom_from_hook` | `denied_custom_from_hook_…` | `[Completion]`: the hook saw `Denied` and went on |

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{Ending, Hook, LayerAt, LayerKind, LayerSpec, Program};
use rig_core::error::ErrorKind;

const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const PROMPT: &str = "Reply with the single word: ready.";

const fn deny(at: LayerAt) -> LayerSpec {
    LayerSpec {
        at,
        layer: LayerKind::DenyAll,
    }
}

const DENIED_TOOL: Program = Program {
    fixture: "mock_leftovers_denied_tool",
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    layers: &[deny(LayerAt::Tool)],
    ..Program::DEFAULT
};
const DENIED_COMPLETION: Program = Program {
    fixture: "mock_leftovers_denied_completion",
    preamble: Some(BASIC_PREAMBLE),
    prompt: PROMPT,
    temperature: Some(0.0),
    ending: Ending::Failed(ErrorKind::Denied),
    layers: &[deny(LayerAt::Model)],
    ..Program::DEFAULT
};
const DENIED_MEMORY_LOAD: Program = Program {
    fixture: "mock_leftovers_denied_memory_load",
    preamble: Some(BASIC_PREAMBLE),
    prompt: PROMPT,
    temperature: Some(0.0),
    conversation: Some(corpus::CONVERSATION),
    ending: Ending::MemoryError,
    layers: &[deny(LayerAt::Memory)],
    ..Program::DEFAULT
};
const DENIED_CUSTOM_FROM_HOOK: Program = Program {
    fixture: "mock_leftovers_denied_custom_from_hook",
    preamble: Some(BASIC_PREAMBLE),
    prompt: PROMPT,
    temperature: Some(0.0),
    hooks: &[Hook::NoteDeniedAtStart],
    layers: &[deny(LayerAt::Note)],
    ..Program::DEFAULT
};

both_interpreters! {
    denied_tool: DENIED_TOOL,
    denied_completion: DENIED_COMPLETION,
    denied_memory_load: DENIED_MEMORY_LOAD,
    denied_custom_from_hook: DENIED_CUSTOM_FROM_HOOK,
}

/// A denial before the handler is never a record.
#[test]
fn a_denial_is_never_a_record() {
    assert_eq!(corpus::golden(DENIED_TOOL.fixture).len(), 2);
    assert_eq!(corpus::golden(DENIED_COMPLETION.fixture).len(), 0);
    assert_eq!(corpus::golden(DENIED_MEMORY_LOAD.fixture).len(), 0);
    assert_eq!(corpus::golden(DENIED_CUSTOM_FROM_HOOK.fixture).len(), 1);
    // The keys the layers sat on are in the handler table with the layer
    // named, whether or not they were ever recorded.
    for (cell, key) in [
        (&DENIED_COMPLETION, "golden/model:default"),
        (&DENIED_MEMORY_LOAD, "golden/memory"),
        (&DENIED_CUSTOM_FROM_HOOK, corpus::NOTE_KEY),
    ] {
        let log = corpus::golden(cell.fixture);
        let handler = log
            .header
            .handlers
            .iter()
            .find(|handler| handler.key.as_str() == key)
            .unwrap_or_else(|| panic!("{}: `{key}` in the handler table", cell.fixture));
        assert_eq!(handler.layers, ["DenyAllLayer"]);
    }
}
