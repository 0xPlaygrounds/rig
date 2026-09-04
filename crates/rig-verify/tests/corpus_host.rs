//! Matrix I: a host's own families over the host's bus.
//!
//! An agent over a host's bus (`over_bus`; Matrix C's ownership cells)
//! shares the bus with the host's own handlers, and a hook can dispatch
//! to them: a `Custom<E>` effect the host defined, the host's embedding
//! model. Those dispatches are the run's — the recorder taps the host's
//! bus — but they are not the agent's row: the required row is what the
//! agent's builder needs (its model, routes, memory, tools, indexes), the
//! signature is what the trace touched. The replay registers the host's
//! handlers from the signature, under the host's keys, before the agent
//! is built; the handler table (the header's first source since #2450)
//! describes the custom kind the row cannot.
//!
//! Both interpreters dispatch the hook's effect at the hook's point: the
//! engine through the hook, the hand driver at the same step. The corpus
//! names the producer's effect type by the same kind label; the payload
//! is data.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | family | `Custom` · `Embed` |
//! | hook point | `on_run_start` · `on_completion_call` · `on_outcome` (after a tool) · `on_run_settled` · start and settled |
//! | dispatches | one · two together |
//! | host policy | `serial_per_handler: true` · `false` |
//! | medium | unary · streamed with events |
//! | the key | served · unserved (the bind refused) |
//!
//! Full cross-product: 2 × 5 × 2 × 2 × 2 × 2 = 160. Recorded: the 12 cells
//! below. Pruned: `Rerank` (the only rerank suites are voyageai's, keyless
//! here, and llamacpp's, local: no cassette can be recorded); the host
//! policy under every point but the double dispatch (one dispatch is
//! served the same way under either); `Embed` at every point but run
//! start and on the anthropic wire (no embedding model there); the
//! streamed twins of the settled, twice and unserved cells (the medium
//! changes the completion's events, not where the hook's dispatch lands).
//!
//! # Cells
//!
//! | golden | producer | shape |
//! |---|---|---|
//! | `anthropic_host_custom_at_start` | anthropic `corpus_host.rs` `custom_at_start_…` | `[Custom, Completion]` |
//! | `anthropic_host_custom_at_completion_call` | `custom_at_completion_call_…` | `[Custom, Completion]` |
//! | `anthropic_host_custom_at_outcome` | `custom_at_outcome_…` | `[Completion, Tool, Custom, Completion]` |
//! | `anthropic_host_custom_at_settled` | `custom_at_settled_…` | `[Completion, Custom]`: the dispatch after the answer is recorded |
//! | `anthropic_host_custom_start_and_settled` | `custom_start_and_settled_…` | `[Custom, Completion, Custom]` |
//! | `anthropic_host_custom_twice_serial` | `custom_twice_serial_…` | `[Custom, Custom, Completion]`, the host serial |
//! | `anthropic_host_custom_twice_concurrent` | `custom_twice_concurrent_…` | the same, the host concurrent |
//! | `anthropic_host_custom_at_start_streamed` | `custom_at_start_streamed_…` | `[Custom, Completion]`, events kept |
//! | `anthropic_host_custom_at_outcome_streamed` | `custom_at_outcome_streamed_…` | `[Completion, Tool, Custom, Completion]`, events kept |
//! | `anthropic_host_custom_unserved` | `custom_unserved_…` | `[Completion]`; the hook's bind refused, the run unaffected |
//! | `openai_host_embed_prompt` | openai `corpus_host.rs` `embed_prompt_…` | `[Embed, Completion]` |
//! | `openai_host_embed_prompt_streamed` | `embed_prompt_streamed_…` | the same, events kept |
//!
//! # What the matrix found
//!
//! - A dispatch from `on_run_settled` is recorded: the hook runs before
//!   the blocking surface returns, the host's recorder is still tapping,
//!   and the record follows the completion that answered the run.
//! - A host-bus golden does not name the host's policy (`bus: None`),
//!   so the serial and concurrent double dispatches are the same log:
//!   the recorder minted the ids at dispatch, in the hook's order, under
//!   either policy. The replay reproduces both under its default.
//! - The replay needs the host's handlers registered: `check_replayable`
//!   walks the signature against the bus (and the agent's row against the
//!   log's handler table) and refuses a signature key no handler serves.
//!   The custom kind is described by the handler table;
//!   `describe_required` cannot name it from the row and does not need to.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{Hook, NOTE_KEY, Program};
use rig_core::effect::{EffectFamily, FamilyDescriptor, HandlerKey};

const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const PROMPT: &str = "Reply with the single word: ready.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";

const HOST: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};
const HOST_WITH_TOOL: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    ..HOST
};

const CUSTOM_AT_START: Program = Program {
    fixture: "anthropic_host_custom_at_start",
    hooks: &[Hook::NoteAtStart],
    ..HOST
};
const CUSTOM_AT_COMPLETION_CALL: Program = Program {
    fixture: "anthropic_host_custom_at_completion_call",
    hooks: &[Hook::NoteAtCompletionCall],
    ..HOST
};
const CUSTOM_AT_OUTCOME: Program = Program {
    fixture: "anthropic_host_custom_at_outcome",
    hooks: &[Hook::NoteAtOutcome],
    ..HOST_WITH_TOOL
};
const CUSTOM_AT_SETTLED: Program = Program {
    fixture: "anthropic_host_custom_at_settled",
    hooks: &[Hook::NoteAtSettled],
    ..HOST
};
const CUSTOM_START_AND_SETTLED: Program = Program {
    fixture: "anthropic_host_custom_start_and_settled",
    hooks: &[Hook::NoteAtStart, Hook::NoteAtSettled],
    ..HOST
};
const CUSTOM_TWICE_SERIAL: Program = Program {
    fixture: "anthropic_host_custom_twice_serial",
    hooks: &[Hook::NoteTwice],
    ..HOST
};
const CUSTOM_TWICE_CONCURRENT: Program = Program {
    fixture: "anthropic_host_custom_twice_concurrent",
    hooks: &[Hook::NoteTwice],
    ..HOST
};
const CUSTOM_AT_START_STREAMED: Program = Program {
    fixture: "anthropic_host_custom_at_start_streamed",
    hooks: &[Hook::NoteAtStart],
    streamed: true,
    ..HOST
};
const CUSTOM_AT_OUTCOME_STREAMED: Program = Program {
    fixture: "anthropic_host_custom_at_outcome_streamed",
    hooks: &[Hook::NoteAtOutcome],
    streamed: true,
    ..HOST_WITH_TOOL
};
const CUSTOM_UNSERVED: Program = Program {
    fixture: "anthropic_host_custom_unserved",
    hooks: &[Hook::NoteUnserved],
    ..HOST
};
const EMBED_PROMPT: Program = Program {
    fixture: "openai_host_embed_prompt",
    hooks: &[Hook::EmbedPrompt],
    max_turns: None,
    ..HOST
};
const EMBED_PROMPT_STREAMED: Program = Program {
    fixture: "openai_host_embed_prompt_streamed",
    hooks: &[Hook::EmbedPrompt],
    max_turns: None,
    streamed: true,
    ..HOST
};

both_interpreters! {
    custom_at_start: CUSTOM_AT_START,
    custom_at_completion_call: CUSTOM_AT_COMPLETION_CALL,
    custom_at_outcome: CUSTOM_AT_OUTCOME,
    custom_at_settled: CUSTOM_AT_SETTLED,
    custom_start_and_settled: CUSTOM_START_AND_SETTLED,
    custom_twice_serial: CUSTOM_TWICE_SERIAL,
    custom_twice_concurrent: CUSTOM_TWICE_CONCURRENT,
    custom_at_start_streamed: CUSTOM_AT_START_STREAMED,
    custom_at_outcome_streamed: CUSTOM_AT_OUTCOME_STREAMED,
    custom_unserved: CUSTOM_UNSERVED,
    embed_prompt: EMBED_PROMPT,
    embed_prompt_streamed: EMBED_PROMPT_STREAMED,
}

/// The custom key is the trace's, not the agent's: in the signature and
/// the handler table (which names its kind), never in the required row.
#[test]
fn custom_key_is_in_the_handler_table_not_the_row() {
    let log = corpus::golden(CUSTOM_AT_START.fixture);
    let key = HandlerKey::from(NOTE_KEY);
    assert!(
        !log.header.required.contains_key(&key),
        "{:?}",
        log.header.required
    );
    assert_eq!(log.header.signature.get(&key), Some(&EffectFamily::Custom));
    let described = log
        .header
        .handlers
        .iter()
        .find(|handler| handler.key == key)
        .expect("the host's handler is in the table");
    assert_eq!(
        described.family,
        FamilyDescriptor::Custom {
            kind: "corpus:note".to_owned()
        }
    );
}

/// The two host policies leave the same trace.
#[test]
fn host_policy_is_invisible_in_the_log() {
    let serial = corpus::golden(CUSTOM_TWICE_SERIAL.fixture);
    let concurrent = corpus::golden(CUSTOM_TWICE_CONCURRENT.fixture);
    assert_eq!(serial.header.bus, None);
    assert_eq!(concurrent.header.bus, None);
    let kinds = |log: &rig_effect_log::EffectLog| {
        log.iter()
            .map(|record| serde_json::to_value(&record.kind).expect("a kind serializes"))
            .collect::<Vec<_>>()
    };
    assert_eq!(kinds(&serial), kinds(&concurrent));
}
