//! Native nested provider completions with actual causal child effects.
use crate::goldens::{
    Hold, LookupArgs, NESTED_PREAMBLE, NESTING_TOOL_KEY, NEVER_KEY, NOTE_KEY, NestedChild, Nesting,
    Note, NoteAck, RELAY_KEY, RelayNote,
};
// Reuse the existing native graph-producing systems, not recorded leaf handlers.
// Here their model child is served by the real provider adapter through cassettes.
// The matrix's corpus (`crate::ecs_matrix::corpus`) includes the same file
// under its own `super`; this copy resolves to the goldens' nesting types.
#[path = "../../../corpus/world_nesting.rs"]
#[allow(dead_code, clippy::duplicate_mod)]
mod nesting;
