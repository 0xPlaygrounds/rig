//! rig inside a Bevy `World`.
//!
//! Two layers. [`bus`]: the effect bus as a plugin — effects are entities,
//! handlers are entities, the driver is a system, an outcome is a
//! component, causality is `ChildOf`, a scene is a checkpoint. And the
//! agent runtime over it — [`agent`] (the run as a graph: agents,
//! documents, utterances, runs, turns, as entities and relationships),
//! [`policy`] (the verbatim strings and the one fold from the graph to the
//! wire `CompletionRequest`), [`systems`] (one system per named set, in the
//! bus's schedule) and [`replay`] (the log header from components). Nothing
//! in this crate awaits, blocks, or holds a future for a host to probe;
//! nothing is copied from `rig-agent`, and a guard refuses its name.
//!
//! The request is a graph in the world and a struct on the wire, with
//! [`policy::fold_request`] as the one function between them. What the
//! agent runtime does today: the run with tools — request assembly, the
//! stream fold, the three output modes and their reprompts, invalid calls
//! as entities with a resolution (fail, ignore, retry, repair, skip), tool
//! calls as effect entities `ChildOf` the turn with the batch as the
//! turn's children, endings, the header. Not yet: hooks as shipped user
//! systems (stage 4 of the programme), memory, retrieval, routing and
//! resume across runs (stage 5).
//!
//! The `bus` module is written as if it were already its own crate (every
//! item `pub` or private to its file, no import from a sibling module, no
//! agent-shaped item, its tests in `tests/bus_*.rs`): the agent modules
//! consume it through its public items only, and it becomes `rig-bevy` by
//! a `git mv` when a second consumer exists.

pub mod agent;
pub mod bus;
pub mod policy;
pub mod replay;
pub mod systems;
