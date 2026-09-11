//! rig inside a Bevy `World`.
//!
//! Start with owned [`commands::Agent`] and [`commands::Prompt`] values. They
//! create ordinary entities and components; the world remains authoritative.
//! Register a completion handler, configure an agent, and submit a prompt:
//!
//! ```
//! use bevy_app::{App, Update};
//! use bevy_ecs::entity::Entity;
//! use rig_core::serve::Serve;
//! use rig_ecs::{
//!     bus::{Handlers, run_to_quiescence},
//!     commands::{Agent, Prompt, install},
//! };
//!
//! fn start(model: impl Serve + 'static) -> Result<(App, Entity), Box<dyn std::error::Error>> {
//!     let mut app = App::new();
//!     install(app.world_mut(), Default::default())?;
//!     app.add_systems(Update, run_to_quiescence);
//!     let model = Handlers::register_in(app.world_mut(), "model", model)?;
//!     let agent = Agent::new(model).preamble("Be concise.").spawn(app.world_mut())?;
//!     let run = Prompt::new(agent, "Hello").spawn(app.world_mut())?;
//!     Ok((app, run))
//! }
//! ```
//!
//! Pass a completion-family handler to `start`. The host calls `app.update()`
//! to collect work and uses [`inspect::inspect`] with the returned run ID to
//! read its status, answer, or failure. See the runnable `agent_with_tools`
//! example for a complete offline host. In systems, [`commands::RigCommands`]
//! provides deferred operations; drain [`commands::CommandFailures`] after
//! commands apply. Ordinary system parameters need no handwritten lifetimes:
//!
//! ```
//! use bevy_ecs::prelude::*;
//! use rig_ecs::{commands::{Prompt, RigCommands}, inspect::RunView};
//!
//! fn submit(commands: &mut Commands, agent: Entity) -> Entity {
//!     commands.prompt(Prompt::new(agent, "Hello"))
//! }
//!
//! fn show_answers(runs: Query<RunView>) {
//!     for run in &runs {
//!         if let Some(answer) = run.answer() {
//!             println!("{answer}");
//!         }
//!     }
//! }
//! ```
//!
//! Two layers. [`bus`]: the effect bus installed in the world — effects are entities,
//! handlers are entities, the driver is a system, an outcome is a
//! component, causality is `ChildOf`, a scene is a checkpoint. And the
//! agent runtime over it — [`agent`] (the run as a graph: agents,
//! documents, utterances, runs, turns, as entities and relationships),
//! [`policy`] (the verbatim strings and the one fold from the graph to the
//! wire `CompletionRequest`), [`systems`] (one system per named set, in the
//! bus's schedule) and the optional `replay` module (the log header from
//! components). Bus-owned tasks drive handler futures without blocking the
//! host schedule. Applications inspect components rather than owning workers;
//! nothing is copied from `rig-agent`, and a guard refuses its name.
//!
//! The request is a graph in the world and a struct on the wire, with
//! [`policy::fold_request`] as the one function between them. What the
//! agent runtime does today: the run with tools — request assembly, the
//! stream fold, the three output modes and their reprompts, invalid calls
//! as entities with a resolution (fail, ignore, retry, repair, skip), tool
//! calls as effect entities `ChildOf` the turn with the batch as the
//! turn's children, endings, the header; and steering as components a
//! user system writes — [`agent::Cancelled`], [`agent::Retry`],
//! [`agent::RequestPatch`], [`agent::Resolution`], `UsesModel` — read by
//! the library at the next set; memory as the graph (an agent that
//! [`agent::Remembers`] loads its [`agent::Conversation`] before the first
//! turn and appends what the run said at the settle); retrieval as
//! [`agent::Retrieves`] links whose effects run before every fold and
//! attach documents and tools to the turn; resume as a scene load
//! ([`agent::scene::WorldScene::save`] with the run's effects, in flight or
//! answered, beside the graph, and [`agent::scene::WorldScene::load`] in a fresh
//! world over the log's tail).
//!
//! [`prelude`] includes construction commands, run inspection, streaming text,
//! scheduling sets, and common components. Behind features: `reflect` — supported
//! graph components derive `Reflect`, `reflect::install_reflect` registers them,
//! `reflect::ReflectedScene` is the world as reflected data beside the
//! serde scene.
//!
//! The `bus` module is independent of agent modules. Its supported components,
//! registration operations, and schedule sets are public; worker bookkeeping
//! remains internal. An executable public consumer tests this boundary. Additional consumers supply
//! evidence for a later crate-boundary decision; they do not require an
//! automatic extraction into a separate `rig-bevy` crate.

// Shared integration fixtures also serve private lifecycle contract tests.
#[cfg(test)]
extern crate self as rig_ecs;

pub mod agent;
pub mod approval;
#[cfg(feature = "assets")]
pub mod assets;
pub mod bus;
pub mod commands;
pub mod inspect;
pub mod lifecycle;
pub mod policy;
pub mod prelude;
#[cfg(feature = "reflect")]
pub mod reflect;
#[cfg(feature = "replay")]
pub mod replay;
pub mod stream;
pub mod systems;
