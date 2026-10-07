//! The agent corpus matrix: one shape, six copies.
//!
//! This module writes the cells once, wire-neutrally, as data ([`cells`]:
//! the rig-cassette corpus's `Program` table plus what a live cell needs
//! beside it — the tools it grants, the store it remembers in, the bus it
//! is served over) and [`agent::run_agent`], the rig-agent producer that
//! builds the program on the builder, runs it over a cassette and writes the
//! golden (`crate::goldens::golden_effects`).
//!
//! The failure rows ([`faults`]) are the same shape: a cell names the
//! fault it drives, the per-wire file supplies its cassettes and the wire's
//! own facts, [`faults::Scripted`] serves the scripted rows over labelled
//! frames, and the driver asserts the failure, the record and the history
//! beside the ending.
//!
//! A per-provider file (`tests/providers/<p>/cassette/corpus_matrix*.rs`)
//! holds only the scenario literals (the cassette census reads them off
//! the call sites), the wire's models and the wire-specific `#[ignore]`
//! reasons.
#[allow(
    dead_code,
    reason = "each target uses a different subset of the corpus"
)]
#[path = "../corpus/mod.rs"]
pub(crate) mod corpus;

#[path = "corpus_matrix/agent.rs"]
pub(crate) mod agent;
#[path = "corpus_matrix/cells.rs"]
pub(crate) mod cells;
#[path = "corpus_matrix/checkpoint.rs"]
pub(crate) mod checkpoint;
#[path = "corpus_matrix/faults.rs"]
pub(crate) mod faults;
#[path = "corpus_matrix/image.rs"]
pub(crate) mod image;
#[path = "corpus_matrix/long_loop.rs"]
pub(crate) mod long_loop;
#[path = "corpus_matrix/reasoning.rs"]
pub(crate) mod reasoning;

use cells::Cell;
use corpus::Program;

/// The owner every producer names itself: the golden's keys are
/// `golden/model:default`, `golden/tool:<name>#<n>`, `golden/memory`.
pub(crate) const OWNER: &str = "golden";

/// A wire: the models a cell is served by, and what the wire cannot take.
pub(crate) struct Wire<M> {
    /// How this wire renders the matrix's thinking control.
    pub(crate) thinking: cells::ThinkingWire,
    /// The default model, under `golden/model:default`.
    pub(crate) model: M,
    /// The route (`golden/model:fast` or `golden/model:late`), where a cell
    /// selects one.
    pub(crate) route: Option<M>,
    /// What the cell's `temperature: Some(0.0)` becomes on this wire: a
    /// model that takes only its default temperature (the gpt-5 family)
    /// gets `None`, so the request carries no temperature at all.
    pub(crate) temperature: Option<f64>,
    /// Request parameters every cell of this wire carries when the cell
    /// names none (a thinking model asked not to think).
    pub(crate) additional_params: Option<fn() -> serde_json::Value>,
    /// Typed options every cell of this wire carries when the cell names
    /// no parameters, as `additional_params`.
    pub(crate) options: Option<fn() -> corpus::TypedOptions>,
}

impl<W, Tr> Wire<rig::driver::Model<W, Tr>>
where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    Tr: rig::driver::Transport<W>,
{
    /// The cell's program on this wire.
    pub(crate) fn program(&self, cell: &Cell) -> Program {
        let mut program = cell.program;
        if program.temperature == Some(0.0) {
            program.temperature = self.temperature;
        }
        if let Some(nesting) = program.nesting.as_mut() {
            nesting.no_temperature = self.temperature.is_none();
        }
        if program.additional_params.is_none() && program.options.is_none() {
            program.additional_params = self.additional_params;
            program.options = self.options;
        }
        // A thinking cell's parameters replace the wire's, typed ones too.
        match cell.thinking {
            cells::Thinking::On => {
                program.additional_params = Some(self.thinking.params(true));
                program.options = None;
            }
            cells::Thinking::SecondTurnOnly => {
                program.additional_params = Some(self.thinking.params(false));
                program.options = None;
                program.thinking_params = Some(self.thinking.params(true));
            }
            cells::Thinking::Off if cell.explicit_thinking_off => {
                program.additional_params = Some(self.thinking.params(false));
                program.options = None;
            }
            cells::Thinking::Off => {}
        }
        program
    }

    /// The route model, for a cell that selects one.
    pub(crate) fn route(&self) -> rig::driver::Model<W, Tr> {
        self.route.clone().expect("the wire names a route model")
    }
}
