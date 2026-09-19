//! The ECS contract matrix: one shape, six copies.
//!
//! Every Anthropic `ecs_*` family (`tests/providers/anthropic/cassette/`)
//! pins one section of `crates/rig-ecs/CONTRACT.md` on real provider bytes.
//! This module writes those cells once, wire-neutrally, as data
//! ([`cells`]: the rig-cassette corpus's `Program` table plus what a live
//! cell needs beside it — the tools it grants, the store it remembers in,
//! the bus it is served over) and two drivers over them:
//!
//! - [`agent::run_agent`], the rig-agent producer: builds the program on
//!   the builder, runs it over a cassette and writes the golden
//!   (`crate::goldens::golden_effects`);
//! - [`world::run_world`], the world cell: the same program as an agent
//!   graph in a Bevy `World`, its hooks as systems (the corpus's
//!   `world_hooks`), served by the real adapters over the same cassette,
//!   its log compared to the producer's golden
//!   (`crate::ecs_goldens::golden_effects`), its graph inspected, its run
//!   despawned, and — where the cell names a cut — saved as a scene at the
//!   cut and resumed in a fresh world over replayers of the log's tail.
//!
//! The failure rows ([`faults`]) are the same shape: a cell names the
//! fault it drives, the per-wire file supplies the transport (a cassette,
//! or the sequenced transport over labelled frames) and the wire's own
//! facts, and the two drivers assert the failure, the record and the
//! history beside the ending.
//!
//! A per-provider file (`tests/providers/<p>/cassette/{corpus,ecs}_matrix*.rs`)
//! holds only the scenario literals (the cassette census reads them off
//! the call sites), the wire's models and the wire-specific `#[ignore]`
//! reasons.
// The corpus is every test target's different subset; its own inner
// `#![allow(dead_code)]` covers it.
#[path = "../corpus/mod.rs"]
pub(crate) mod corpus;

#[path = "ecs_matrix/agent.rs"]
pub(crate) mod agent;
#[path = "ecs_matrix/cells.rs"]
pub(crate) mod cells;
#[path = "ecs_matrix/checkpoint.rs"]
pub(crate) mod checkpoint;
#[path = "ecs_matrix/checkpoint_world.rs"]
pub(crate) mod checkpoint_world;
#[path = "ecs_matrix/extra.rs"]
pub(crate) mod extra;
#[path = "ecs_matrix/faults.rs"]
pub(crate) mod faults;
#[path = "ecs_matrix/image.rs"]
pub(crate) mod image;
#[path = "ecs_matrix/long_loop.rs"]
pub(crate) mod long_loop;
#[path = "ecs_matrix/long_loop_world.rs"]
pub(crate) mod long_loop_world;
#[path = "ecs_matrix/reasoning.rs"]
pub(crate) mod reasoning;
#[path = "ecs_matrix/stream_delivery.rs"]
pub(crate) mod stream_delivery;
#[path = "ecs_matrix/world.rs"]
pub(crate) mod world;

use rig_agent::completion::CompletionModel;

use rig_core::driver::Bound;
use rig_core::http_client::BoxedHttpClient;
use rig_core::providers::anthropic::wire as anthropic;
use rig_core::providers::gemini::completion as gemini;
use rig_core::providers::openai::responses_api::wire as responses;
use rig_core::providers::openai::wire as openai;
use rig_core::wire::Secret;

use cells::Cell;
use corpus::Program;
use rig_core::providers::registry::ProviderConfig;
use rig_ecs::bus::ProviderBinding;

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
}

impl<M: CompletionModel + Clone + 'static> Wire<M> {
    /// The cell's program on this wire.
    pub(crate) fn program(&self, cell: &Cell) -> Program {
        let mut program = cell.program;
        if program.temperature == Some(0.0) {
            program.temperature = self.temperature;
        }
        if let Some(nesting) = program.nesting.as_mut() {
            nesting.no_temperature = self.temperature.is_none();
        }
        if program.additional_params.is_none() {
            program.additional_params = self.additional_params;
        }
        match cell.thinking {
            cells::Thinking::On => {
                program.additional_params = Some(self.thinking.params(true));
            }
            cells::Thinking::SecondTurnOnly => {
                program.additional_params = Some(self.thinking.params(false));
                program.thinking_params = Some(self.thinking.params(true));
            }
            cells::Thinking::Off if cell.explicit_thinking_off => {
                program.additional_params = Some(self.thinking.params(false));
            }
            cells::Thinking::Off => {}
        }
        program
    }

    /// The route model, for a cell that selects one.
    pub(crate) fn route(&self) -> M {
        self.route.clone().expect("the wire names a route model")
    }

    /// The default model as data (CONTRACT §12): the `ProviderBinding` a
    /// head world binds `golden/model:default` through instead of a
    /// hand-registered adapter, beside what the harness `Materializer`
    /// resolves and sends through — the head wire's own key (under the
    /// reference [`CASSETTE_CREDENTIAL`]) and its own socket, so the
    /// materialized model is the head model rebuilt: same model id, same
    /// configuration, same credential, same cassette. `None` for a model the
    /// harness cannot describe (a wrapped transport, a mock): that wire
    /// stays hand-registered.
    ///
    /// A bound wire already holds the provider's configuration, so the
    /// binding carries *that* — dialect, host and every typed option
    /// included — rather than a re-description of it. No dialect is
    /// undescribable any more: what used to need an enum variant per vendor
    /// is now the configuration itself. The one thing the arms add is the
    /// route, because a concrete wire says which endpoint it is and a
    /// configuration whose dialect flagships the other one would otherwise
    /// rebuild the wrong grammar.
    pub(crate) fn binding(&self) -> Option<WireBinding> {
        let model: &dyn std::any::Any = &self.model;
        let key = format!("{OWNER}/model:default");
        let describe = |config: ProviderConfig,
                        model_id: &str,
                        api_key: &Secret,
                        transport: &BoxedHttpClient| {
            Some(WireBinding {
                // The credential is a *reference* in the world: the
                // configuration the binding holds carries none, and the
                // harness resolver supplies the head wire's key at
                // materialization.
                binding: ProviderBinding::configured(
                    key.clone(),
                    config.with_credential(""),
                    model_id,
                    CASSETTE_CREDENTIAL,
                )
                .labelled("default"),
                api_key: api_key.expose().to_owned(),
                transport: transport.clone(),
            })
        };
        if let Some(bound) = model.downcast_ref::<Bound<anthropic::Messages, BoxedHttpClient>>() {
            return describe(
                ProviderConfig::Anthropic(bound.wire.provider.clone()),
                &bound.wire.model,
                &bound.wire.provider.api_key,
                &bound.http,
            );
        }
        let chat = |wire: &openai::Chat, http: &BoxedHttpClient| {
            describe(
                ProviderConfig::OpenAi(wire.provider.clone().with_route(openai::Route::Chat)),
                &wire.model,
                &wire.provider.api_key,
                http,
            )
        };
        let responses = |wire: &responses::Responses, http: &BoxedHttpClient| {
            describe(
                ProviderConfig::OpenAi(wire.provider.clone().with_route(openai::Route::Responses)),
                &wire.model,
                &wire.provider.api_key,
                http,
            )
        };
        if let Some(bound) = model.downcast_ref::<Bound<openai::Chat, BoxedHttpClient>>() {
            return chat(&bound.wire, &bound.http);
        }
        if let Some(bound) = model.downcast_ref::<Bound<responses::Responses, BoxedHttpClient>>() {
            return responses(&bound.wire, &bound.http);
        }
        if let Some(bound) = model.downcast_ref::<Bound<openai::OpenAiWire, BoxedHttpClient>>() {
            return match &bound.wire {
                openai::OpenAiWire::Chat(wire) => chat(wire, &bound.http),
                openai::OpenAiWire::Responses(wire) => responses(wire, &bound.http),
            };
        }
        if let Some(bound) = model.downcast_ref::<Bound<gemini::GenerateContent, BoxedHttpClient>>()
        {
            return describe(
                ProviderConfig::Gemini(bound.wire.provider.clone()),
                &bound.wire.model,
                &bound.wire.provider.api_key,
                &bound.http,
            );
        }
        None
    }
}

/// The credential reference every wire binding names; the harness
/// resolver maps it to the cassette's key.
pub(crate) const CASSETTE_CREDENTIAL: &str = "cassette";

/// A wire's default model as a binding, with what the harness
/// `Materializer` needs to rebuild the head model from it.
pub(crate) struct WireBinding {
    /// The binding, under `golden/model:default`.
    pub(crate) binding: ProviderBinding,
    /// The head wire's key: what `cassette` resolves to.
    pub(crate) api_key: String,
    /// The head wire's socket: what the factory hands every binding.
    pub(crate) transport: BoxedHttpClient,
}
