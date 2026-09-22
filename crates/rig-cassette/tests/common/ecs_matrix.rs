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
//!   its record and graph asserted against the cell, its run despawned,
//!   and — where the cell names a cut — saved as a scene at the
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
#[path = "ecs_matrix/long_tasks.rs"]
pub(crate) mod long_tasks;
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
use rig_core::serve::ErasedHandler;

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

    /// Return a credential-free provider recipe only when it reproduces the
    /// complete model wire. The host retains the credential and transport.
    /// Return `None` for custom transports or model options absent from
    /// `ProviderConfig`; callers must rebind the supplied adapter instead.
    pub(crate) fn binding(&self) -> Option<WireBinding> {
        let model: &dyn std::any::Any = &self.model;
        let describe = |config: ProviderConfig,
                        model_id: &str,
                        api_key: &Secret,
                        transport: &BoxedHttpClient| {
            Some(WireBinding {
                // The credential is the host's: the configuration that
                // travels as data carries none, and the host supplies the
                // head wire's key when it builds the client.
                config: config.with_credential(""),
                model: model_id.to_owned(),
                api_key: api_key.expose().to_owned(),
                transport: transport.clone(),
            })
        };
        if let Some(bound) = model.downcast_ref::<Bound<anthropic::Messages, BoxedHttpClient>>() {
            // ProviderConfig does not retain model-level cache and tool options.
            // Keep the original adapter when rebuilding would discard them.
            if bound.wire != bound.wire.provider.messages(bound.wire.model.clone()) {
                return None;
            }
            return describe(
                ProviderConfig::Anthropic(bound.wire.provider.clone()),
                &bound.wire.model,
                &bound.wire.provider.api_key,
                &bound.http,
            );
        }
        let chat = |wire: &openai::Chat, http: &BoxedHttpClient| {
            if *wire != wire.provider.chat(wire.model.clone()) {
                return None;
            }
            describe(
                ProviderConfig::OpenAi(wire.provider.clone().with_route(openai::Route::Chat)),
                &wire.model,
                &wire.provider.api_key,
                http,
            )
        };
        let responses = |wire: &responses::Responses, http: &BoxedHttpClient| {
            if *wire != wire.provider.responses(wire.model.clone()) {
                return None;
            }
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
            if bound.wire
                != bound
                    .wire
                    .provider
                    .generate_content(bound.wire.model.clone())
            {
                return None;
            }
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

/// The adapter label the default model's handler carries, on both sides:
/// the hand-registered adapter's and the one a host builds from [`WireBinding`].
pub(crate) const DEFAULT_LABEL: &str = "default";

/// A wire's default model as host-owned configuration, with the two things
/// a host needs beside it to rebuild the head model: the credential and the
/// socket. Neither is ever written to a world, so neither is ever saved.
pub(crate) struct WireBinding {
    /// The provider configuration, credential-free: the part that is data.
    pub(crate) config: ProviderConfig,
    /// The provider's own model identifier.
    pub(crate) model: String,
    /// The head wire's key, supplied at construction by the host alone.
    pub(crate) api_key: String,
    /// The head wire's socket: what the host binds the rebuilt client to.
    pub(crate) transport: BoxedHttpClient,
}

impl WireBinding {
    /// The completion handler this recipe describes, built here — in the
    /// host — and served under the harness IO runtime, as a hand-registered
    /// adapter is. Replay never calls this: a resumed world is handed the
    /// handlers its checkpoint requires, live or recorded, by whoever opens it.
    pub(crate) fn handler(&self, runtime: &tokio::runtime::Handle) -> ErasedHandler {
        let handler = self
            .config
            .clone()
            .with_credential(self.api_key.clone())
            .completion_handler(DEFAULT_LABEL, &self.model, self.transport.clone());
        ErasedHandler::new(crate::ecs_agent::RuntimeHandler {
            inner: std::sync::Arc::new(handler),
            runtime: runtime.clone(),
        })
    }
}
