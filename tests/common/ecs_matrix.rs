//! The ECS contract matrix: one shape, six copies.
//!
//! Every Anthropic `ecs_*` family (`tests/providers/anthropic/cassette/`)
//! pins one section of `crates/rig-ecs/CONTRACT.md` on real provider bytes.
//! This module writes those cells once, wire-neutrally, as data
//! ([`cells`]: the rig-verify corpus's `Program` table plus what a live
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
#[path = "../../crates/rig-verify/tests/corpus/mod.rs"]
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

use rig_core::http_client::BoxedHttpClient;

use rig_ecs::bus::{ProviderBinding, ProviderKind};

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
    /// resolves and sends through — the head client's own key (under the
    /// reference [`CASSETTE_CREDENTIAL`]) and its own transport, so the
    /// materialized client is the head client rebuilt: same model id, same
    /// base URL, same headers, same cassette. `None` for a model the
    /// harness cannot describe (a wrapped transport, a mock): that wire
    /// stays hand-registered.
    pub(crate) fn binding(&self) -> Option<WireBinding> {
        let model: &dyn std::any::Any = &self.model;
        let key = format!("{OWNER}/model:default");
        let describe = |kind: ProviderKind,
                        model_id: &str,
                        client_base_url: &str,
                        api_key: Option<String>,
                        transport: &BoxedHttpClient| {
            Some(WireBinding {
                binding: ProviderBinding::new(key.clone(), kind, model_id, CASSETTE_CREDENTIAL)
                    .labelled("default")
                    .at(client_base_url),
                api_key: api_key?,
                transport: transport.clone(),
            })
        };
        if let Some(model) = model.downcast_ref::<rig_core::providers::anthropic::CompletionModel>()
        {
            let client = model.client();
            return describe(
                ProviderKind::Anthropic,
                &model.model,
                client.base_url(),
                header(client.headers(), "x-api-key", None),
                client.http_client(),
            );
        }
        if let Some(model) = model.downcast_ref::<rig_core::providers::openai::CompletionModel>() {
            let client = model.client();
            return describe(
                ProviderKind::OpenAiChat,
                &model.model,
                client.base_url(),
                header(client.headers(), "authorization", Some("Bearer ")),
                client.http_client(),
            );
        }
        if let Some(model) =
            model.downcast_ref::<rig_core::providers::openai::ResponsesCompletionModel>()
        {
            let client = model.client();
            return describe(
                ProviderKind::OpenAiResponses,
                &model.model,
                client.base_url(),
                header(client.headers(), "authorization", Some("Bearer ")),
                client.http_client(),
            );
        }
        if let Some(model) = model.downcast_ref::<rig_core::providers::gemini::CompletionModel>() {
            let client = model.client();
            return describe(
                ProviderKind::Gemini,
                &model.model,
                client.base_url(),
                Some(client.provider().api_key().to_owned()),
                client.http_client(),
            );
        }
        if let Some(model) = model.downcast_ref::<rig_core::providers::deepseek::CompletionModel>()
        {
            let client = model.client();
            return describe(
                ProviderKind::DeepSeek,
                &model.model,
                client.base_url(),
                header(client.headers(), "authorization", Some("Bearer ")),
                client.http_client(),
            );
        }
        None
    }
}

/// The credential reference every wire binding names; the harness
/// resolver maps it to the cassette's key.
pub(crate) const CASSETTE_CREDENTIAL: &str = "cassette";

/// A wire's default model as a binding, with what the harness
/// `Materializer` needs to rebuild the head client from it.
pub(crate) struct WireBinding {
    /// The binding, under `golden/model:default`.
    pub(crate) binding: ProviderBinding,
    /// The head client's key: what `cassette` resolves to.
    pub(crate) api_key: String,
    /// The head client's transport: what the factory hands every client.
    pub(crate) transport: BoxedHttpClient,
}

/// A header the client sends on every request, with `prefix` stripped.
fn header(
    headers: &rig_core::http_client::HeaderMap,
    name: &str,
    prefix: Option<&str>,
) -> Option<String> {
    let value = headers.get(name)?.to_str().ok()?;
    match prefix {
        Some(prefix) => value.strip_prefix(prefix).map(str::to_owned),
        None => Some(value.to_owned()),
    }
}
