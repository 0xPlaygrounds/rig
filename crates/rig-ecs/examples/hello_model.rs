//! The advanced bus-only path: install into a World, register a handler,
//! spawn an effect, and observe its answer. No agent runtime is needed.
//! The mock uses the same Serve boundary as a real CompletionAdapter.

use bevy_app::{App, AppExit, ScheduleRunnerPlugin, Update};
use bevy_ecs::prelude::*;
use rig_core::serve::Dispatch;
use rig_core::{
    completion::{
        CompletionRequestBuilder, CompletionResponse, ModelRef, ProviderCapabilities, Usage,
    },
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    message::AssistantContent,
    serve::{Serve, ServingPolicy},
};
use rig_ecs::bus::{EffectOutcome, Handlers, PendingEffect, install_bus, run_to_quiescence};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut app = App::new();
    install_bus(app.world_mut(), ServingPolicy::default());
    Handlers::register_in(app.world_mut(), "model", Mock)?;
    app.add_plugins(ScheduleRunnerPlugin::default())
        .add_systems(Update, run_to_quiescence)
        .add_observer(print_the_answer);
    app.world_mut().spawn(PendingEffect::new(
        "model",
        EffectKind::Completion {
            request: CompletionRequestBuilder::unbound("hello?").build(),
            stream: false,
        },
    ));
    if app.run().is_success() {
        Ok(())
    } else {
        Err(std::io::Error::other("model effect failed").into())
    }
}

fn print_the_answer(
    answered: On<Add, EffectOutcome>,
    outcomes: Query<&EffectOutcome>,
    mut exit: MessageWriter<AppExit>,
) {
    if let Ok(outcome) = outcomes.get(answered.event().entity) {
        match &outcome.0 {
            Ok(Outcome::Completion(_)) => {
                println!("the model said: {}", text(&outcome.0));
                exit.write(AppExit::Success);
            }
            _ => {
                eprintln!("{}", text(&outcome.0));
                exit.write(AppExit::error());
            }
        }
    }
}

// ---- the mock, in place of a provider ----

struct Mock;

impl Serve for Mock {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("model"),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new("mock"),
                capabilities: ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        let response = CompletionResponse::new(
            vec![AssistantContent::text("hello from the world")],
            Usage::new(),
            "mock",
        );
        rig_core::serve::Reply::Outcome(Ok(Outcome::Completion(response)))
    }
}

fn text(outcome: &Result<Outcome, rig_core::error::ErrorReport>) -> String {
    match outcome {
        Ok(Outcome::Completion(response)) => response
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.clone()),
                AssistantContent::Reasoning(_)
                | AssistantContent::Image(_)
                | AssistantContent::ToolCall(_) => None,
            })
            .collect(),
        Ok(other) => format!("a {} answer", other.family()),
        Err(report) => format!("failed: {report}"),
    }
}
