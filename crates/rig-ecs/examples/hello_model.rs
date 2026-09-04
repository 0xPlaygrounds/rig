//! Hello, model: the bus in a `World` in under thirty lines of user code.
//! Add the plugin, register a handler, spawn an effect, observe the answer.
//! The model is a scripted mock, so no provider feature and no key: the
//! shape is the same with a real `CompletionAdapter` registered under the
//! key.

use bevy_app::{App, AppExit, ScheduleRunnerPlugin, Startup};
use bevy_ecs::prelude::*;
use rig_core::{
    completion::{
        CompletionRequest, CompletionResponse, Message, ModelRef, ProviderCapabilities, Usage,
    },
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    message::AssistantContent,
    serve::{OutcomeSink, Serve},
};
use rig_ecs::bus::{BusPlugin, EffectOutcome, Handlers, PendingEffect};

// ---- the user's program: the next thirty lines ----

fn main() {
    App::new()
        .add_plugins((ScheduleRunnerPlugin::default(), BusPlugin::default()))
        .add_systems(Startup, (register_the_model, ask).chain())
        .add_observer(print_the_answer)
        .run();
}

fn register_the_model(mut handlers: Handlers) {
    if let Err(report) = handlers.register("model", Mock) {
        eprintln!("could not register the model: {report}");
    }
}

fn ask(mut commands: Commands) {
    let request = CompletionRequest {
        model: None,
        chat_history: vec![Message::user("hello?")],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };
    commands.spawn(PendingEffect::new(
        "model",
        EffectKind::Completion {
            request,
            stream: false,
        },
    ));
}

fn print_the_answer(
    answered: On<Add, EffectOutcome>,
    outcomes: Query<&EffectOutcome>,
    mut exit: MessageWriter<AppExit>,
) {
    if let Ok(outcome) = outcomes.get(answered.event().entity) {
        println!("the model said: {}", text(&outcome.0));
        exit.write(AppExit::Success);
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

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        let response = CompletionResponse::new(
            vec![AssistantContent::text("hello from the world")],
            Usage::new(),
            "mock",
        );
        sink.resolve(Ok(Outcome::Completion(response))).await;
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
