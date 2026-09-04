//! `examples/agent_with_human_in_the_loop` side by side: every tool call
//! waits for a human's decision — there an `AgentHook::on_dispatch` that
//! awaits stdin, here a system in `BusSet::Gate` that reads a line before
//! the bus takes the tool child. Approve: the child goes on. Deny: an
//! `EffectOutcome` with the reason, never dispatched — the model reads the
//! denial as the tool's result. Abort, or no input at all (closed stdin):
//! `Cancelled` on the run, fail-closed.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::type_complexity,
    reason = "an example: user code, thirty lines, a mock behind it"
)]

mod support;

use bevy_app::Startup;
use bevy_ecs::prelude::*;
use rig_core::{
    effect::EffectKind,
    error::{ErrorKind, ErrorReport},
    message::AssistantContent,
};
use rig_ecs::{
    agent::{Order, ToolCallSlot, Turn},
    bus::{Handlers, PendingEffect, RigSchedule},
    prelude::*,
    systems::spawn_run,
};

fn main() {
    support::app()
        .add_systems(Startup, ask)
        .add_systems(RigSchedule, approve.in_set(BusSet::Gate))
        .add_observer(support::print_the_answer_and_exit)
        .run();
}

fn ask(mut handlers: Handlers, mut commands: Commands) {
    let model = support::Scripted::new(vec![
        vec![support::call(
            "send_email",
            serde_json::json!({"to": "ada@example.com", "subject": "Hi", "body": "Hello, Ada."}),
        )],
        vec![AssistantContent::text("Done.")],
    ]);
    let (model, tools) = support::register(&mut handlers, model, vec![support::send_email()]);
    let agent = support::agent(
        &mut commands,
        model,
        "You are an assistant with an email tool.",
        2,
    );
    commands.spawn((Grant(tools[0]), Order(0), ChildOf(agent)));
    commands.queue(move |world: &mut World| {
        spawn_run(world, agent, &[], "Email Ada to say hello.", false, None);
    });
}

/// `on_dispatch`, as a system in `Gate`: a fresh tool child waits for the
/// human's line before the bus's `Dispatch` sees it.
fn approve(
    calls: Query<(Entity, &PendingEffect, &ChildOf), (Added<PendingEffect>, With<ToolCallSlot>)>,
    turns: Query<&ChildOf, With<Turn>>,
    mut commands: Commands,
) {
    for (call, effect, turn_of) in &calls {
        let EffectKind::ToolCall { name, args } = &effect.kind else {
            continue;
        };
        println!("\nthe agent wants to run a tool: {name} {args}");
        println!("[a]pprove / [d]eny / a[b]ort?");
        let mut line = String::new();
        let decision = match std::io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(line.trim().to_ascii_lowercase()),
        };
        match decision.as_deref() {
            Some("a" | "approve") => println!("approved"),
            Some("d" | "deny") => {
                println!("denied");
                commands
                    .entity(call)
                    .insert(EffectOutcome(Err(ErrorReport::new(
                        ErrorKind::Denied,
                        "denied by the human reviewer",
                    ))));
            }
            _ => {
                println!("aborting (fail-closed)");
                if let Ok(run_of) = turns.get(turn_of.parent()) {
                    commands
                        .entity(run_of.parent())
                        .insert(Cancelled("no reviewer approval".to_owned()));
                }
            }
        }
    }
}
