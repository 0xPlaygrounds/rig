//! Nonblocking approval: the host keeps ticking while a tool waits for input.
//! Unix CLI input is readiness-polled without a background thread. On any
//! native platform, `--decision approve|deny|cancel` supplies scripted input.
//! Tools and the model are local mocks; no email is actually sent.

#[cfg(unix)]
#[path = "human_in_the_loop/input.rs"]
mod input;
mod support;

use bevy_app::{App, Update};
use bevy_ecs::prelude::*;
use rig_core::{message::AssistantContent, observe::Emitter};
use rig_ecs::{
    agent::ToolCallSlot,
    approval::{ApprovalChoice, ApprovalRequest, ApprovalRequired, decide},
    bus::{BusSet, Handlers, PendingEffect, RigSchedule, run_to_quiescence},
    commands::{Agent, Prompt, install},
    inspect::inspect,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let scripted = match arguments.as_slice() {
        [] => None,
        [flag, value] if flag == "--decision" => Some(choice(value)?),
        _ => return Err("usage: human_in_the_loop [--decision approve|deny|cancel]".into()),
    };
    #[cfg(not(unix))]
    if scripted.is_none() {
        return Err("interactive CLI input requires Unix; use --decision or supply decisions from your application's UI".into());
    }
    #[cfg(unix)]
    let mut input = input::Input::new(std::io::stdin());

    let mut app = App::new();
    install(app.world_mut(), Default::default())?;
    app.add_systems(Update, run_to_quiescence)
        .add_systems(RigSchedule, require_approval.in_set(BusSet::Gate));
    let world = app.world_mut();
    let model = Handlers::register_in(
        world,
        support::MODEL,
        support::Scripted::new(vec![
            vec![support::call(
                "send_email",
                serde_json::json!({"to": "ada@example.com", "subject": "Hi", "body": "Hello, Ada."}),
            )],
            vec![AssistantContent::text("Done.")],
        ]),
    )?;
    let tool = Handlers::register_in(world, "demo/email", support::send_email())?;
    let agent = Agent::new(model)
        .preamble("You are an assistant with an email tool.")
        .tools([tool])
        .max_turns(2)
        .spawn(world)?;
    let run = Prompt::new(agent, "Email Ada to say hello.").spawn(world)?;
    let mut requests = app.world_mut().query::<&ApprovalRequest>();
    let mut displayed = None;
    loop {
        app.update();
        let view = inspect(app.world(), run)?;
        if let Some(failure) = view.failure() {
            return Err(format!("run failed: {failure:?}").into());
        }
        if let Some(answer) = view.answer() {
            println!("{answer}");
            return Ok(());
        }
        if displayed.is_none()
            && let Some(request) = requests
                .iter(app.world())
                .find(|request| request.run() == run && request.is_pending())
        {
            println!(
                "review {}: {} {}",
                request.ticket().revision(),
                request.name(),
                request.args()
            );
            println!("approve / deny / cancel?");
            displayed = Some(request.ticket());
        }
        if let Some(ticket) = displayed {
            let mut decision = scripted.clone();
            #[cfg(unix)]
            if decision.is_none() {
                decision = match input.poll()? {
                    Some(input::InputEvent::Line(line)) => {
                        Some(choice(line.trim()).unwrap_or_else(|_| {
                            ApprovalChoice::Cancel("invalid reviewer input".into())
                        }))
                    }
                    Some(input::InputEvent::Closed) => {
                        Some(ApprovalChoice::Cancel("reviewer input closed".into()))
                    }
                    None => None,
                };
            }
            if let Some(decision) = decision {
                // Keep the ticket that was displayed, even if the proposal has
                // changed since then. A stale reply is rejected and redisplayed.
                match decide(app.world_mut(), ticket, decision) {
                    Ok(_) => println!("decision applied"),
                    Err(error) => eprintln!("decision rejected: {error}"),
                }
                displayed = None;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn choice(text: &str) -> Result<ApprovalChoice, &'static str> {
    match text {
        "approve" => Ok(ApprovalChoice::Approve),
        "deny" => Ok(ApprovalChoice::Deny("denied by the reviewer".into())),
        "cancel" => Ok(ApprovalChoice::Cancel("cancelled by the reviewer".into())),
        _ => Err("expected approve, deny, or cancel"),
    }
}

fn require_approval(
    calls: Query<Entity, (Added<PendingEffect>, With<ToolCallSlot>)>,
    mut commands: Commands,
) {
    for call in &calls {
        commands
            .entity(call)
            .insert(ApprovalRequired(Emitter::named("app/human")));
    }
}
