//! Host custom effects through native providers, bus and application policies.
use super::corpus_host::{ADD_PROMPT, PROMPT};
use crate::{
    ecs_agent::{EcsAgent, RuntimeHandler, io_runtime},
    goldens::{NOTE_KEY, NoteTaker},
    support::{Adder, BASIC_PREAMBLE, TOOLS_PREAMBLE},
};
use bevy_ecs::{prelude::*, system::RunSystemOnce};
use rig::serve::ServingPolicy;
use rig_ecs::{
    agent::{PolicyVersion, Temperature},
    bus::{Handlers, PendingEffect, Policy, RigSchedule},
    systems::{RigSet, RunCommands},
};
use std::sync::Arc;
#[path = "ecs_host/policies.rs"]
mod policies;
use policies::*;

/// The hooks a cell registers, in order.
#[derive(Clone, Copy, Debug)]
enum Hooks {
    AtStart,
    AtCompletionCall,
    AtOutcome,
    AtSettled,
    StartAndSettled,
    Twice,
}

/// What a cell asks of the host.
struct Host {
    /// Register the note taker.
    notes: bool,
    /// The host's serving policy.
    serial: bool,
    /// Keep stream events.
    streamed: bool,
    /// Advertise `add` and ask for a sum.
    with_tool: bool,
}

const PLAIN: Host = Host {
    notes: true,
    serial: false,
    streamed: false,
    with_tool: false,
};

fn agent<
    W: rig_core::wire::Wire<Op = rig_core::operation::Completion>,
    T: rig_core::driver::Transport<W>,
>(
    model: rig_core::driver::Model<W, T>,
    host: &Host,
    hooks: Hooks,
) -> EcsAgent {
    let preamble = if host.with_tool {
        TOOLS_PREAMBLE
    } else {
        BASIC_PREAMBLE
    };
    let mut ecs = EcsAgent::for_golden(model, preamble, host.streamed);
    ecs.app.world_mut().resource_mut::<Policy>().0 = ServingPolicy {
        serial_per_handler: host.serial,
        ..ServingPolicy::default()
    };
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert(Temperature(Some(0.0)));
    if host.notes {
        Handlers::with(ecs.app.world_mut(), |handlers| {
            handlers.register(
                NOTE_KEY,
                RuntimeHandler {
                    inner: Arc::new(NoteTaker),
                    runtime: io_runtime(),
                },
            )
        })
        .expect("bus installed")
        .expect("fresh note key");
    }
    if host.with_tool {
        ecs.tool(Adder);
    }
    let names: &[&str] = match hooks {
        Hooks::AtStart => {
            ecs.app.add_observer(at_start);
            &["NoteAtStart"]
        }
        Hooks::AtCompletionCall => {
            ecs.app.add_systems(
                RigSchedule,
                at_completion_call
                    .after(RigSet::Select)
                    .before(RigSet::Assemble),
            );
            &["NoteAtCompletionCall"]
        }
        Hooks::AtOutcome => {
            ecs.app.add_observer(at_outcome);
            &["NoteAtOutcome"]
        }
        Hooks::AtSettled => {
            ecs.app.add_observer(at_settled);
            &["NoteAtSettled"]
        }
        Hooks::StartAndSettled => {
            ecs.app.add_observer(at_start).add_observer(at_settled);
            &["NoteAtStart", "NoteAtSettled"]
        }
        Hooks::Twice => {
            ecs.app.add_observer(twice);
            &["NoteTwice"]
        }
    };
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert(PolicyVersion(format!("ecs-host/v1:{}", names.join(","))));
    ecs.app.configure_sets(
        RigSchedule,
        (
            RigSet::Advance.run_if(ready),
            RigSet::Assemble.run_if(ready),
            RigSet::Materialise.run_if(ready),
        ),
    );

    ecs
}
async fn run_prompt(ecs: &mut EcsAgent, host: &Host) -> String {
    let prompt = if host.with_tool { ADD_PROMPT } else { PROMPT };
    let run = ecs
        .app
        .world_mut()
        .spawn_run(ecs.agent, &[], prompt, host.streamed, Some(3));
    let output = ecs.wait_for_success(run).await;
    // Native Settled publishes before an application-owned settled note finishes.
    // Await its real acknowledgement before exposing this consumer's response.
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            ecs.app.update();
            if ecs
                .app
                .world_mut()
                .run_system_once(ready)
                .expect("note checks")
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("host note acknowledgement deadline");

    output
}

#[path = "ecs_host/tests.rs"]
mod tests;
