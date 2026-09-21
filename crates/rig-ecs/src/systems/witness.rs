//! Run endings for the world's witness, including settlement, failure,
//! cancellation and removal before an ending. Lifecycle observers see the
//! same ending components whether written by the runtime or a host system.
//!
//! ```
//! let mut world = bevy_ecs::world::World::new();
//! rig_ecs::systems::witness::install(&mut world);
//! ```

use bevy_ecs::prelude::*;
use rig_core::observe::{Action, Emitter, HostAction, Observation, Reason, Stage, Subject};
use serde::{Deserialize, Serialize};

use crate::{
    agent::{Failed, Failure, Run, Settled},
    bus::{Subjects, Witnessing},
};

/// The agent runtime's emitter name.
pub const AGENT_EMITTER: &str = "rig-ecs/agent";

/// The agent runtime as an emitter: its name and crate version.
pub fn agent_emitter() -> Emitter {
    Emitter::versioned(AGENT_EMITTER, env!("CARGO_PKG_VERSION"))
}

/// The runtime re-issued a completion after a retryable provider failure
/// (CONTRACT §5): which attempt this is, out of how many, and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderRetry {
    /// The attempt about to be issued, counting from 1.
    pub attempt: usize,
    /// The run's retry budget.
    pub budget: usize,
    /// The failure's kind code and message.
    pub reason: Reason,
}

impl HostAction for ProviderRetry {
    const KIND: &'static str = "rig-ecs/agent/provider_retry";
}

/// Tell the witness a completion is being re-issued.
pub(crate) fn observe_provider_retry(
    witness: &Witnessing,
    subject: Subject,
    attempt: usize,
    budget: usize,
    report: &rig_core::error::ErrorReport,
) {
    let fact = ProviderRetry {
        attempt,
        budget,
        reason: Reason::with_detail(report.kind.code(), report.message.clone()),
    };
    if let Ok(action) = fact.action() {
        witness.observe(Observation::new(
            subject,
            Stage::Runtime,
            agent_emitter(),
            action,
        ));
    }
}

/// Install the runtime's observers on `world`.
pub fn install(world: &mut World) {
    world.add_observer(observe_settled);
    world.add_observer(observe_failed);
    world.add_observer(observe_run_despawned);
}

fn observe_settled(
    added: On<bevy_ecs::lifecycle::Add, Settled>,
    subjects: Subjects,
    witness: Option<Res<Witnessing>>,
) {
    let Some(witness) = witness else {
        return;
    };
    let observation = Observation::new(
        subjects.of_scope(added.event().entity),
        Stage::Runtime,
        agent_emitter(),
        Action::Ended {
            ending: Reason::code("settled"),
        },
    );
    witness.observe(observation);
}

/// The ending's code and detail, by the failure's variant.
pub fn ending_of(failure: &Failure) -> Reason {
    match failure {
        Failure::Content(error) => Reason::with_detail("invalid_content", error.to_string()),
        Failure::MaxTurns { limit } => Reason::with_detail(
            "max_turns",
            format!("the model-call budget of {limit} ran out"),
        ),
        Failure::UnknownToolCall { name } => Reason::with_detail("unknown_tool_call", name.clone()),
        Failure::Provider(report) => Reason::with_detail(
            "provider",
            format!("{}: {}", report.kind.code(), report.message),
        ),
        Failure::Cancelled(report) => Reason::with_detail("cancelled", report.message.clone()),
        Failure::Unsupported(what) => Reason::with_detail("unsupported", what.clone()),
        Failure::OutputToolCollision { name } => {
            Reason::with_detail("output_tool_collision", name.clone())
        }
        Failure::Tool(report) => Reason::with_detail(
            "tool",
            format!("{}: {}", report.kind.code(), report.message),
        ),
        Failure::Memory(report) => Reason::with_detail(
            "memory",
            format!("{}: {}", report.kind.code(), report.message),
        ),
    }
}

fn observe_failed(
    added: On<bevy_ecs::lifecycle::Add, Failed>,
    failures: Query<&Failed>,
    subjects: Subjects,
    witness: Option<Res<Witnessing>>,
) {
    let Some(witness) = witness else {
        return;
    };
    let entity = added.event().entity;
    let Ok(Failed(failure)) = failures.get(entity) else {
        return;
    };
    let observation = Observation::new(
        subjects.of_scope(entity),
        Stage::Runtime,
        agent_emitter(),
        Action::Ended {
            ending: ending_of(failure),
        },
    );
    witness.observe(observation);
}

fn observe_run_despawned(
    removed: On<bevy_ecs::lifecycle::Despawn, Run>,
    runs: Query<(Has<Settled>, Has<Failed>)>,
    subjects: Subjects,
    witness: Option<Res<Witnessing>>,
) {
    let entity = removed.event().entity;
    let Some(witness) = witness else { return };
    let Ok((settled, failed)) = runs.get(entity) else {
        return;
    };
    if settled || failed {
        return;
    }
    let observation = Observation::new(
        subjects.of_scope(entity),
        Stage::Runtime,
        agent_emitter(),
        Action::Ended {
            ending: Reason::code("despawned_without_ending"),
        },
    );
    witness.observe(observation);
}
