//! The agent runtime's facts for the world's witness: a run's ending, a
//! cancellation request, a retried turn, an invalid call's resolution. Each
//! is observed at the component write that is the decision, by a lifecycle
//! observer installed with the runtime, so a host system that writes the
//! component (a `Retry`, a `Cancelled`) is observed the same way the
//! library's own systems are.

use bevy_ecs::{lifecycle::Insert, prelude::*};
use rig_core::observe::{Action, Emitter, Observation, Reason, RunTiming, Stage};

use crate::{
    agent::{Cancelled, Failed, Failure, InvalidCall, Resolution, Retry, Run, Settled},
    bus::{Subjects, Witnessing},
};

/// The agent runtime's emitter name.
pub const AGENT_EMITTER: &str = "rig-ecs/agent";

/// The agent runtime as an emitter: its name and crate version.
pub fn agent_emitter() -> Emitter {
    Emitter::versioned(AGENT_EMITTER, env!("CARGO_PKG_VERSION"))
}

/// Install the runtime's observers on `world`.
pub fn install(world: &mut World) {
    world.add_observer(observe_settled);
    world.add_observer(observe_failed);
    world.add_observer(observe_cancel_requested);
    world.add_observer(observe_retry);
    world.add_observer(observe_resolution);
    world.add_observer(observe_run_despawned);
}

/// Runtime-only interval state; a scene cannot resume an old clock interval.
#[derive(Component)]
pub(super) struct RunTimer {
    interval: Option<(std::time::Duration, Witnessing)>,
}

impl RunTimer {
    pub(super) fn start(world: &World) -> Option<Self> {
        let witness = world.get_resource::<Witnessing>()?.clone();
        let started = witness.sink().elapsed()?;
        Some(Self {
            interval: Some((started, witness)),
        })
    }

    fn finish(&mut self, complete: bool) -> Option<RunTiming> {
        let (start, witness) = self.interval.take()?;
        Some(RunTiming {
            duration: if complete {
                witness
                    .sink()
                    .elapsed()
                    .and_then(|end| end.checked_sub(start))
            } else {
                None
            },
            complete,
        })
    }
}

fn observe_settled(
    added: On<bevy_ecs::lifecycle::Add, Settled>,
    subjects: Subjects,
    witness: Option<Res<Witnessing>>,
    mut timings: Query<&mut RunTimer>,
) {
    let Some(witness) = witness else {
        return;
    };
    let mut observation = Observation::new(
        subjects.of_scope(added.event().entity),
        Stage::Runtime,
        agent_emitter(),
        Action::Ended {
            ending: Reason::code("settled"),
        },
    );
    observation.run_timing = timings
        .get_mut(added.event().entity)
        .ok()
        .and_then(|mut timer| timer.finish(true));
    witness.observe(observation);
}

/// The ending's code and detail, by the failure's variant.
pub fn ending_of(failure: &Failure) -> Reason {
    match failure {
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
    mut timings: Query<&mut RunTimer>,
) {
    let Some(witness) = witness else {
        return;
    };
    let entity = added.event().entity;
    let Ok(Failed(failure)) = failures.get(entity) else {
        return;
    };
    let mut observation = Observation::new(
        subjects.of_scope(entity),
        Stage::Runtime,
        agent_emitter(),
        Action::Ended {
            ending: ending_of(failure),
        },
    );
    observation.run_timing = timings
        .get_mut(entity)
        .ok()
        .and_then(|mut timer| timer.finish(true));
    witness.observe(observation);
}

fn observe_run_despawned(
    removed: On<bevy_ecs::lifecycle::Despawn, Run>,
    runs: Query<(Has<Settled>, Has<Failed>)>,
    mut timings: Query<&mut RunTimer>,
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
    let mut observation = Observation::new(
        subjects.of_scope(entity),
        Stage::Runtime,
        agent_emitter(),
        Action::Ended {
            ending: Reason::code("despawned_without_ending"),
        },
    );
    observation.run_timing = timings
        .get_mut(entity)
        .ok()
        .and_then(|mut timer| timer.finish(false));
    witness.observe(observation);
}

fn observe_cancel_requested(
    added: On<bevy_ecs::lifecycle::Add, Cancelled>,
    reasons: Query<&Cancelled>,
    subjects: Subjects,
    witness: Option<Res<Witnessing>>,
) {
    let Some(witness) = witness else {
        return;
    };
    let entity = added.event().entity;
    let Ok(Cancelled(reason)) = reasons.get(entity) else {
        return;
    };
    witness.emit(
        subjects.of_scope(entity),
        Stage::Runtime,
        Emitter::unknown(),
        Action::CancelRequested {
            reason: Reason::with_detail("cancelled", reason.clone()),
        },
    );
}

fn observe_retry(
    added: On<bevy_ecs::lifecycle::Add, Retry>,
    retries: Query<&Retry>,
    subjects: Subjects,
    witness: Option<Res<Witnessing>>,
) {
    let Some(witness) = witness else {
        return;
    };
    let entity = added.event().entity;
    let Ok(retry) = retries.get(entity) else {
        return;
    };
    witness.emit(
        subjects.of_scope(entity),
        Stage::Runtime,
        Emitter::unknown(),
        Action::Retry {
            feedback: retry.feedback.clone(),
        },
    );
}

fn observe_resolution(
    inserted: On<Insert, Resolution>,
    calls: Query<(&InvalidCall, &Resolution)>,
    subjects: Subjects,
    witness: Option<Res<Witnessing>>,
) {
    let Some(witness) = witness else {
        return;
    };
    let entity = inserted.event().entity;
    let Ok((call, resolution)) = calls.get(entity) else {
        return;
    };
    let resolution = match resolution {
        Resolution::Fail => Reason::code("fail"),
        Resolution::Ignore => Reason::code("ignore"),
        Resolution::Retry { feedback } => Reason::with_detail("retry", feedback.clone()),
        Resolution::Repair { to } => Reason::with_detail("repair", to.clone()),
        Resolution::Skip { reason } => Reason::with_detail("skip", reason.clone()),
    };
    witness.emit(
        subjects.of_scope(entity),
        Stage::Runtime,
        Emitter::unknown(),
        Action::InvalidCall {
            name: call.name.clone(),
            resolution,
        },
    );
}
