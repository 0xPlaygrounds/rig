//! Explicit schedule and system-set ordering for ECS progression.

use bevy_ecs::{
    schedule::{ApplyDeferred, IntoScheduleConfigs, Schedule, ScheduleLabel, SystemSet},
    world::World,
};

/// Runtime schedules are driven by both blocking and streaming handles.
#[derive(ScheduleLabel, Clone, Debug, Eq, Hash, PartialEq)]
pub enum RuntimeSchedule {
    /// Snapshot authoritative state and create owned effects.
    Dispatch,
    /// Validate and commit owned effect completions.
    Ingress,
    /// Resolve policy, quiescence, and terminal winners.
    Progress,
    /// Retire retained terminal state after observers can see it.
    Cleanup,
}

/// Deterministic system ordering within runtime schedules.
#[derive(SystemSet, Clone, Debug, Eq, Hash, PartialEq)]
pub enum RuntimeSet {
    Validate,
    Snapshot,
    Dispatch,
    ApplyDispatch,
    Ingress,
    Commit,
    ApplyCommit,
    Policy,
    Quiescence,
    Terminal,
    Retention,
}

pub(crate) fn install(world: &mut World) {
    world.init_resource::<crate::runtime::DispatchQueue>();
    world.init_resource::<crate::runtime::DispatchOutbox>();
    world.init_resource::<crate::runtime::ToolDispatchQueue>();
    world.init_resource::<crate::runtime::ToolDispatchOutbox>();
    world.init_resource::<crate::runtime::MemoryDispatchQueue>();
    world.init_resource::<crate::runtime::MemoryDispatchOutbox>();
    world.init_resource::<crate::runtime::VectorDispatchQueue>();
    world.init_resource::<crate::runtime::VectorDispatchOutbox>();
    world.init_resource::<crate::runtime::IngressQueue>();
    world.init_resource::<crate::runtime::IngressOutbox>();
    world.init_resource::<crate::runtime::ToolIngressQueue>();
    world.init_resource::<crate::runtime::ToolIngressOutbox>();
    world.init_resource::<crate::runtime::ToolTurnIngressQueue>();
    world.init_resource::<crate::runtime::ToolTurnIngressOutbox>();
    world.init_resource::<crate::runtime::MemoryAppendIngressQueue>();
    world.init_resource::<crate::runtime::MemoryAppendIngressOutbox>();
    world.init_resource::<crate::runtime::MemoryLoadIngressQueue>();
    world.init_resource::<crate::runtime::MemoryLoadIngressOutbox>();
    world.init_resource::<crate::runtime::VectorIngressQueue>();
    world.init_resource::<crate::runtime::VectorIngressOutbox>();
    world.init_resource::<crate::runtime::StreamingIngressQueue>();
    world.init_resource::<crate::runtime::StreamingIngressOutbox>();
    world.init_resource::<crate::runtime::HostedEffectQueue>();
    world.init_resource::<crate::runtime::PolicyQueue>();
    world.init_resource::<crate::runtime::PolicyOutbox>();
    let mut dispatch = Schedule::new(RuntimeSchedule::Dispatch);
    dispatch.configure_sets(
        (
            RuntimeSet::Validate,
            RuntimeSet::Snapshot,
            RuntimeSet::Dispatch,
            RuntimeSet::ApplyDispatch,
        )
            .chain(),
    );
    dispatch.add_systems(
        (
            crate::runtime::dispatch_system,
            crate::runtime::tool_dispatch_system,
            crate::runtime::memory_dispatch_system,
            crate::runtime::vector_dispatch_system,
        )
            .chain()
            .in_set(RuntimeSet::Dispatch),
    );
    dispatch.add_systems(ApplyDeferred.in_set(RuntimeSet::ApplyDispatch));
    world.add_schedule(dispatch);

    let mut ingress = Schedule::new(RuntimeSchedule::Ingress);
    ingress.configure_sets(
        (
            RuntimeSet::Validate,
            RuntimeSet::Ingress,
            RuntimeSet::Commit,
            RuntimeSet::ApplyCommit,
        )
            .chain(),
    );
    ingress.add_systems(
        (
            crate::runtime::completion_ingress_system,
            crate::runtime::tool_ingress_system,
            crate::runtime::tool_turn_ingress_system,
            crate::runtime::memory_load_ingress_system,
            crate::runtime::memory_append_ingress_system,
            crate::runtime::vector_ingress_system,
            crate::runtime::streaming_ingress_system,
        )
            .chain()
            .in_set(RuntimeSet::Ingress),
    );
    ingress.add_systems(ApplyDeferred.in_set(RuntimeSet::ApplyCommit));
    world.add_schedule(ingress);

    let mut progress = Schedule::new(RuntimeSchedule::Progress);
    progress.configure_sets(
        (
            RuntimeSet::Policy,
            RuntimeSet::Quiescence,
            RuntimeSet::Terminal,
        )
            .chain(),
    );
    progress.add_systems(
        (
            crate::runtime::policy_command_system,
            crate::runtime::policy_progress_system,
        )
            .chain()
            .in_set(RuntimeSet::Policy),
    );
    progress.add_systems(crate::runtime::quiescence_progress_system.in_set(RuntimeSet::Quiescence));
    progress.add_systems(crate::runtime::terminal_progress_system.in_set(RuntimeSet::Terminal));
    world.add_schedule(progress);

    let mut cleanup = Schedule::new(RuntimeSchedule::Cleanup);
    cleanup.configure_sets(RuntimeSet::Retention);
    cleanup.add_systems(crate::runtime::cleanup_system.in_set(RuntimeSet::Retention));
    world.add_schedule(cleanup);
}
