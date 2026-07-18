//! Stable domain identity and explicit ownership relationships.

use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::component::Component;
use serde::{Deserialize, Serialize};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_WORLD: AtomicU64 = AtomicU64::new(1);

macro_rules! stable_id {
    ($name:ident) => {
        #[doc = concat!("Stable ", stringify!($name), " domain identifier.")]
        #[derive(
            Component,
            Clone,
            Copy,
            Debug,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            Serialize,
            Deserialize,
        )]
        pub struct $name(pub u64);

        impl $name {
            pub(crate) fn allocate() -> Self {
                Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
            }
        }
    };
}

stable_id!(AgentId);
stable_id!(RunId);
stable_id!(OperationId);
stable_id!(ToolCallId);
stable_id!(CapabilityId);
stable_id!(StoreOperationId);

/// Stable runtime-world identity used to reject foreign completions and handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WorldId(pub u64);

impl WorldId {
    pub(crate) fn allocate() -> Self {
        Self(NEXT_WORLD.fetch_add(1, Ordering::Relaxed))
    }
}

fn advance_counter(counter: &AtomicU64, observed: u64) -> bool {
    let Some(next) = observed.checked_add(1) else {
        return false;
    };
    let mut current = counter.load(Ordering::Relaxed);
    while current < next {
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
    true
}

pub(crate) fn allocate_after_restore(max_stable: u64, max_world: u64) -> Option<WorldId> {
    if !advance_counter(&NEXT_ID, max_stable) || !advance_counter(&NEXT_WORLD, max_world) {
        return None;
    }
    Some(WorldId::allocate())
}

/// Explicit tenant boundary checked at every effect and capability ingress.
#[derive(
    Component,
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
pub struct TenantId(pub u64);

/// Generation of a restorable or replaceable domain object.
#[derive(
    Component,
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
pub struct Generation(pub u64);

impl Generation {
    /// Return the next generation, saturating instead of wrapping identity.
    pub fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

/// Correlation carried by every owned effect request and completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EffectIdentity {
    /// Runtime world that dispatched the effect.
    pub world: WorldId,
    /// Owning tenant.
    pub tenant: TenantId,
    /// Stable run identifier.
    pub run: RunId,
    /// Stable operation identifier.
    pub operation: OperationId,
    /// Run generation at dispatch.
    pub generation: Generation,
    /// Monotonic per-run correlation value.
    pub correlation: u64,
}

/// Stable handle identity validated before any world mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HandleIdentity {
    /// Runtime world that created the handle.
    pub world: WorldId,
    /// Owning tenant.
    pub tenant: TenantId,
    /// Stable domain run.
    pub run: RunId,
    /// Generation observed when the handle was created.
    pub generation: Generation,
}

/// Explicit relationship from an operation or call to its stable owning run.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnedByRun(pub RunId);

/// Explicit relationship from a run to its stable owning agent.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnedByAgent(pub AgentId);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_ids_and_world_ids_are_distinct() {
        assert_ne!(RunId::allocate(), RunId::allocate());
        assert_ne!(WorldId::allocate(), WorldId::allocate());
    }

    #[test]
    fn generation_never_wraps() {
        assert_eq!(Generation(u64::MAX).next(), Generation(u64::MAX));
    }
}
