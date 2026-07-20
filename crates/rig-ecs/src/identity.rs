//! Stable domain identity used across ECS entities, effects, and snapshots.

use std::fmt;

use bevy_ecs::prelude::Component;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

macro_rules! stable_id {
    ($name:ident, $description:literal, $redacted_debug:literal) => {
        #[doc = $description]
        #[derive(Clone, Copy, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
        #[serde(transparent)]
        pub struct $name(Uuid);

        impl $name {
            #[doc = concat!("Generate a new ", stringify!($name), ".")]
            #[must_use]
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }

            #[doc = concat!("Construct a ", stringify!($name), " from a stable UUID.")]
            #[must_use]
            pub const fn from_uuid(value: Uuid) -> Self {
                Self(value)
            }

            #[doc = concat!("Borrow the UUID backing this ", stringify!($name), ".")]
            #[must_use]
            pub const fn as_uuid(self) -> Uuid {
                self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                if $redacted_debug {
                    formatter.write_str(concat!(stringify!($name), "(<redacted>)"))
                } else {
                    self.0.fmt(formatter)
                }
            }
        }
    };
}

stable_id!(
    RuntimeId,
    "Stable identity of one local or hosted ECS world.",
    false
);
stable_id!(
    AgentId,
    "Stable identity of an agent specification entity.",
    false
);
stable_id!(RunId, "Stable identity of one agent run.", false);
stable_id!(
    OperationId,
    "Stable identity of one asynchronous effect operation.",
    false
);
stable_id!(
    ModelId,
    "Stable identity of a rebound model implementation.",
    false
);
stable_id!(
    CapabilityId,
    "Stable identity of one versioned executable capability.",
    false
);
stable_id!(
    GrantId,
    "Stable identity of an agent-to-capability authorization grant.",
    false
);
stable_id!(
    MemoryId,
    "Stable identity of a rebound conversation-memory implementation.",
    false
);
stable_id!(
    CorrelationId,
    "Unforgeable correlation identity for one effect attempt.",
    false
);
stable_id!(TenantId, "Stable tenant security boundary.", true);

/// Caller-defined stable implementation and configuration revision identity.
///
/// Snapshot rebinding compares this value instead of Rust type names, so two
/// instances of the same type with incompatible endpoints, credentials, model
/// settings, or behavior revisions cannot be confused accidentally.
#[derive(Clone, Copy, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct BindingIdentity([u8; 32]);

impl BindingIdentity {
    /// Derive a stable identity from an implementation name and configuration revision.
    #[must_use]
    pub fn new(implementation: &str, revision: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(implementation.as_bytes());
        hasher.update([0]);
        hasher.update(revision.as_bytes());
        Self(hasher.finalize().into())
    }

    /// Borrow the digest backing this identity.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for BindingIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BindingIdentity(<redacted>)")
    }
}

impl fmt::Display for BindingIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0.iter().take(6) {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Monotonic run generation used to reject superseded async results.
#[derive(
    Clone,
    Copy,
    Component,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
#[serde(transparent)]
pub struct Generation(pub u64);

impl Generation {
    /// Return the next generation, saturating at the integer limit.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}
