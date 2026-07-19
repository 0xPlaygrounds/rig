//! Typed construction, runtime, and snapshot failures.

use std::{fmt, sync::Arc};

use crate::{
    AgentId, CapabilityId, CapabilityKind, GrantId, MemoryEffectError, MemoryId, ModelEffectError,
    ModelId, RunId, RuntimeId, TenantId,
};

/// Failures returned by ECS runtime operations.
#[derive(thiserror::Error)]
#[non_exhaustive]
pub enum RuntimeError {
    /// A bounded runtime setting was zero or otherwise invalid.
    #[error("runtime configuration field `{field}` must be non-zero")]
    InvalidConfiguration {
        /// Invalid configuration field.
        field: &'static str,
    },
    /// An agent specification contains an inconsistent capability relationship.
    #[error("invalid agent specification: {reason}")]
    InvalidAgentSpec {
        /// Stable diagnostic that contains no user data.
        reason: &'static str,
    },
    /// A run prompt would make the canonical transcript invalid at insertion.
    #[error("invalid run prompt: {reason}")]
    InvalidPrompt {
        /// Stable diagnostic that contains no user data.
        reason: &'static str,
    },
    /// A model binding was not registered.
    #[error("model binding `{0}` is not registered")]
    UnknownModel(ModelId),
    /// A memory binding was not registered.
    #[error("memory binding `{0}` is not registered")]
    UnknownMemory(MemoryId),
    /// An agent identity was not present in the world.
    #[error("agent `{0}` is not present in this runtime")]
    UnknownAgent(AgentId),
    /// A run identity was not present in the world.
    #[error("run `{0}` is not present in this runtime")]
    UnknownRun(RunId),
    /// Hosted runs have one schedule driver to preserve ordered ownership.
    #[error("run `{0}` already has an active hosted driver")]
    RunAlreadyDriven(RunId),
    /// A capability identity was not present in the world or registry.
    #[error("capability `{0}` is not present in this runtime")]
    UnknownCapability(CapabilityId),
    /// Replacing a persistable capability requires the replacement's stable identity.
    #[error("replacement for persistable capability `{0}` requires a binding identity")]
    PersistenceIdentityRequired(CapabilityId),
    /// A grant identity was not present in the world.
    #[error("grant `{0}` is not present in this runtime")]
    UnknownGrant(GrantId),
    /// A revoked grant cannot authorize another capability version.
    #[error("grant `{0}` is revoked and cannot authorize replacement")]
    RevokedGrant(GrantId),
    /// A retired capability cannot be used as a replacement predecessor.
    #[error("capability `{0}` is retired and cannot be replaced")]
    RetiredCapability(CapabilityId),
    /// The immutable capability revision chain exhausted its integer domain.
    #[error("capability `{0}` cannot advance beyond its current revision")]
    CapabilityRevisionOverflow(CapabilityId),
    /// One agent cannot advertise two active grants under the same provider name.
    #[error("agent `{agent_id}` already has an active capability named `{name}`")]
    DuplicateToolName {
        /// Agent whose active capability namespace would become ambiguous.
        agent_id: AgentId,
        /// Colliding provider-facing name.
        name: String,
    },
    /// A replacement implementation must preserve the capability category.
    #[error(
        "capability `{capability_id}` has kind {actual:?}; this replacement requires {expected:?}"
    )]
    CapabilityKindMismatch {
        /// Existing immutable capability identity.
        capability_id: CapabilityId,
        /// Kind required by the replacement API.
        expected: CapabilityKind,
        /// Kind recorded on the predecessor capability.
        actual: CapabilityKind,
    },
    /// A handle belongs to a different runtime world.
    #[error("handle belongs to runtime `{actual}`, not `{expected}")]
    ForeignRuntime {
        /// Runtime expected by the receiver.
        expected: RuntimeId,
        /// Runtime carried by the handle.
        actual: RuntimeId,
    },
    /// A handle generation is no longer authoritative.
    #[error("run `{run_id}` generation is stale")]
    StaleHandle {
        /// Stale run identity.
        run_id: RunId,
    },
    /// A tenant attempted to cross an ownership boundary.
    #[error("tenant ownership boundary mismatch")]
    TenantMismatch {
        /// Tenant that owns the target state.
        expected: TenantId,
        /// Tenant presented by the operation.
        actual: TenantId,
    },
    /// The run exhausted its total model-call budget.
    #[error("run exhausted its total model-call budget")]
    ModelCallBudgetExhausted,
    /// A provider model effect failed.
    #[error("model effect failed")]
    ModelEffect(#[source] Arc<ModelEffectError>),
    /// A conversation-memory effect failed.
    #[error("memory effect failed")]
    MemoryEffect(#[source] Arc<MemoryEffectError>),
    /// A structured output could not be recovered within policy bounds.
    #[error("structured output validation failed: {0}")]
    StructuredOutput(String),
    /// A run failed under a typed ECS policy or topology invariant.
    #[error("run failed with code `{code}`: {diagnostic}")]
    RunFailed {
        /// Stable machine-readable terminal code.
        code: String,
        /// Redacted operator-facing diagnostic.
        diagnostic: String,
    },
    /// The bounded ingress channel was closed unexpectedly.
    #[error("the runtime effect ingress channel closed")]
    IngressClosed,
    /// The bounded ingress queue rejected externally injected work.
    #[error("the runtime effect ingress queue is full")]
    IngressFull,
    /// A requested typed provider final did not match the concrete response.
    #[error("provider final type mismatch: expected `{expected}`, received `{actual}`")]
    RawFinalTypeMismatch {
        /// Concrete type requested by the caller.
        expected: &'static str,
        /// Concrete type produced by the provider binding.
        actual: &'static str,
    },
    /// A bounded lifecycle subscriber did not keep up with the producer.
    #[error("runtime event subscriber lagged by {skipped} events")]
    EventStreamLagged {
        /// Number of events evicted before the subscriber observed them.
        skipped: u64,
    },
    /// The convenience streaming collector reached its configured event bound.
    #[error("streaming event collection reached configured capacity {capacity}")]
    EventCollectionLimit {
        /// Maximum events collected by the convenience surface.
        capacity: usize,
    },
    /// Schedule progression exceeded the configured pass bound.
    #[error("the ECS schedule did not reach quiescence within the configured pass bound")]
    Livelock,
    /// No owned effect produced ingress within the configured effect timeout.
    #[error("owned effect produced no ingress before its configured timeout")]
    EffectWaitTimedOut,
    /// The run was explicitly cancelled.
    #[error("run was cancelled")]
    Cancelled,
    /// The run stopped under policy without producing a completion.
    #[error("run stopped under policy")]
    Stopped,
    /// The run completed without canonical assistant text.
    #[error("run completed without canonical assistant text")]
    MissingFinalText,
    /// A protected snapshot operation failed.
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),
}

impl fmt::Debug for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self {
            Self::InvalidConfiguration { .. } => "InvalidConfiguration",
            Self::InvalidAgentSpec { .. } => "InvalidAgentSpec",
            Self::InvalidPrompt { .. } => "InvalidPrompt",
            Self::UnknownModel(_) => "UnknownModel",
            Self::UnknownMemory(_) => "UnknownMemory",
            Self::UnknownAgent(_) => "UnknownAgent",
            Self::UnknownRun(_) => "UnknownRun",
            Self::RunAlreadyDriven(_) => "RunAlreadyDriven",
            Self::UnknownCapability(_) => "UnknownCapability",
            Self::PersistenceIdentityRequired(_) => "PersistenceIdentityRequired",
            Self::UnknownGrant(_) => "UnknownGrant",
            Self::RevokedGrant(_) => "RevokedGrant",
            Self::RetiredCapability(_) => "RetiredCapability",
            Self::CapabilityRevisionOverflow(_) => "CapabilityRevisionOverflow",
            Self::DuplicateToolName { .. } => "DuplicateToolName",
            Self::CapabilityKindMismatch { .. } => "CapabilityKindMismatch",
            Self::ForeignRuntime { .. } => "ForeignRuntime",
            Self::StaleHandle { .. } => "StaleHandle",
            Self::TenantMismatch { .. } => "TenantMismatch",
            Self::ModelCallBudgetExhausted => "ModelCallBudgetExhausted",
            Self::ModelEffect(_) => "ModelEffect",
            Self::MemoryEffect(_) => "MemoryEffect",
            Self::StructuredOutput(_) => "StructuredOutput",
            Self::RunFailed { .. } => "RunFailed",
            Self::IngressClosed => "IngressClosed",
            Self::IngressFull => "IngressFull",
            Self::RawFinalTypeMismatch { .. } => "RawFinalTypeMismatch",
            Self::EventStreamLagged { .. } => "EventStreamLagged",
            Self::EventCollectionLimit { .. } => "EventCollectionLimit",
            Self::Livelock => "Livelock",
            Self::EffectWaitTimedOut => "EffectWaitTimedOut",
            Self::Cancelled => "Cancelled",
            Self::Stopped => "Stopped",
            Self::MissingFinalText => "MissingFinalText",
            Self::Snapshot(_) => "Snapshot",
        };
        formatter
            .debug_struct("RuntimeError")
            .field("kind", &kind)
            .finish()
    }
}

impl RuntimeError {
    /// Inspect a provider response body preserved by a typed model failure.
    #[must_use]
    pub fn provider_response_body(&self) -> Option<&str> {
        match self {
            Self::ModelEffect(error) => error.provider_response_body(),
            _ => None,
        }
    }

    /// Inspect a provider response status preserved by a typed model failure.
    #[must_use]
    pub fn provider_response_status(&self) -> Option<http::StatusCode> {
        match self {
            Self::ModelEffect(error) => error.provider_response_status(),
            _ => None,
        }
    }

    /// Parse a preserved provider response body as JSON.
    pub fn provider_response_json(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        match self {
            Self::ModelEffect(error) => error.provider_response_json(),
            _ => Ok(None),
        }
    }
}

/// Failures while protecting, validating, restoring, or rebinding a snapshot.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SnapshotError {
    /// Snapshot protection failed.
    #[error("snapshot protection failed: {0}")]
    Protect(String),
    /// Snapshot unprotection failed.
    #[error("snapshot unprotection failed: {0}")]
    Unprotect(String),
    /// The protected payload digest did not match.
    #[error("snapshot integrity validation failed")]
    Integrity,
    /// The schema version is unsupported.
    #[error("unsupported snapshot schema version {0}")]
    UnsupportedVersion(u32),
    /// Stable records violate an ownership or relationship invariant.
    #[error("snapshot relationship validation failed: {0}")]
    InvalidRelationship(String),
    /// A required exact model implementation was not rebound.
    #[error("snapshot requires missing model binding `{0}`")]
    MissingModel(ModelId),
    /// A required exact capability implementation was not rebound.
    #[error("snapshot requires missing capability binding `{0}`")]
    MissingCapability(CapabilityId),
    /// A required exact memory implementation was not rebound.
    #[error("snapshot requires missing memory binding `{0}`")]
    MissingMemory(MemoryId),
    /// A live implementation was not assigned a stable persistence identity.
    #[error("snapshot requires an explicit persistence identity for {kind} binding `{id}`")]
    MissingBindingIdentity {
        /// Stable binding identifier.
        id: String,
        /// Binding category.
        kind: &'static str,
    },
    /// Metadata-only persistence cannot safely retain a memory conversation key.
    #[error(
        "metadata-only snapshot cannot represent memory-enabled agent `{0}`; use canonical run state or remove conversation memory"
    )]
    MetadataOnlyMemory(AgentId),
    /// Metadata-only persistence would remove a behavior-bearing system preamble.
    #[error(
        "metadata-only snapshot cannot represent preamble-bearing agent `{0}`; use canonical run state or remove the preamble"
    )]
    MetadataOnlyPreamble(AgentId),
    /// Arbitrary provider parameters are intentionally never persisted.
    #[error(
        "snapshot cannot represent agent `{0}` with provider additional parameters; remove or re-register that configuration before snapshotting"
    )]
    NonPersistableProviderParameters(AgentId),
    /// A rebound implementation has the right stable ID but the wrong concrete type.
    #[error("snapshot binding `{id}` has a mismatched {kind} implementation")]
    BindingMismatch {
        /// Stable binding identity.
        id: String,
        /// Binding category.
        kind: &'static str,
    },
    /// Snapshot serialization failed.
    #[error("snapshot serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    /// The world is not at a safe persistence point.
    #[error("snapshot requires quiescent runs with no active effects")]
    NotQuiescent,
}
