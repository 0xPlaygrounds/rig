//! The reflection wrappers for the rig-core types the bus's components
//! hold, so an inspector shows an effect entity's payload and a checkpoint
//! saves it: `bevy_reflect` remote wrappers, each opaque — the value
//! reflects as a whole, serialized through its serde form
//! (`ReflectSerialize` / `ReflectDeserialize`), which is the wire form the
//! log already has. The runtime-only components (`Serving`, `Streaming`,
//! `Handler`, `Publishing`, `Asked`, `Answer`, `WorldOutcome`,
//! `CollectedOutcome`, `Typed`, and the witness's `Refused`, `SeenOutcome`)
//! reflect nothing: tasks, answer inboxes and observation bookkeeping are
//! transient.

use bevy_reflect::{ReflectDeserialize, ReflectSerialize, prelude::ReflectDefault};
use rig_core::{
    effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome},
    error::ErrorReport,
    providers::registry::ProviderRef,
    streaming::StreamEvent,
    tool::ToolContext,
};

crate::reflect::opaque_reflect! {
    /// [`HandlerKey`], reflected.
    struct HandlerKeyReflect(HandlerKey): PartialEq;
    /// [`EffectKind`], reflected.
    enum EffectKindReflect(EffectKind):;
    /// [`EffectId`], reflected.
    struct EffectIdReflect(EffectId): PartialEq;
    /// [`HandlerDescriptor`], reflected.
    struct HandlerDescriptorReflect(HandlerDescriptor): PartialEq;
    /// A [`ProviderBinding`](super::ProviderBinding)'s
    /// [`ProviderRef`], reflected: the provider selection or explicit
    /// configuration travels as a whole, through the serde form a scene
    /// already writes.
    struct ProviderRefReflect(ProviderRef): PartialEq;
    /// [`ToolContext`], reflected.
    struct ToolContextReflect(ToolContext): Default, PartialEq;
    /// An effect's answer, `Result<Outcome, ErrorReport>`, reflected.
    enum OutcomeReflect(Result<Outcome, ErrorReport>):;
    /// A stream's answer so far, `Option<Result<Outcome, ErrorReport>>`, reflected.
    enum StreamedOutcomeReflect(Option<Result<Outcome, ErrorReport>>): Default;
    /// A stream's events, `Vec<StreamEvent>`, reflected.
    struct StreamEventsReflect(Vec<StreamEvent>): Default, PartialEq;
    /// Stream error reports with their item positions, reflected.
    struct StreamErrorsReflect(Vec<(usize, ErrorReport)>): Default, PartialEq;
}
