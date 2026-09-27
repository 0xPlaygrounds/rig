//! Opaque reflection wrappers for rig-core values stored in bus components.
//!
//! Values reflect through their serde forms. Runtime tasks, answer inboxes, and
//! observation bookkeeping remain transient and are not checkpointed.
//!
//! ```
//! let mut registry = bevy_reflect::TypeRegistry::default();
//! registry.register::<rig_ecs::bus::reflect::HandlerKeyReflect>();
//! ```

use bevy_reflect::{ReflectDeserialize, ReflectSerialize, prelude::ReflectDefault};
use rig_core::{
    effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome},
    error::RigError,
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
    /// [`ToolContext`], reflected.
    struct ToolContextReflect(ToolContext): Default, PartialEq;
    /// An effect's answer, `Result<Outcome, RigError>`, reflected.
    enum OutcomeReflect(Result<Outcome, RigError>):;
    /// A stream's answer so far, `Option<Result<Outcome, RigError>>`, reflected.
    enum StreamedOutcomeReflect(Option<Result<Outcome, RigError>>): Default;
    /// A stream's events, `Vec<StreamEvent>`, reflected.
    struct StreamEventsReflect(Vec<StreamEvent>): Default, PartialEq;
    /// Stream error reports with their item positions, reflected.
    struct StreamErrorsReflect(Vec<(usize, RigError)>): Default, PartialEq;
}
