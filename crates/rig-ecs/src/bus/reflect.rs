//! The `reflect` feature's wrappers for the rig-core types the bus's
//! components hold, so an inspector shows an effect entity's payload:
//! `bevy_reflect` remote wrappers, each opaque — the value reflects as a
//! whole, serialized through its serde form (`ReflectSerialize` /
//! `ReflectDeserialize`), which is the wire form the log already has. The
//! runtime-only components (`Serving`, `Streaming`, `Publishing`, `Asked`,
//! `Answer`, `Typed`) reflect nothing: a task is not data.

use bevy_reflect::{ReflectDeserialize, ReflectSerialize, prelude::ReflectDefault, reflect_remote};
use rig_core::{
    effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome},
    error::ErrorReport,
    streaming::StreamEvent,
    tool::ToolContext,
};
use serde::{Deserialize, Serialize};

/// [`HandlerKey`], reflected.
#[reflect_remote(HandlerKey)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, PartialEq, Serialize, Deserialize)]
pub struct HandlerKeyReflect {}

/// [`EffectKind`], reflected.
#[reflect_remote(EffectKind)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, Serialize, Deserialize)]
pub enum EffectKindReflect {}

/// [`EffectId`], reflected.
#[reflect_remote(EffectId)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, PartialEq, Serialize, Deserialize)]
pub struct EffectIdReflect {}

/// [`HandlerDescriptor`], reflected.
#[reflect_remote(HandlerDescriptor)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, PartialEq, Serialize, Deserialize)]
pub struct HandlerDescriptorReflect {}

/// [`ToolContext`], reflected.
#[reflect_remote(ToolContext)]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolContextReflect {}

/// An effect's answer, `Result<Outcome, ErrorReport>`, reflected.
#[reflect_remote(Result<Outcome, ErrorReport>)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, Serialize, Deserialize)]
pub enum OutcomeReflect {}

/// A stream's answer so far, `Option<Result<Outcome, ErrorReport>>`, reflected.
#[reflect_remote(Option<Result<Outcome, ErrorReport>>)]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, Default, Serialize, Deserialize)]
pub enum StreamedOutcomeReflect {}

/// A stream's events, `Vec<StreamEvent>`, reflected.
#[reflect_remote(Vec<StreamEvent>)]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamEventsReflect {}
