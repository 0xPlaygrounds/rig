//! The reflection wrappers for the rig-core types the agent's components
//! hold (the bus's wrappers, [`crate::bus::reflect`], cover the rest):
//! opaque remote wrappers, serialized through their serde form.
//!
//! ```
//! let mut registry = bevy_reflect::TypeRegistry::default();
//! registry.register::<rig_ecs::agent::reflect::UsageReflect>();
//! ```

use bevy_reflect::{ReflectDeserialize, ReflectSerialize, prelude::ReflectDefault};
use rig_core::completion::{
    Usage,
    message::{AssistantContent, Message, ProviderCallId, ToolCallId, ToolChoice},
};
use serde::{Deserialize, Serialize};

pub use crate::bus::reflect::ToolContextReflect;

crate::reflect::opaque_reflect! {
    /// A `serde_json::Value`, reflected.
    enum JsonReflect(serde_json::Value): Default, PartialEq;
    /// An `Option<serde_json::Value>`, reflected.
    enum OptionalJsonReflect(Option<serde_json::Value>): Default, PartialEq;
    /// An `Option<ToolChoice>`, reflected.
    enum ToolChoiceReflect(Option<ToolChoice>): Default, PartialEq;
    /// A [`ToolCallId`], reflected.
    struct ToolCallIdReflect(ToolCallId): PartialEq;
    /// An `Option<ProviderCallId>`, reflected.
    enum ProviderCallIdReflect(Option<ProviderCallId>): Default, PartialEq;
    /// An assistant turn's parts, `Vec<AssistantContent>`, reflected.
    struct AssistantContentsReflect(Vec<AssistantContent>): Default, PartialEq;
    /// A [`Message`], reflected.
    enum MessageReflect(Message): PartialEq;
}

/// The wire [`Usage`], reflected.
#[bevy_reflect::reflect_remote(Usage)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageReflect {}
