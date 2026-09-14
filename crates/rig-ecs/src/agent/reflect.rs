//! The `reflect` feature's wrappers for the rig-core types the agent's
//! components hold (the bus's wrappers, [`crate::bus::reflect`], cover the
//! rest): opaque remote wrappers, serialized through their serde form.

use bevy_reflect::{ReflectDeserialize, ReflectSerialize, prelude::ReflectDefault, reflect_remote};
use rig_core::completion::{
    Usage,
    message::{AssistantContent, Message, ProviderCallId, ToolCallId, ToolChoice},
};
use serde::{Deserialize, Serialize};

pub use crate::bus::reflect::ToolContextReflect;

/// A `serde_json::Value`, reflected.
#[reflect_remote(serde_json::Value)]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum JsonReflect {}

/// An `Option<serde_json::Value>`, reflected.
#[reflect_remote(Option<serde_json::Value>)]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum OptionalJsonReflect {}

/// An `Option<ToolChoice>`, reflected.
#[reflect_remote(Option<ToolChoice>)]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum ToolChoiceReflect {}

/// The wire [`Usage`], reflected.
#[reflect_remote(Usage)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageReflect {}

/// A [`ToolCallId`], reflected.
#[reflect_remote(ToolCallId)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCallIdReflect {}

/// An `Option<ProviderCallId>`, reflected.
#[reflect_remote(Option<ProviderCallId>)]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum ProviderCallIdReflect {}

/// An assistant turn's parts, `Vec<AssistantContent>`, reflected.
#[reflect_remote(Vec<AssistantContent>)]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AssistantContentsReflect {}

/// A [`Message`], reflected.
#[reflect_remote(Message)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Debug, PartialEq, Serialize, Deserialize)]
pub enum MessageReflect {}
