//! Reflection wrappers for shared transport DTOs; typed ECS part fields remain visible.
use bevy_reflect::{ReflectDeserialize, ReflectSerialize, reflect_remote};
use rig_core::message;
use serde::{Deserialize, Serialize};

/// Reflected shared transport value `message::Text`.
#[reflect_remote(message::Text)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextPartReflect {}

/// Reflected shared transport value `message::ToolCall`.
#[reflect_remote(message::ToolCall)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCallPartReflect {}

/// Reflected shared transport value `message::Reasoning`.
#[reflect_remote(message::Reasoning)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReasoningPartReflect {}

/// Reflected shared transport value `serde_json::Value`.
#[reflect_remote(serde_json::Value)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Debug, PartialEq, Serialize, Deserialize)]
pub struct JsonPartReflect {}

/// Reflected shared transport value `Option<message::ImageMediaType>`.
#[reflect_remote(Option<message::ImageMediaType>)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageMediaReflect {}

/// Reflected shared transport value `Option<message::AudioMediaType>`.
#[reflect_remote(Option<message::AudioMediaType>)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioMediaReflect {}

/// Reflected shared transport value `Option<message::VideoMediaType>`.
#[reflect_remote(Option<message::VideoMediaType>)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Debug, PartialEq, Serialize, Deserialize)]
pub struct VideoMediaReflect {}

/// Reflected shared transport value `Option<message::DocumentMediaType>`.
#[reflect_remote(Option<message::DocumentMediaType>)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Debug, PartialEq, Serialize, Deserialize)]
pub struct DocumentMediaReflect {}

/// Reflected shared transport value `Option<message::AdditionalParams>`.
#[reflect_remote(Option<message::AdditionalParams>)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartParamsReflect {}

/// Reflected shared transport value `Option<message::ImageDetail>`.
#[reflect_remote(Option<message::ImageDetail>)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageDetailReflect {}
