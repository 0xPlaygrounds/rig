//! Reflection wrappers for shared transport DTOs; typed ECS part fields remain visible.
//!
//! ```
//! let mut registry = bevy_reflect::TypeRegistry::default();
//! registry.register::<rig_ecs::agent::content::reflect::TextPartReflect>();
//! ```

use bevy_reflect::{ReflectDeserialize, ReflectSerialize};
use rig_core::message;

crate::reflect::opaque_reflect! {
    /// Reflected shared transport value `message::Text`.
    struct TextPartReflect(message::Text): PartialEq;
    /// Reflected shared transport value `message::ToolCall`.
    struct ToolCallPartReflect(message::ToolCall): PartialEq;
    /// Reflected shared transport value `message::Reasoning`.
    struct ReasoningPartReflect(message::Reasoning): PartialEq;
    /// Reflected shared transport value `serde_json::Value`.
    struct JsonPartReflect(serde_json::Value): PartialEq;
    /// Reflected shared transport value `Option<message::ImageMediaType>`.
    struct ImageMediaReflect(Option<message::ImageMediaType>): PartialEq;
    /// Reflected shared transport value `Option<message::AudioMediaType>`.
    struct AudioMediaReflect(Option<message::AudioMediaType>): PartialEq;
    /// Reflected shared transport value `Option<message::VideoMediaType>`.
    struct VideoMediaReflect(Option<message::VideoMediaType>): PartialEq;
    /// Reflected shared transport value `Option<message::DocumentMediaType>`.
    struct DocumentMediaReflect(Option<message::DocumentMediaType>): PartialEq;
    /// Reflected shared transport value `Option<message::AdditionalParams>`.
    struct PartParamsReflect(Option<message::AdditionalParams>): PartialEq;
    /// Reflected shared transport value `Option<message::ImageDetail>`.
    struct ImageDetailReflect(Option<message::ImageDetail>): PartialEq;
}
