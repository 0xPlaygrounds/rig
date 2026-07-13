//! Canonical model-visible tool output.

use serde::Serialize;

use crate::{OneOrMany, message::ToolResultContent, tool::ToolExecutionError};

/// The canonical model-visible output produced by a tool.
///
/// Ordinary serializable Rust values are converted through [`IntoToolOutput`]:
/// strings remain literal text and all other values remain structured JSON.
/// Multimodal tools opt in explicitly with [`ToolOutput::content`]. Rig never
/// reparses text as JSON to guess whether it represents rich content.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ToolOutput {
    /// Literal model-visible text.
    Text(String),
    /// Structured JSON, rendered as JSON text for providers that accept only
    /// text tool results.
    Json(serde_json::Value),
    /// Explicit typed content blocks.
    Content(OneOrMany<ToolResultContent>),
}

impl ToolOutput {
    /// Construct literal text output.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Construct structured JSON output.
    ///
    /// A JSON string is normalized to [`Self::Text`] so strings can never be
    /// mistaken for tagged rich-content envelopes.
    pub fn json(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::String(text) => Self::Text(text),
            value => Self::Json(value),
        }
    }

    /// Construct explicit model content.
    pub fn content(content: OneOrMany<ToolResultContent>) -> Self {
        Self::Content(content)
    }

    /// Construct one explicit model-content block.
    pub fn one(content: ToolResultContent) -> Self {
        Self::Content(OneOrMany::one(content))
    }

    /// Return literal text when this output is text.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Json(_) | Self::Content(_) => None,
        }
    }

    /// Convert this output into the canonical message content sent to a model.
    pub fn into_content(self) -> OneOrMany<ToolResultContent> {
        match self {
            Self::Text(text) => OneOrMany::one(ToolResultContent::text(text)),
            Self::Json(value) => OneOrMany::one(ToolResultContent::json(value)),
            Self::Content(content) => content,
        }
    }

    /// Render a stable text representation for telemetry and diagnostics.
    ///
    /// This is a terminal rendering operation; the returned text is never used
    /// to reconstruct structured output.
    pub fn render(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Json(value) => value.to_string(),
            Self::Content(content) => serde_json::to_string(content)
                .unwrap_or_else(|_| "<structured tool output>".to_string()),
        }
    }
}

impl From<String> for ToolOutput {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for ToolOutput {
    fn from(text: &str) -> Self {
        Self::Text(text.to_string())
    }
}

impl From<serde_json::Value> for ToolOutput {
    fn from(value: serde_json::Value) -> Self {
        Self::json(value)
    }
}

impl From<ToolResultContent> for ToolOutput {
    fn from(content: ToolResultContent) -> Self {
        Self::one(content)
    }
}

impl From<OneOrMany<ToolResultContent>> for ToolOutput {
    fn from(content: OneOrMany<ToolResultContent>) -> Self {
        Self::content(content)
    }
}

/// Conversion into Rig's canonical tool output.
///
/// A blanket implementation keeps ordinary [`Serialize`] outputs ergonomic.
/// Tool authors only need to name this trait when implementing a custom output
/// conversion; explicit multimodal output uses [`ToolOutput`] directly.
pub trait IntoToolOutput {
    /// Convert this value without routing structured data through a string.
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError>;
}

impl<T> IntoToolOutput for T
where
    T: Serialize,
{
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
        serde_json::to_value(self)
            .map(ToolOutput::json)
            .map_err(|error| {
                ToolExecutionError::other(format!("failed to serialize tool output: {error}"))
                    .with_source(error)
            })
    }
}

impl IntoToolOutput for ToolOutput {
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::message::{DocumentSourceKind, ImageMediaType};

    use super::*;

    #[test]
    fn json_shaped_strings_remain_literal_text() {
        let text = r#"{"type":"image","data":"not-an-envelope"}"#.to_string();
        let output = text.clone().into_tool_output().unwrap();

        assert_eq!(output, ToolOutput::Text(text.clone()));
        let content = output.into_content();
        assert!(matches!(content.first(), ToolResultContent::Text(value) if value.text == text));
    }

    #[test]
    fn structured_values_remain_json_until_terminal_rendering() {
        let value = serde_json::json!({"status": "ok", "count": 2});
        let output = value.clone().into_tool_output().unwrap();

        assert_eq!(output, ToolOutput::Json(value.clone()));
        assert_eq!(output.render(), value.to_string());
        let content = output.into_content();
        assert!(matches!(
            content.first(),
            ToolResultContent::Json { value: content_value } if content_value == value
        ));
    }

    #[test]
    fn explicit_image_content_preserves_its_type() {
        let output = ToolOutput::one(ToolResultContent::image_base64(
            "base64data==",
            Some(ImageMediaType::JPEG),
            None,
        ));

        let content = output.into_content();
        assert!(matches!(
            content.first(),
            ToolResultContent::Image(image)
                if image.media_type == Some(ImageMediaType::JPEG)
                    && matches!(&image.data, DocumentSourceKind::Base64(data) if data == "base64data==")
        ));
    }
}
