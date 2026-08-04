//! Canonical model-visible tool output.

use std::fmt;

use serde::Serialize;

use crate::{OneOrMany, message::ToolResultContent, tool::ToolExecutionError};

/// The canonical model-visible output produced by a tool.
///
/// Every output is stored as one or more typed [`ToolResultContent`] blocks.
/// Ordinary serializable Rust values are converted through
/// [`serialize_to_tool_output`]: values that serialize as JSON strings become
/// literal text blocks and all other values become structured JSON blocks. An
/// explicit [`serde_json::Value`], including a JSON string, stays JSON.
/// Multimodal tools opt in explicitly with [`Self::content`]. Rig never
/// reparses text as JSON to guess whether it represents rich content.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct ToolOutput {
    content: OneOrMany<ToolResultContent>,
}

impl fmt::Debug for ToolOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let content_kinds = self
            .content
            .iter()
            .map(|content| match content {
                ToolResultContent::Text(_) => "text",
                ToolResultContent::Image(_) => "image",
                ToolResultContent::Json { .. } => "json",
            })
            .collect::<Vec<_>>();
        formatter
            .debug_struct("ToolOutput")
            .field("content_count", &self.content.len())
            .field("content_kinds", &content_kinds)
            .finish()
    }
}

impl ToolOutput {
    /// Construct literal text output.
    pub fn text(text: impl Into<String>) -> Self {
        Self::one(ToolResultContent::text(text))
    }

    /// Construct structured JSON output.
    ///
    /// Unlike an ordinary Rust string tool output, an explicit JSON string stays
    /// a JSON content block.
    pub fn json(value: serde_json::Value) -> Self {
        Self::one(ToolResultContent::json(value))
    }

    /// Construct explicit model content.
    pub fn content(content: OneOrMany<ToolResultContent>) -> Self {
        Self { content }
    }

    /// Construct one explicit model-content block.
    pub fn one(content: ToolResultContent) -> Self {
        Self::content(OneOrMany::one(content))
    }

    /// Return literal text when this output is exactly one plain text block.
    pub fn as_text(&self) -> Option<&str> {
        if self.content.len() != 1 {
            return None;
        }

        match self.content.first_ref() {
            ToolResultContent::Text(text) if text.additional_params.is_none() => Some(&text.text),
            ToolResultContent::Text(_)
            | ToolResultContent::Image(_)
            | ToolResultContent::Json { .. } => None,
        }
    }

    /// Return structured JSON when this output is exactly one JSON block.
    pub fn as_json(&self) -> Option<&serde_json::Value> {
        if self.content.len() != 1 {
            return None;
        }

        match self.content.first_ref() {
            ToolResultContent::Json { value } => Some(value),
            ToolResultContent::Text(_) | ToolResultContent::Image(_) => None,
        }
    }

    /// Borrow the canonical ordered content blocks.
    pub fn as_content(&self) -> &OneOrMany<ToolResultContent> {
        &self.content
    }

    /// Convert this output into the canonical message content sent to a model.
    pub fn into_content(self) -> OneOrMany<ToolResultContent> {
        self.content
    }

    /// Render a stable text representation for telemetry and diagnostics.
    ///
    /// This is a terminal rendering operation; the returned text is never used
    /// to reconstruct structured output.
    pub fn render(&self) -> String {
        if let Some(text) = self.as_text() {
            text.to_string()
        } else if let Some(value) = self.as_json() {
            value.to_string()
        } else {
            serde_json::to_string(&self.content)
                .unwrap_or_else(|_| "<structured tool output>".to_string())
        }
    }
}

impl From<String> for ToolOutput {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for ToolOutput {
    fn from(text: &str) -> Self {
        Self::text(text)
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

/// Serialize an ordinary Rust value into canonical tool output.
///
/// This is the one serialization path for plain data: a value that serializes
/// as a JSON string becomes a literal text block, and every other value
/// becomes a structured JSON block. Construct [`ToolOutput::json`] directly
/// when a JSON string must stay JSON, and use the rich-content constructors
/// ([`ToolOutput::content`], [`ToolOutput::one`]) for multimodal output —
/// rich content must never be routed through this function.
pub fn serialize_to_tool_output<T: Serialize + ?Sized>(
    value: &T,
) -> Result<ToolOutput, ToolExecutionError> {
    serde_json::to_value(value)
        .map(|value| match value {
            serde_json::Value::String(text) => ToolOutput::text(text),
            value => ToolOutput::json(value),
        })
        .map_err(|error| {
            ToolExecutionError::other(format!("failed to serialize tool output: {error}"))
                .with_source(error)
        })
}

/// Conversion into Rig's canonical tool output.
///
/// This trait is implemented explicitly, never blanket-implemented, so the
/// rich output types keep identity semantics without any runtime type
/// inspection. Rig provides implementations for [`ToolOutput`] (identity),
/// [`ToolResultContent`] and `OneOrMany<ToolResultContent>` (explicit rich
/// content), [`serde_json::Value`] (stays JSON, including JSON strings),
/// strings (literal text), and plain data (`()`, `bool`, numbers, `char`,
/// `Option<T>`, and `Vec<T>` via [`serialize_to_tool_output`]).
///
/// Implement it for a custom serializable output type with one call to
/// [`serialize_to_tool_output`]:
///
/// ```
/// use rig_core::tool::{IntoToolOutput, ToolExecutionError, ToolOutput, serialize_to_tool_output};
///
/// #[derive(serde::Serialize)]
/// struct Sum {
///     value: i64,
/// }
///
/// impl IntoToolOutput for Sum {
///     fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
///         serialize_to_tool_output(&self)
///     }
/// }
/// ```
pub trait IntoToolOutput {
    /// Convert this value without routing structured data through a string.
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError>;
}

impl IntoToolOutput for ToolOutput {
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
        Ok(self)
    }
}

impl IntoToolOutput for ToolResultContent {
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
        Ok(ToolOutput::one(self))
    }
}

impl IntoToolOutput for OneOrMany<ToolResultContent> {
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
        Ok(ToolOutput::content(self))
    }
}

impl IntoToolOutput for serde_json::Value {
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
        // An explicit JSON value stays JSON, including a JSON string.
        Ok(ToolOutput::json(self))
    }
}

impl IntoToolOutput for String {
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
        Ok(ToolOutput::text(self))
    }
}

impl IntoToolOutput for &str {
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
        Ok(ToolOutput::text(self))
    }
}

/// Plain-data implementations: everything routes through the one
/// serialization path.
macro_rules! serialize_into_tool_output {
    ($($ty:ty),* $(,)?) => {$(
        impl IntoToolOutput for $ty {
            fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
                serialize_to_tool_output(&self)
            }
        }
    )*};
}

serialize_into_tool_output!(
    (),
    bool,
    char,
    i8,
    i16,
    i32,
    i64,
    i128,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    isize,
    f32,
    f64,
);

/// Tuple outputs are plain data: serialized as JSON arrays, exactly as any
/// serializable aggregate.
macro_rules! serialize_tuple_into_tool_output {
    ($(($($name:ident),+)),* $(,)?) => {$(
        impl<$($name: Serialize),+> IntoToolOutput for ($($name,)+) {
            fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
                serialize_to_tool_output(&self)
            }
        }
    )*};
}

serialize_tuple_into_tool_output!(
    (A),
    (A, B),
    (A, B, C),
    (A, B, C, D),
    (A, B, C, D, E),
    (A, B, C, D, E, F),
    (A, B, C, D, E, F, G),
    (A, B, C, D, E, F, G, H),
);

impl<T: Serialize> IntoToolOutput for Vec<T> {
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
        serialize_to_tool_output(&self)
    }
}

impl<T: Serialize> IntoToolOutput for Option<T> {
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
        serialize_to_tool_output(&self)
    }
}

#[cfg(test)]
mod debug_tests {
    use crate::message::ImageMediaType;

    use super::*;

    #[test]
    fn debug_reports_shape_without_tool_content() {
        let output = ToolOutput::content(
            OneOrMany::many(vec![
                ToolResultContent::text("Bearer secret-tool-output"),
                ToolResultContent::json(serde_json::json!({
                    "credential": "secret-json-output"
                })),
                ToolResultContent::image_base64(
                    "secret-image-output",
                    Some(ImageMediaType::PNG),
                    None,
                ),
            ])
            .unwrap(),
        );

        let debug = format!("{output:?}");
        assert!(debug.contains("content_count: 3"));
        assert!(debug.contains("text"));
        assert!(debug.contains("json"));
        assert!(debug.contains("image"));
        for secret in [
            "secret-tool-output",
            "secret-json-output",
            "secret-image-output",
        ] {
            assert!(!debug.contains(secret));
        }
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

        assert_eq!(output, ToolOutput::text(text.clone()));
        let content = output.into_content();
        assert!(matches!(content.first(), ToolResultContent::Text(value) if value.text == text));
    }

    #[test]
    fn structured_values_remain_json_until_terminal_rendering() {
        let value = serde_json::json!({"status": "ok", "count": 2});
        let output = value.clone().into_tool_output().unwrap();

        assert_eq!(output, ToolOutput::json(value.clone()));
        assert_eq!(output.render(), value.to_string());
        let content = output.into_content();
        assert!(matches!(
            content.first(),
            ToolResultContent::Json { value: content_value } if content_value == value
        ));
    }

    #[test]
    fn explicit_json_string_is_distinct_from_literal_text() {
        let explicit = serde_json::Value::String("hello".to_string());

        let json_output = explicit.clone().into_tool_output().unwrap();
        let text_output = "hello".to_string().into_tool_output().unwrap();

        assert_eq!(json_output, ToolOutput::json(explicit.clone()));
        assert_eq!(json_output.as_json(), Some(&explicit));
        assert_eq!(json_output.as_text(), None);
        assert_eq!(text_output, ToolOutput::text("hello"));
        assert_eq!(text_output.as_text(), Some("hello"));
    }

    #[test]
    fn explicit_image_content_preserves_its_type() {
        let image =
            ToolResultContent::image_base64("base64data==", Some(ImageMediaType::JPEG), None);
        let output = image.into_tool_output().unwrap();

        let content = output.into_content();
        assert!(matches!(
            content.first(),
            ToolResultContent::Image(image)
                if image.media_type == Some(ImageMediaType::JPEG)
                    && matches!(&image.data, DocumentSourceKind::Base64(data) if data == "base64data==")
        ));
    }

    #[test]
    fn direct_ordered_content_is_not_serialized_as_json() {
        let content = OneOrMany::many(vec![
            ToolResultContent::text("before"),
            ToolResultContent::image_base64("base64data==", Some(ImageMediaType::PNG), None),
            ToolResultContent::json(serde_json::json!({"after": true})),
        ])
        .unwrap();

        let output = content.clone().into_tool_output().unwrap();

        assert_eq!(output.as_content(), &content);
    }

    #[test]
    fn tool_output_passes_through_the_blanket_impl_unchanged() {
        let output = ToolOutput::content(
            OneOrMany::many(vec![
                ToolResultContent::text("before"),
                ToolResultContent::image_base64("base64data==", Some(ImageMediaType::PNG), None),
            ])
            .unwrap(),
        );

        assert_eq!(output.clone().into_tool_output().unwrap(), output);
    }

    #[test]
    fn tool_output_serde_round_trips() {
        let output = ToolOutput::content(
            OneOrMany::many(vec![
                ToolResultContent::text("before"),
                ToolResultContent::image_base64("base64data==", Some(ImageMediaType::PNG), None),
                ToolResultContent::json(serde_json::json!({"after": true})),
            ])
            .unwrap(),
        );

        let json = serde_json::to_value(&output).unwrap();
        let restored: ToolOutput = serde_json::from_value(json).unwrap();
        assert_eq!(restored, output);

        // Transparent representation: a ToolOutput serializes as its content.
        assert_eq!(
            serde_json::to_value(&output).unwrap(),
            serde_json::to_value(output.as_content()).unwrap()
        );
    }

    #[test]
    fn singleton_plain_content_has_one_canonical_representation() {
        assert_eq!(
            ToolOutput::text("hello"),
            ToolOutput::one(ToolResultContent::text("hello"))
        );
        assert_eq!(
            ToolOutput::json(serde_json::json!({"ok": true})),
            ToolOutput::one(ToolResultContent::json(serde_json::json!({"ok": true})))
        );
    }
}
