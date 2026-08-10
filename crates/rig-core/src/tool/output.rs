//! Canonical model-visible tool output.

use std::{any::Any, fmt};

use serde::Serialize;

use crate::{message::ToolResultContent, tool::ToolExecutionError};

/// The canonical model-visible output produced by a tool.
///
/// Every output is stored as one or more typed [`ToolResultContent`] blocks.
/// Ordinary serializable Rust values are converted through [`IntoToolOutput`]:
/// values that serialize as JSON strings become literal text blocks and all
/// other values become structured JSON blocks. An explicit
/// [`serde_json::Value`], including a JSON string, stays JSON. Multimodal tools
/// opt in explicitly with [`Self::content`]. Rig never reparses text as JSON to
/// guess whether it represents rich content.
#[derive(Clone, PartialEq)]
pub struct ToolOutput {
    content: Vec<ToolResultContent>,
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
    ///
    /// Rejects an empty list. On `main` the argument type made emptiness
    /// unrepresentable; as a `Vec` the check lives here instead, because this
    /// is the one funnel every multi-block construction passes through —
    /// tools returning [`ToolOutput`] directly and hooks rewriting one
    /// included. A zero-block tool result cannot be sent (the request
    /// boundary rejects it), so rejecting at construction surfaces the
    /// defect where it is made rather than one request later. A tool with a
    /// genuinely empty result returns one empty text block ([`Self::text`]
    /// with `""`).
    pub fn content(content: Vec<ToolResultContent>) -> Result<Self, ToolExecutionError> {
        let content = crate::message::require_non_empty(content, || {
            ToolExecutionError::other(
                "tool output has no content blocks; return at least one block — \
                 an empty text block is valid",
            )
        })?;
        Ok(Self { content })
    }

    /// Construct one explicit model-content block.
    pub fn one(content: ToolResultContent) -> Self {
        Self {
            content: vec![content],
        }
    }

    /// Return literal text when this output is exactly one plain text block.
    pub fn as_text(&self) -> Option<&str> {
        if self.content.len() != 1 {
            return None;
        }

        match self.content.first()? {
            // `Some` params always carry data (`AdditionalParams` is
            // non-empty by construction), so plain `is_none` is the whole
            // annotation check.
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

        match self.content.first()? {
            ToolResultContent::Json { value } => Some(value),
            ToolResultContent::Text(_) | ToolResultContent::Image(_) => None,
        }
    }

    /// Borrow the canonical ordered content blocks.
    pub fn as_content(&self) -> &[ToolResultContent] {
        &self.content
    }

    /// Convert this output into the canonical message content sent to a model.
    pub fn into_content(self) -> Vec<ToolResultContent> {
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

impl TryFrom<Vec<ToolResultContent>> for ToolOutput {
    type Error = ToolExecutionError;

    // `From` on `main` — the source type was non-empty by construction, so the
    // conversion could not fail. With `Vec` the emptiness check makes it
    // fallible; a `From` here would be the unguarded bypass around
    // [`ToolOutput::content`].
    fn try_from(content: Vec<ToolResultContent>) -> Result<Self, Self::Error> {
        Self::content(content)
    }
}

/// Conversion into Rig's canonical tool output.
///
/// A blanket implementation keeps ordinary [`Serialize`] outputs ergonomic.
/// Because that blanket implementation already covers every serializable type,
/// it cannot be overridden with another implementation for a serializable
/// custom type. Return [`ToolOutput`] from [`PortableTool::call`](crate::tool::PortableTool::call)
/// when that type needs a custom presentation. Implement this trait directly
/// only for output types that do not implement [`Serialize`].
pub trait IntoToolOutput {
    /// Convert this value without routing structured data through a string.
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError>;
}

#[cfg(test)]
mod debug_tests {
    use crate::message::ImageMediaType;

    use super::*;

    #[test]
    fn debug_reports_shape_without_tool_content() {
        let output = ToolOutput::content(vec![
            ToolResultContent::text("Bearer secret-tool-output"),
            ToolResultContent::json(serde_json::json!({
                "credential": "secret-json-output"
            })),
            ToolResultContent::image_base64("secret-image-output", Some(ImageMediaType::PNG), None),
        ])
        .expect("fixture content is non-empty");

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

impl<T> IntoToolOutput for T
where
    T: Serialize + 'static,
{
    fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
        // `ToolResultContent` and `Vec<ToolResultContent>` are serializable
        // because they also serve as transcript types. They nevertheless mean
        // explicit rich output here; serializing them through the fallback would
        // silently turn an image into a JSON object. Stable Rust cannot express
        // a blanket `Serialize` impl with negative exceptions, so preserve these
        // two canonical rich types before taking the serialization path.
        let value = &self as &dyn Any;
        if let Some(content) = value.downcast_ref::<ToolResultContent>() {
            return Ok(ToolOutput::one(content.clone()));
        }
        if let Some(content) = value.downcast_ref::<Vec<ToolResultContent>>() {
            // `ToolOutput::content` rejects an empty list, so an empty
            // rich-content return surfaces here as a normal tool failure the
            // agent feeds back to the model, instead of a zero-block result
            // entering history and aborting the whole run at the next
            // request's boundary validation. Deliberately not normalized to
            // an empty text block: inventing content the tool never produced
            // is the fabrication this crate removed.
            return ToolOutput::content(content.clone());
        }
        let is_explicit_json = value.is::<serde_json::Value>();

        serde_json::to_value(self)
            .map(|value| match value {
                serde_json::Value::String(text) if !is_explicit_json => ToolOutput::text(text),
                value => ToolOutput::json(value),
            })
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
    fn an_empty_content_list_cannot_become_a_tool_output() {
        // A zero-block tool result cannot be sent — the request boundary
        // rejects it — so the failure surfaces at construction as an ordinary
        // tool error instead of aborting the run one request later. Every
        // route is closed: the rich-content tool return, the explicit
        // constructor, and the fallible conversion. One empty text block, by
        // contrast, is a legitimate empty result and passes.
        let error = Vec::<ToolResultContent>::new()
            .into_tool_output()
            .expect_err("an empty rich-content list must not become a ToolOutput");
        assert!(error.to_string().contains("no content blocks"));

        assert!(ToolOutput::content(Vec::new()).is_err());
        assert!(ToolOutput::try_from(Vec::<ToolResultContent>::new()).is_err());

        let output = vec![ToolResultContent::text("")]
            .into_tool_output()
            .unwrap();
        assert_eq!(output, ToolOutput::text(""));
    }

    #[test]
    fn json_shaped_strings_remain_literal_text() {
        let text = r#"{"type":"image","data":"not-an-envelope"}"#.to_string();
        let output = text.clone().into_tool_output().unwrap();

        assert_eq!(output, ToolOutput::text(text.clone()));
        let content = output.into_content();
        assert!(
            matches!(content.first(), Some(ToolResultContent::Text(value)) if value.text == text)
        );
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
            Some(ToolResultContent::Json { value: content_value }) if *content_value == value
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
            Some(ToolResultContent::Image(image))
                if image.media_type == Some(ImageMediaType::JPEG)
                    && matches!(&image.data, DocumentSourceKind::Base64(data) if data == "base64data==")
        ));
    }

    #[test]
    fn direct_ordered_content_is_not_serialized_as_json() {
        let content = vec![
            ToolResultContent::text("before"),
            ToolResultContent::image_base64("base64data==", Some(ImageMediaType::PNG), None),
            ToolResultContent::json(serde_json::json!({"after": true})),
        ];

        let output = content.clone().into_tool_output().unwrap();

        assert_eq!(output.as_content(), &content);
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
