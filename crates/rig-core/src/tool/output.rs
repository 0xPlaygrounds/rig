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
mod debug_tests;

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
mod tests;
