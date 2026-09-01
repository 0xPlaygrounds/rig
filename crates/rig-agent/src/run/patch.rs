//! A per-turn request patch: what a driver may change about one model call
//! before [`prepare_request`](super::prepare::prepare_request)
//! binds it — plain data, produced by hooks in rig-agent and by any other
//! driver's equivalent.

use rig_core::completion::Document;
use rig_core::message::{Message, ToolChoice};
use serde::{Deserialize, Serialize};

/// A non-sticky patch applied only to the current turn's completion request.
///
/// A driver's hook stack merges patches in hook registration order according to these
/// rules:
///
/// - `extra_context` documents are appended in order.
/// - JSON-object `additional_params` values are shallow-merged, with later
///   top-level keys winning; a later non-object value replaces an earlier value.
/// - `active_tools` allow-lists are intersected.
/// - Scalar fields and `history` use last-writer-wins semantics, with a warning
///   when multiple hooks set the same field.
///
/// The merged patch does not mutate the agent's configured baseline and is not
/// carried into subsequent turns.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RequestPatch {
    /// Preamble to use instead of the agent's configured preamble for this turn.
    pub preamble: Option<String>,
    /// Sampling temperature to use for this turn.
    pub temperature: Option<f64>,
    /// Maximum output-token count to use for this turn.
    pub max_tokens: Option<u64>,
    /// Tool-choice policy to use for this turn.
    pub tool_choice: Option<ToolChoice>,
    /// Allow-list used to narrow the tools advertised for this turn.
    pub active_tools: Option<Vec<String>>,
    /// Provider-specific request parameters to apply for this turn.
    pub additional_params: Option<serde_json::Value>,
    /// Context documents appended to the request for this turn.
    pub extra_context: Vec<Document>,
    /// Conversation history to use instead of the current history for this turn.
    pub history: Option<Vec<Message>>,
}

fn merge_last_wins<T>(earlier: Option<T>, later: Option<T>, field: &str) -> Option<T> {
    match (earlier, later) {
        (Some(_), Some(later)) => {
            tracing::warn!(
                patch_field = field,
                "two hooks set the same request field; later wins"
            );
            Some(later)
        }
        (earlier, later) => later.or(earlier),
    }
}

impl RequestPatch {
    /// Creates an empty request patch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the agent's configured preamble for this turn.
    pub fn preamble(mut self, value: impl Into<String>) -> Self {
        self.preamble = Some(value.into());
        self
    }

    /// Sets the sampling temperature for this turn.
    pub fn temperature(mut self, value: f64) -> Self {
        self.temperature = Some(value);
        self
    }

    /// Sets the maximum output-token count for this turn.
    pub fn max_tokens(mut self, value: u64) -> Self {
        self.max_tokens = Some(value);
        self
    }

    /// Sets the tool-choice policy for this turn.
    pub fn tool_choice(mut self, value: ToolChoice) -> Self {
        self.tool_choice = Some(value);
        self
    }

    /// Sets the allow-list used to narrow the tools advertised for this turn.
    pub fn active_tools<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.active_tools = Some(values.into_iter().map(Into::into).collect());
        self
    }

    /// Sets provider-specific request parameters for this turn.
    ///
    /// When multiple patches provide JSON objects, their top-level keys are
    /// shallow-merged and values from later hooks win.
    pub fn additional_params(mut self, value: serde_json::Value) -> Self {
        self.additional_params = Some(value);
        self
    }

    /// Appends context documents to the request for this turn.
    pub fn extra_context<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = Document>,
    {
        self.extra_context.extend(values);
        self
    }

    /// Appends one context document to the request for this turn.
    pub fn context(mut self, value: Document) -> Self {
        self.extra_context.push(value);
        self
    }

    /// Replaces the conversation history for this turn.
    pub fn history<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = Message>,
    {
        self.history = Some(values.into_iter().collect());
        self
    }

    /// Whether the patch sets nothing at all.
    pub fn is_empty(&self) -> bool {
        self.preamble.is_none()
            && self.temperature.is_none()
            && self.max_tokens.is_none()
            && self.tool_choice.is_none()
            && self.active_tools.is_none()
            && self.additional_params.is_none()
            && self.extra_context.is_empty()
            && self.history.is_none()
    }

    /// Merge a later patch over this one under the rules documented on the
    /// type: `extra_context` appends, object `additional_params` shallow-merge
    /// (later keys win), `active_tools` intersect, scalars and `history` take
    /// the later value (with a warning when both are set).
    pub fn merge(mut self, later: Self) -> Self {
        self.extra_context.extend(later.extra_context);
        self.additional_params = match (self.additional_params.take(), later.additional_params) {
            (Some(base), Some(patch)) if base.is_object() && patch.is_object() => {
                Some(rig_core::json_utils::merge(base, patch))
            }
            (base, patch) => patch.or(base),
        };
        self.preamble = merge_last_wins(self.preamble, later.preamble, "preamble");
        self.temperature = merge_last_wins(self.temperature, later.temperature, "temperature");
        self.max_tokens = merge_last_wins(self.max_tokens, later.max_tokens, "max_tokens");
        self.tool_choice = merge_last_wins(self.tool_choice, later.tool_choice, "tool_choice");
        self.history = merge_last_wins(self.history, later.history, "history");
        self.active_tools = match (self.active_tools.take(), later.active_tools) {
            (Some(earlier), Some(later)) => {
                let later: std::collections::BTreeSet<_> = later.iter().collect();
                Some(
                    earlier
                        .into_iter()
                        .filter(|name| later.contains(name))
                        .collect(),
                )
            }
            (earlier, later) => earlier.or(later),
        };
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A patch a host may cache in serializable state round-trips.
    #[test]
    fn request_patch_round_trips_through_serde() {
        let patch = RequestPatch {
            preamble: Some("p".to_string()),
            temperature: Some(0.5),
            max_tokens: Some(64),
            tool_choice: Some(ToolChoice::Auto),
            active_tools: Some(vec!["add".to_string()]),
            additional_params: Some(serde_json::json!({"k": 1})),
            extra_context: Vec::new(),
            history: Some(vec![Message::user("hi")]),
        };
        let json = serde_json::to_string(&patch).expect("serialize patch");
        assert_eq!(
            serde_json::from_str::<RequestPatch>(&json).expect("deserialize patch"),
            patch
        );
    }
}
