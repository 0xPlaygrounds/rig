//! What GenerateContent's stream adds to its decoder: the record a streamed
//! reply keeps as `raw`, the frames that carry only a response id, and the
//! observation projection that reads verdicts and usage off each frame.

use serde::Deserialize;

use super::api;
use crate::observe::ObservedError;
use crate::providers::internal::wire;
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, ObservationSink, WireEvent, WireFrame,
};

/// The `raw` a reply records: its usage, finish and identity under
/// Google's own field names.
pub(crate) fn summary(
    usage: Option<api::UsageMetadata>,
    finish_reason: Option<api::FinishReason>,
    finish_message: Option<String>,
    model_version: Option<String>,
    response_id: Option<String>,
) -> serde_json::Value {
    let mut summary = serde_json::Map::new();
    let mut put = |key: &str, value: Option<serde_json::Value>| {
        if let Some(value) = value {
            summary.insert(key.to_owned(), value);
        }
    };
    put(
        "usageMetadata",
        usage.and_then(|usage| serde_json::to_value(usage).ok()),
    );
    put(
        "finishReason",
        finish_reason.map(|reason| serde_json::Value::String(reason.as_str().to_owned())),
    );
    put(
        "finishMessage",
        finish_message.map(serde_json::Value::String),
    );
    put("modelVersion", model_version.map(serde_json::Value::String));
    put("responseId", response_id.map(serde_json::Value::String));
    serde_json::Value::Object(summary)
}

/// Whether `frame` carries nothing but a response id.
pub(crate) fn is_analysis_only(frame: &WireFrame) -> bool {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ResponseIdOnly {
        #[serde(rename = "responseId")]
        _id: String,
    }
    matches!(
        wire::classify_marker_keyed_frame::<ResponseIdOnly>(&frame.as_str(), &["responseId"]),
        WireEvent::Known(_)
    )
}

/// Project provider verdicts, usage, response identity and errors before
/// normalization. Parsing failures leave the decoder's verdict alone.
pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
    if let Ok(ObservedUsageOnly { usage: Some(usage) }) =
        serde_json::from_slice::<ObservedUsageOnly>(payload)
    {
        sink.emit(AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: usage.prompt_token_count,
                output_tokens: usage.candidates_token_count,
                total_tokens: usage.total_token_count,
                cached_input_tokens: usage.cached_content_token_count,
                reasoning_tokens: usage.thoughts_token_count,
                tool_input_tokens: usage.tool_use_prompt_token_count,
            },
        });
    }
    // Metadata is read on its own so a malformed candidate cannot erase an
    // otherwise valid usage report.
    let Ok(metadata) = serde_json::from_slice::<ObservedMetadata>(payload) else {
        return;
    };
    let candidate = metadata.candidates.into_iter().next().unwrap_or_default();
    let scrub = |value: String| sink.scrub(&value);
    let verdict = AdapterVerdict {
        finish_reason: candidate.finish_reason.map(scrub),
        block_reason: metadata
            .prompt_feedback
            .and_then(|feedback| feedback.block_reason)
            .map(scrub),
        detail: candidate.finish_message.map(scrub),
        model: metadata.model_version.map(scrub),
    };
    let response_id = metadata.response_id.map(scrub);
    sink.provider(verdict, response_id);
    if let Some(error) = metadata.error {
        error.emit(sink);
    }
}

#[derive(Deserialize)]
struct ObservedUsageOnly {
    #[serde(rename = "usageMetadata")]
    usage: Option<ObservedUsage>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ObservedUsage {
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    prompt_token_count: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    candidates_token_count: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    total_token_count: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    cached_content_token_count: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    thoughts_token_count: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    tool_use_prompt_token_count: Option<u64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ObservedMetadata {
    #[serde(default)]
    candidates: Vec<ObservedCandidate>,
    prompt_feedback: Option<ObservedFeedback>,
    model_version: Option<String>,
    response_id: Option<String>,
    error: Option<ObservedError>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ObservedCandidate {
    finish_reason: Option<String>,
    finish_message: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ObservedFeedback {
    block_reason: Option<String>,
}
