//! Bedrock Converse's typed request options and reply extras: the Converse
//! body fields with no portable [`GenerationOptions`] form, and the reply
//! fields Rig does not normalize. Every route of Bedrock is Converse, so the
//! options have the shared section only.
//!
//! [`GenerationOptions`]: rig_core::completion::GenerationOptions
//!
//! ```
//! use rig_bedrock::extension::{Bedrock, BedrockOptions, Guardrail, GuardrailTrace};
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//!
//! let options = BedrockOptions::default()
//!     .guardrail(Guardrail::new("gr-1", "DRAFT").trace(GuardrailTrace::Enabled));
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<Bedrock>(&options)?);
//! # Ok::<(), rig_core::completion::OptionsError>(())
//! ```

use std::collections::BTreeMap;

use rig_core::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use rig_core::message::Api;
use serde::de::DeserializeOwned;
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

/// The `aws_bedrock` provider: its key, [`BedrockOptions`] and
/// [`BedrockExtras`].
#[derive(Debug)]
pub enum Bedrock {}

impl ProviderExtension for Bedrock {
    const PROVIDER: &'static str = crate::completion::PROVIDER_NAME;
    type Options = BedrockOptions;
    type Extras = BedrockExtras;
}

/// Bedrock's request options. Each field is sent on unary and streamed
/// Converse calls alike.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct BedrockOptions {
    /// The fields every Converse call reads.
    #[serde(rename = "*")]
    pub shared: BedrockShared,
}

impl ExtensionOptions for BedrockOptions {}

impl BedrockOptions {
    /// Apply `guardrail` to the request (`guardrailConfig`).
    pub fn guardrail(mut self, guardrail: Guardrail) -> Self {
        self.shared.guardrail = Some(guardrail);
        self
    }

    /// Ask for the latency profile `latency` (`performanceConfig.latency`).
    pub fn performance_latency(mut self, latency: PerformanceLatency) -> Self {
        self.shared.performance_latency = Some(latency);
        self
    }

    /// Tag the request with `key` = `value` (`requestMetadata`), for
    /// invocation logs and cost allocation. Bedrock takes at most 16 pairs.
    pub fn request_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.shared
            .request_metadata
            .insert(key.into(), value.into());
        self
    }

    /// Ask for the model's own reply field at the JSON Pointer `path`
    /// (`additionalModelResponseFieldPaths`). Bedrock takes at most 10.
    pub fn additional_response_field_path(mut self, path: impl Into<String>) -> Self {
        self.shared
            .additional_response_field_paths
            .push(path.into());
        self
    }

    /// Enable the Anthropic beta `beta` for a Claude model
    /// (`additionalModelRequestFields.anthropic_beta`).
    pub fn anthropic_beta(mut self, beta: impl Into<String>) -> Self {
        self.shared.anthropic_beta.push(beta.into());
        self
    }

    /// Sample from the `top_k` most likely tokens
    /// (`additionalModelRequestFields.top_k`), the spelling Claude models
    /// read. Other model families reject it.
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.shared.top_k = Some(top_k);
        self
    }
}

/// The fields every Converse call reads, serialized as the Converse body
/// fragment they write.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BedrockShared {
    /// `guardrailConfig`.
    pub guardrail: Option<Guardrail>,
    /// `performanceConfig.latency`.
    pub performance_latency: Option<PerformanceLatency>,
    /// `requestMetadata`, sent when not empty.
    pub request_metadata: BTreeMap<String, String>,
    /// `additionalModelResponseFieldPaths`, sent when not empty.
    pub additional_response_field_paths: Vec<String>,
    /// `additionalModelRequestFields.anthropic_beta`, sent when not empty.
    pub anthropic_beta: Vec<String>,
    /// `additionalModelRequestFields.top_k`.
    pub top_k: Option<u32>,
}

/// The fields Bedrock passes to the model as they are.
#[derive(Serialize)]
struct ModelFields<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    anthropic_beta: &'a [String],
}

#[derive(Serialize)]
struct Performance {
    latency: PerformanceLatency,
}

impl Serialize for BedrockShared {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        if let Some(guardrail) = &self.guardrail {
            map.serialize_entry("guardrailConfig", guardrail)?;
        }
        if let Some(latency) = self.performance_latency {
            map.serialize_entry("performanceConfig", &Performance { latency })?;
        }
        if !self.request_metadata.is_empty() {
            map.serialize_entry("requestMetadata", &self.request_metadata)?;
        }
        if !self.additional_response_field_paths.is_empty() {
            map.serialize_entry(
                "additionalModelResponseFieldPaths",
                &self.additional_response_field_paths,
            )?;
        }
        if self.top_k.is_some() || !self.anthropic_beta.is_empty() {
            map.serialize_entry(
                "additionalModelRequestFields",
                &ModelFields {
                    top_k: self.top_k,
                    anthropic_beta: &self.anthropic_beta,
                },
            )?;
        }
        map.end()
    }
}

/// A [Bedrock guardrail](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
/// to apply. An intervention ends the reply with
/// [`FinishReason::ContentFilter`](rig_core::completion::FinishReason::ContentFilter),
/// and [`BedrockExtras::trace`] holds the guardrail's trace when one was
/// asked for.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Guardrail {
    /// The guardrail's id or ARN.
    #[serde(rename = "guardrailIdentifier")]
    pub identifier: String,
    /// Its version, or `DRAFT`.
    #[serde(rename = "guardrailVersion")]
    pub version: String,
    /// Whether the reply carries the guardrail's trace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<GuardrailTrace>,
}

impl Guardrail {
    /// The guardrail `identifier` (an id or ARN) at `version` (a version or
    /// `DRAFT`), with no trace asked for.
    pub fn new(identifier: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            identifier: identifier.into(),
            version: version.into(),
            trace: None,
        }
    }

    /// Ask for the guardrail's trace.
    pub fn trace(mut self, trace: GuardrailTrace) -> Self {
        self.trace = Some(trace);
        self
    }
}

/// Whether a reply carries its guardrail's trace.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailTrace {
    /// `enabled`.
    Enabled,
    /// `disabled`.
    Disabled,
    /// `enabled_full`: the trace of every policy, not only the ones that
    /// intervened.
    EnabledFull,
}

/// A Converse latency profile.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceLatency {
    /// `standard`.
    Standard,
    /// `optimized`, on the models and regions that offer it.
    Optimized,
}

/// The fields of a unary Converse reply Rig does not normalize. A field
/// the reply does not carry is `None`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BedrockExtras {
    /// `stopReason`, as Bedrock spells it (`guardrail_intervened`, ...).
    pub stop_reason: Option<String>,
    /// `metrics.latencyMs`.
    pub latency_ms: Option<u64>,
    /// `usage.cacheDetails`: the cache writes by TTL.
    pub cache_details: Option<Vec<CacheDetail>>,
    /// `serviceTier.type`, the tier that served the request.
    pub service_tier: Option<String>,
    /// `performanceConfig.latency`, the latency profile that served it.
    pub performance_latency: Option<String>,
    /// `trace`: the guardrail and prompt-router traces, as Bedrock sent them.
    pub trace: Option<Value>,
    /// `trace.promptRouter.invokedModelId`: the model a prompt router chose.
    pub invoked_model_id: Option<String>,
    /// `additionalModelResponseFields`: the fields
    /// [`BedrockOptions::additional_response_field_path`] asked for.
    pub additional_model_response_fields: Option<Value>,
}

/// One cache write of a reply.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CacheDetail {
    /// `inputTokens` written.
    pub input_tokens: Option<u64>,
    /// `ttl` of the write (`5m`, `1h`).
    pub ttl: Option<String>,
}

/// The value at `pointer` in `raw`, or `None` when it is absent or `null`.
fn at<T: DeserializeOwned>(raw: &Value, pointer: &str) -> Result<Option<T>, serde_json::Error> {
    match raw.pointer(pointer) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => T::deserialize(value).map(Some),
    }
}

impl ReplyExtras for BedrockExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Ok(Self {
            stop_reason: at(raw, "/stopReason")?,
            latency_ms: at(raw, "/metrics/latencyMs")?,
            cache_details: at(raw, "/usage/cacheDetails")?,
            service_tier: at(raw, "/serviceTier/type")?,
            performance_latency: at(raw, "/performanceConfig/latency")?,
            trace: at(raw, "/trace")?,
            invoked_model_id: at(raw, "/trace/promptRouter/invokedModelId")?,
            additional_model_response_fields: at(raw, "/additionalModelResponseFields")?,
        })
    }
}

#[cfg(test)]
mod tests;
