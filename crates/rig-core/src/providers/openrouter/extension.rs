//! OpenRouter's typed request options and reply extras. Every option is
//! in the shared section, so one entry serves the Chat and the Responses
//! route alike.
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::openrouter::extension::{
//!     ModelFallbacks, OpenRouter, OpenRouterOptions, ProviderPreferences,
//! };
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let options = OpenRouterOptions::new()
//!     .provider(ProviderPreferences::new().only(["anthropic"]).zdr(true))
//!     .models(ModelFallbacks::new(["openai/gpt-4o-mini"])?);
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<OpenRouter>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// OpenRouter's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpenRouter;

impl ProviderExtension for OpenRouter {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = OpenRouterOptions;
    type Extras = OpenRouterExtras;
}

/// OpenRouter's request options, all sent on every route.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OpenRouterOptions {
    /// The fields every route takes.
    #[serde(rename = "*")]
    pub shared: OpenRouterShared,
}

/// The fields OpenRouter takes on every route.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OpenRouterShared {
    /// Which upstream providers may serve the request, and in what order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderPreferences>,
    /// The models tried after the request's own, in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<ModelFallbacks>,
    /// Plugins such as `{"id": "web"}`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub plugins: Vec<Value>,
    /// Groups requests into one session in OpenRouter's logs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// String pairs attached to the request.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
    /// Reasoning fields sent beside the mapped effort.
    #[serde(skip_serializing_if = "ReasoningExtra::is_empty")]
    pub reasoning: ReasoningExtra,
    /// Sample from the `k` most likely tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// The minimum probability of a token, relative to the most likely one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    /// Top-a sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_a: Option<f64>,
    /// Penalizes tokens already in the prompt and the output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    /// A stable id of the end user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// The `reasoning` fields OpenRouter takes beside the effort or budget the
/// generation options map.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ReasoningExtra {
    /// Reason, but leave the reasoning out of the reply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
    /// How much of the reasoning the reply summarizes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

impl ReasoningExtra {
    fn is_empty(&self) -> bool {
        self.exclude.is_none() && self.summary.is_none()
    }
}

/// How much of the reasoning a reply summarizes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummary {
    /// The provider decides.
    Auto,
    /// A short summary.
    Concise,
    /// A detailed summary.
    Detailed,
}

impl OpenRouterOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Route by `preferences`.
    pub fn provider(mut self, preferences: ProviderPreferences) -> Self {
        self.shared.provider = Some(preferences);
        self
    }

    /// Fall back to `models`, in order, after the request's own model.
    pub fn models(mut self, models: ModelFallbacks) -> Self {
        self.shared.models = Some(models);
        self
    }

    /// Add `plugin`, such as `{"id": "web"}`.
    pub fn plugin(mut self, plugin: Value) -> Self {
        self.shared.plugins.push(plugin);
        self
    }

    /// Group the request under session `id`.
    pub fn session_id(mut self, id: impl Into<String>) -> Self {
        self.shared.session_id = Some(id.into());
        self
    }

    /// Attach the metadata pair `key`, `value`.
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.shared.metadata.insert(key.into(), value.into());
        self
    }

    /// Reason but leave the reasoning out of the reply when `exclude`.
    pub fn reasoning_exclude(mut self, exclude: bool) -> Self {
        self.shared.reasoning.exclude = Some(exclude);
        self
    }

    /// Summarize the reasoning at `summary`.
    pub fn reasoning_summary(mut self, summary: ReasoningSummary) -> Self {
        self.shared.reasoning.summary = Some(summary);
        self
    }

    /// Sample from the `k` most likely tokens.
    pub fn top_k(mut self, k: u32) -> Self {
        self.shared.top_k = Some(k);
        self
    }

    /// Set the minimum relative token probability.
    pub fn min_p(mut self, p: f64) -> Self {
        self.shared.min_p = Some(p);
        self
    }

    /// Set top-a sampling.
    pub fn top_a(mut self, a: f64) -> Self {
        self.shared.top_a = Some(a);
        self
    }

    /// Set the repetition penalty.
    pub fn repetition_penalty(mut self, penalty: f64) -> Self {
        self.shared.repetition_penalty = Some(penalty);
        self
    }

    /// Name the end user.
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.shared.user = Some(user.into());
        self
    }
}

impl ExtensionOptions for OpenRouterOptions {}

/// A non-empty list of fallback models.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ModelFallbacks(Vec<String>);

/// A fallback list with no model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("model fallbacks need at least one model")]
pub struct EmptyFallbacks;

impl ModelFallbacks {
    /// The models tried, in order, after the request's own.
    ///
    /// # Errors
    ///
    /// When `models` is empty.
    pub fn new(
        models: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, EmptyFallbacks> {
        let models: Vec<String> = models.into_iter().map(Into::into).collect();
        if models.is_empty() {
            return Err(EmptyFallbacks);
        }
        Ok(Self(models))
    }

    /// The fallback models, in order.
    pub fn models(&self) -> &[String] {
        &self.0
    }
}

/// Whether providers that may store request data are eligible.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DataCollection {
    /// Any provider.
    Allow,
    /// Only providers that store no request data beyond the request.
    Deny,
}

/// A quantization an upstream provider serves a model at.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    /// 4-bit integers.
    Int4,
    /// 8-bit integers.
    Int8,
    /// 16-bit floats.
    Fp16,
    /// Brain floats.
    Bf16,
    /// 32-bit floats.
    Fp32,
    /// 8-bit floats.
    Fp8,
    /// Not stated by the provider.
    Unknown,
}

/// What providers are ordered by.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderSortStrategy {
    /// Cheapest first.
    Price,
    /// Highest throughput first.
    Throughput,
    /// Lowest latency first.
    Latency,
}

/// How a request with fallback models sorts its providers.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SortPartition {
    /// Within each model.
    Model,
    /// Across every model.
    None,
}

/// A sort with a partition.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProviderSortConfig {
    /// What providers are ordered by.
    pub by: ProviderSortStrategy,
    /// How fallback models are grouped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partition: Option<SortPartition>,
}

impl ProviderSortConfig {
    /// Sort by `by`.
    pub fn new(by: ProviderSortStrategy) -> Self {
        Self {
            by,
            partition: None,
        }
    }

    /// Group fallback models by `partition`.
    pub fn partition(mut self, partition: SortPartition) -> Self {
        self.partition = Some(partition);
        self
    }
}

/// A provider sort: a strategy, or a strategy with a partition.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ProviderSort {
    /// A strategy alone.
    Simple(ProviderSortStrategy),
    /// A strategy and a partition.
    Complex(ProviderSortConfig),
}

impl From<ProviderSortStrategy> for ProviderSort {
    fn from(strategy: ProviderSortStrategy) -> Self {
        Self::Simple(strategy)
    }
}

impl From<ProviderSortConfig> for ProviderSort {
    fn from(config: ProviderSortConfig) -> Self {
        Self::Complex(config)
    }
}

/// A throughput floor in tokens per second, alone or per percentile.
/// Providers below it are tried later, not excluded.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ThroughputThreshold {
    /// One floor.
    Simple(f64),
    /// A floor per percentile.
    Percentile(PercentileThresholds),
}

/// A latency ceiling in seconds, alone or per percentile. Providers above
/// it are tried later, not excluded.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum LatencyThreshold {
    /// One ceiling.
    Simple(f64),
    /// A ceiling per percentile.
    Percentile(PercentileThresholds),
}

/// A threshold per percentile.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PercentileThresholds {
    /// The median.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p50: Option<f64>,
    /// The 75th percentile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p75: Option<f64>,
    /// The 90th percentile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p90: Option<f64>,
    /// The 99th percentile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p99: Option<f64>,
}

impl PercentileThresholds {
    /// No threshold.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the median threshold.
    pub fn p50(mut self, value: f64) -> Self {
        self.p50 = Some(value);
        self
    }

    /// Set the 75th-percentile threshold.
    pub fn p75(mut self, value: f64) -> Self {
        self.p75 = Some(value);
        self
    }

    /// Set the 90th-percentile threshold.
    pub fn p90(mut self, value: f64) -> Self {
        self.p90 = Some(value);
        self
    }

    /// Set the 99th-percentile threshold.
    pub fn p99(mut self, value: f64) -> Self {
        self.p99 = Some(value);
        self
    }
}

/// Price ceilings, in USD per million tokens or per item. A request no
/// eligible provider can serve under them fails.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MaxPrice {
    /// Per million prompt tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<f64>,
    /// Per million completion tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion: Option<f64>,
    /// Per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<f64>,
    /// Per image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<f64>,
}

impl MaxPrice {
    /// No ceiling.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cap the prompt price.
    pub fn prompt(mut self, price: f64) -> Self {
        self.prompt = Some(price);
        self
    }

    /// Cap the completion price.
    pub fn completion(mut self, price: f64) -> Self {
        self.completion = Some(price);
        self
    }

    /// Cap the price per request.
    pub fn request(mut self, price: f64) -> Self {
        self.request = Some(price);
        self
    }

    /// Cap the price per image.
    pub fn image(mut self, price: f64) -> Self {
        self.image = Some(price);
        self
    }
}

/// Which upstream providers may serve a request and in what order
/// (<https://openrouter.ai/docs/guides/routing/provider-selection>). Unset
/// fields keep OpenRouter's defaults. Slugs and limits are sent as given.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ProviderPreferences {
    /// Provider slugs tried first, in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    /// The only eligible provider slugs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    /// Provider slugs never used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    /// Whether providers outside `order` may serve the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    /// Whether only providers taking every request parameter are eligible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    /// Whether providers that store request data are eligible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<DataCollection>,
    /// Whether only zero-data-retention endpoints are eligible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
    /// The order providers are tried in, in place of load balancing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<ProviderSort>,
    /// The throughput below which a provider is tried later.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_min_throughput: Option<ThroughputThreshold>,
    /// The latency above which a provider is tried later.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_max_latency: Option<LatencyThreshold>,
    /// The price ceilings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_price: Option<MaxPrice>,
    /// The quantizations a provider may serve the model at.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<Quantization>>,
}

fn strings(items: impl IntoIterator<Item = impl Into<String>>) -> Option<Vec<String>> {
    Some(items.into_iter().map(Into::into).collect())
}

impl ProviderPreferences {
    /// No preference.
    pub fn new() -> Self {
        Self::default()
    }

    /// Try `providers` first, in order.
    pub fn order(mut self, providers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.order = strings(providers);
        self
    }

    /// Make only `providers` eligible.
    pub fn only(mut self, providers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.only = strings(providers);
        self
    }

    /// Never use `providers`.
    pub fn ignore(mut self, providers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.ignore = strings(providers);
        self
    }

    /// Whether providers outside `order` may serve the request.
    pub fn allow_fallbacks(mut self, allow: bool) -> Self {
        self.allow_fallbacks = Some(allow);
        self
    }

    /// Whether only providers taking every request parameter are eligible.
    pub fn require_parameters(mut self, require: bool) -> Self {
        self.require_parameters = Some(require);
        self
    }

    /// Set the data collection policy.
    pub fn data_collection(mut self, policy: DataCollection) -> Self {
        self.data_collection = Some(policy);
        self
    }

    /// Whether only zero-data-retention endpoints are eligible.
    pub fn zdr(mut self, enable: bool) -> Self {
        self.zdr = Some(enable);
        self
    }

    /// Try providers in the order `sort` gives.
    pub fn sort(mut self, sort: impl Into<ProviderSort>) -> Self {
        self.sort = Some(sort.into());
        self
    }

    /// Try providers below `threshold` later.
    pub fn preferred_min_throughput(mut self, threshold: ThroughputThreshold) -> Self {
        self.preferred_min_throughput = Some(threshold);
        self
    }

    /// Try providers above `threshold` later.
    pub fn preferred_max_latency(mut self, threshold: LatencyThreshold) -> Self {
        self.preferred_max_latency = Some(threshold);
        self
    }

    /// Fail the request when no provider is under `price`.
    pub fn max_price(mut self, price: MaxPrice) -> Self {
        self.max_price = Some(price);
        self
    }

    /// Make only providers serving one of `quantizations` eligible.
    pub fn quantizations(mut self, quantizations: impl IntoIterator<Item = Quantization>) -> Self {
        self.quantizations = Some(quantizations.into_iter().collect());
        self
    }

    /// Try the cheapest provider first.
    pub fn cheapest(self) -> Self {
        self.sort(ProviderSortStrategy::Price)
    }
}

/// OpenRouter's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpenRouterExtras {
    /// The upstream provider that served the request. Chat; Responses when
    /// sent.
    pub provider: Option<String>,
    /// The upstream provider's own finish reason. Chat only.
    pub native_finish_reason: Option<String>,
    /// The service tier that served the request. Both routes.
    pub service_tier: Option<String>,
    /// The upstream backend's fingerprint. Chat only.
    pub system_fingerprint: Option<String>,
    /// The request's cost in credits. Both routes.
    pub cost: Option<f64>,
    /// The cost broken down by upstream charge. Both routes, keyed as each
    /// route spells them.
    pub cost_details: Option<Map<String, Value>>,
    /// Whether the request ran on the caller's own provider key. Both
    /// routes.
    pub is_byok: Option<bool>,
    /// Prompt token details: `usage.prompt_tokens_details` on Chat,
    /// `usage.input_tokens_details` on Responses.
    pub prompt_tokens_details: Option<Value>,
    /// Server tool usage, such as web searches. Chat only.
    pub server_tool_use_details: Option<Value>,
    /// OpenRouter's routing metadata. Chat only.
    pub openrouter_metadata: Option<Value>,
    /// The message's annotations, such as URL citations. Chat only.
    pub annotations: Option<Vec<Value>>,
}

impl ReplyExtras for OpenRouterExtras {
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        if api.as_str() == "openai.responses" {
            return Ok(Self {
                provider: reply_field(raw, "/provider")?,
                service_tier: reply_field(raw, "/service_tier")?,
                cost: reply_field(raw, "/usage/cost")?,
                cost_details: reply_field(raw, "/usage/cost_details")?,
                is_byok: reply_field(raw, "/usage/is_byok")?,
                prompt_tokens_details: reply_field(raw, "/usage/input_tokens_details")?,
                ..Self::default()
            });
        }
        Ok(Self {
            provider: reply_field(raw, "/provider")?,
            native_finish_reason: reply_field(raw, "/choices/0/native_finish_reason")?,
            service_tier: reply_field(raw, "/service_tier")?,
            system_fingerprint: reply_field(raw, "/system_fingerprint")?,
            cost: reply_field(raw, "/usage/cost")?,
            cost_details: reply_field(raw, "/usage/cost_details")?,
            is_byok: reply_field(raw, "/usage/is_byok")?,
            prompt_tokens_details: reply_field(raw, "/usage/prompt_tokens_details")?,
            server_tool_use_details: reply_field(raw, "/usage/server_tool_use_details")?,
            openrouter_metadata: reply_field(raw, "/openrouter_metadata")?,
            annotations: reply_field(raw, "/choices/0/message/annotations")?,
        })
    }
}

#[cfg(test)]
mod tests;
