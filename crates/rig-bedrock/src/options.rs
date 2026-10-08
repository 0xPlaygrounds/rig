//! How Bedrock Converse answers
//! [`GenerationOptions`](rig_core::completion::GenerationOptions): reasoning
//! goes in the model's own request fields (`additionalModelRequestFields`)
//! and depends on the model's family and catalog entry; the rest are
//! Converse fields or checkpoints the base builder places.

use rig_core::catalog::{ModelSpec, Sampling};
use rig_core::completion::options::{Mapping, OptionFields, OptionMap};
use rig_core::completion::{CacheRetention, CompletionRequest, Effort, Reasoning, ServiceTier};
use serde_json::json;

use crate::completion::Family;

/// `value` under `additionalModelRequestFields`.
fn model_fields(value: serde_json::Value) -> Mapping {
    Mapping::Send(json!({ "additionalModelRequestFields": value }))
}

/// Whether `model` is Amazon Nova 2, which takes a reasoning effort.
fn nova_2(model: &str) -> bool {
    model.contains("nova-2")
}

/// Claude's reasoning on Bedrock, for the model `spec` describes (`None`: a
/// model the catalog does not list, such as an application inference
/// profile).
fn claude_reasoning(
    spec: Option<&ModelSpec>,
    reasoning: &Reasoning,
    max_tokens: Option<u64>,
) -> Mapping {
    // A model the catalog says does not reason is answered by the shared
    // rule: nothing to turn off, and nothing else to take.
    if let Some(spec) = spec.filter(|spec| !spec.reasoning.supported()) {
        return match spec.reasoning.refusal(reasoning) {
            Some(reason) => Mapping::unsupported(reason),
            None => Mapping::Omit("the model does not reason"),
        };
    }
    match reasoning {
        Reasoning::Off => match spec {
            Some(spec) if spec.reasoning.can_disable() == Some(false) => {
                Mapping::unsupported("thinking cannot be disabled on this model")
            }
            Some(spec) if !spec.compat.adaptive_thinking => {
                Mapping::Omit("thinking is off unless the request asks for it")
            }
            // A model the catalog does not list may think by default: it
            // gets the newest models' shape.
            _ => {
                let off = spec
                    .and_then(|spec| spec.compat.thinking_off.as_deref())
                    .unwrap_or("disabled");
                model_fields(json!({"thinking": {"type": off}}))
            }
        },
        Reasoning::Effort(effort @ (Effort::Low | Effort::Medium | Effort::High)) => match spec {
            Some(spec) if takes_effort(spec) && !spec.compat.adaptive_thinking => {
                model_fields(json!({"output_config": {"effort": effort.as_str()}}))
            }
            Some(spec) if takes_effort(spec) => model_fields(json!({
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort.as_str()},
            })),
            _ => Mapping::unsupported("this model takes a thinking budget, not an effort level"),
        },
        Reasoning::Effort(Effort::Minimal) => {
            Mapping::unsupported("no Bedrock model lists a `minimal` effort level")
        }
        Reasoning::Effort(effort) => Mapping::unsupported(format!(
            "unverified for aws_bedrock: which Claude models take `{}` on Bedrock",
            effort.as_str()
        )),
        Reasoning::Budget { tokens } => match spec {
            // Known levels with no budget: the catalog says it takes none.
            Some(spec)
                if spec.reasoning.levels().is_some() && spec.reasoning.budget().is_none() =>
            {
                Mapping::unsupported("this model takes an effort level, not a thinking budget")
            }
            _ if *tokens < 1024 => {
                Mapping::unsupported("a thinking budget must be at least 1024 tokens")
            }
            _ if max_tokens.is_some_and(|cap| u64::from(*tokens) >= cap) => {
                Mapping::unsupported("a thinking budget must be below `max_tokens`")
            }
            _ => model_fields(json!({
                "thinking": {"type": "enabled", "budget_tokens": tokens},
            })),
        },
        _ => Mapping::unsupported(NO_FORM),
    }
}

/// Whether the model `spec` describes lists effort levels. A model whose
/// levels the catalog does not know is answered as one it does not list.
fn takes_effort(spec: &ModelSpec) -> bool {
    spec.reasoning
        .levels()
        .is_some_and(|levels| !levels.is_empty())
}

/// The answer for a setting a later rig adds that this crate has no form
/// for: refused, never dropped.
const NO_FORM: &str = "this setting has no Converse form";

/// Nova 2's reasoning. `high` takes no sampling or output cap beside it.
fn nova_reasoning(request: &CompletionRequest, reasoning: &Reasoning) -> Mapping {
    match reasoning {
        Reasoning::Off => model_fields(json!({"reasoningConfig": {"type": "disabled"}})),
        Reasoning::Effort(Effort::High)
            if request.temperature.is_some() || request.max_tokens.is_some() =>
        {
            Mapping::unsupported("Nova 2 at `high` effort takes no temperature or max_tokens")
        }
        Reasoning::Effort(effort @ (Effort::Low | Effort::Medium | Effort::High)) => {
            model_fields(json!({"reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": effort.as_str(),
            }}))
        }
        Reasoning::Effort(effort) => {
            Mapping::unsupported(format!("Nova 2 has no `{}` effort level", effort.as_str()))
        }
        Reasoning::Budget { .. } => {
            Mapping::unsupported("Nova 2 takes an effort level, not a budget")
        }
        _ => Mapping::unsupported(NO_FORM),
    }
}

/// How Converse answers `fields` for `request` to `model`, of `family`.
pub(crate) fn converse(
    wire: &crate::completion::Converse,
    family: Family,
    model: &str,
    request: &CompletionRequest,
    fields: OptionFields<'_>,
) -> OptionMap {
    let OptionFields {
        reasoning,
        cache,
        service_tier,
        verbosity,
        parallel_tool_calls,
        top_p,
        seed,
        stop,
    } = fields;
    let spec = (family == Family::Claude)
        .then(|| wire.spec(model))
        .flatten();
    let always_reasons = model.contains("deepseek.r1");
    let caches = family == Family::Claude || family == Family::Nova;
    let short_ttl_only = ["claude-3-7", "claude-3-5-sonnet-20241022-v2"]
        .iter()
        .any(|id| model.contains(id));
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match family {
            Family::Claude => claude_reasoning(spec, reasoning, request.max_tokens),
            Family::Nova if nova_2(model) => nova_reasoning(request, reasoning),
            _ if always_reasons => {
                Mapping::unsupported("this model always reasons and takes no setting")
            }
            _ => match reasoning {
                Reasoning::Off => Mapping::Omit("the model does not reason"),
                _ => Mapping::unsupported("the model does not reason"),
            },
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::None => Mapping::Omit("no cache checkpoint is placed"),
            _ if !caches => Mapping::unsupported("this model has no explicit prompt caching"),
            CacheRetention::Long if short_ttl_only => {
                Mapping::unsupported("this model takes five minute checkpoints only")
            }
            CacheRetention::Short | CacheRetention::Long => Mapping::Place,
            _ => Mapping::unsupported(NO_FORM),
        }),
        service_tier: Mapping::of(service_tier, |tier| match tier {
            ServiceTier::Auto => Mapping::Omit("Bedrock picks the tier unless one is named"),
            ServiceTier::Default => Mapping::Send(json!({"serviceTier": {"type": "default"}})),
            ServiceTier::Flex => Mapping::Send(json!({"serviceTier": {"type": "flex"}})),
            ServiceTier::Priority => Mapping::Send(json!({"serviceTier": {"type": "priority"}})),
            _ => Mapping::unsupported(NO_FORM),
        }),
        verbosity: Mapping::of(verbosity, |_| {
            Mapping::unsupported("Converse has no verbosity setting")
        }),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| {
            Mapping::unsupported("Converse's `toolConfig` has no parallel tool call switch")
        }),
        top_p: Mapping::of(top_p, |top_p| match spec.and_then(|spec| spec.sampling) {
            Some(Sampling::Never) => {
                Mapping::unsupported("this Claude model does not take `top_p`")
            }
            _ => Mapping::Send(json!({"inferenceConfig": {"topP": top_p}})),
        }),
        seed: Mapping::of(seed, |_| {
            Mapping::unsupported("Converse's `inferenceConfig` has no seed")
        }),
        stop: Mapping::of_stop(stop, |stop| {
            Mapping::Send(json!({"inferenceConfig": {"stopSequences": stop}}))
        }),
    }
}

#[cfg(test)]
mod tests;
