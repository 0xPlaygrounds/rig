//! How Bedrock Converse answers
//! [`GenerationOptions`](rig_core::completion::GenerationOptions): reasoning
//! goes in the model's own request fields (`additionalModelRequestFields`)
//! and depends on the model's family and Claude class; the rest are
//! Converse fields or checkpoints the base builder places.

use rig_core::completion::options::{Mapping, OptionFields, OptionMap};
use rig_core::completion::{CacheRetention, CompletionRequest, Effort, Reasoning, ServiceTier};
use rig_core::providers::anthropic::completion::{ClaudeClass, claude_class};
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

/// Claude's reasoning on Bedrock, by class.
fn claude_reasoning(
    class: Option<ClaudeClass>,
    reasoning: &Reasoning,
    max_tokens: Option<u64>,
) -> Mapping {
    use ClaudeClass::{A0, A1, A3, A6, A7, A8};
    match reasoning {
        Reasoning::Off => match class {
            Some(A7 | A8) => Mapping::unsupported("thinking cannot be disabled on this model"),
            Some(A6) => model_fields(json!({"thinking": {"type": "between_tools"}})),
            Some(A0 | A1) => Mapping::Omit("thinking is off unless the request asks for it"),
            // A model the table does not name (an application inference
            // profile, say) may think by default: it gets the newest
            // models' shape, which A0 and A1 take too.
            Some(_) | None => model_fields(json!({"thinking": {"type": "disabled"}})),
        },
        Reasoning::Effort(effort @ (Effort::Low | Effort::Medium | Effort::High)) => match class {
            Some(A0) | None => {
                Mapping::unsupported("this model takes a thinking budget, not an effort level")
            }
            Some(A1) => model_fields(json!({"output_config": {"effort": effort.as_str()}})),
            Some(_) => model_fields(json!({
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort.as_str()},
            })),
        },
        Reasoning::Effort(Effort::Minimal) => {
            Mapping::unsupported("no Bedrock model lists a `minimal` effort level")
        }
        Reasoning::Effort(effort) => Mapping::unsupported(format!(
            "unverified for aws_bedrock: which Claude models take `{}` on Bedrock",
            effort.as_str()
        )),
        Reasoning::Budget { tokens } => match class {
            Some(class) if class >= A3 => {
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
    let class = (family == Family::Claude)
        .then(|| claude_class(model))
        .flatten();
    let always_reasons = model.contains("deepseek.r1");
    let caches = family == Family::Claude || family == Family::Nova;
    let short_ttl_only = ["claude-3-7", "claude-3-5-sonnet-20241022-v2"]
        .iter()
        .any(|id| model.contains(id));
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match family {
            Family::Claude => claude_reasoning(class, reasoning, request.max_tokens),
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
        top_p: Mapping::of(top_p, |top_p| match class {
            Some(class) if class >= ClaudeClass::A3 => {
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
