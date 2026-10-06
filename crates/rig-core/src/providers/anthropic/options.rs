//! How each Messages-format dialect answers [`GenerationOptions`]: the JSON
//! it merges into the request body, the default it relies on, or the reason
//! it refuses. Anthropic's own answers depend on the model's class, which
//! [`claude_class`] reads from the model id however the serving API spells
//! it.
//!
//! [`GenerationOptions`]: crate::completion::GenerationOptions

use serde_json::json;

use crate::completion::options::{Mapping, OptionFields, OptionMap};
use crate::completion::{CacheRetention, CompletionRequest, Effort, Reasoning, ServiceTier};
use crate::message::ToolChoice;

use super::wire::{ANTHROPIC, MINIMAX, MOONSHOT, Messages, XIAOMIMIMO, ZAI};

/// The thinking and sampling behaviour of a Claude model, by the class the
/// option mapping table names. Ordered oldest first.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ClaudeClass {
    /// Claude Haiku 4.5 and Sonnet 4.5: a thinking budget, no effort.
    A0,
    /// Claude Opus 4.5: a budget, and `low` to `high` effort.
    A1,
    /// Claude Opus 4.6 and Sonnet 4.6: adaptive thinking, `low` to `max`
    /// effort but `xhigh`, a deprecated budget.
    A2,
    /// Claude Opus 4.7 and 4.8: adaptive thinking, every effort level, no
    /// budget, a fixed `top_p`.
    A3,
    /// Claude Sonnet 5.
    A4,
    /// Claude Opus 5.
    A5,
    /// Claude Sonnet 5.5: off is `between_tools`.
    A6,
    /// Claude Opus 5.5: thinking cannot be turned off.
    A7,
    /// Claude Fable 5 and 5.1: thinking is always on.
    A8,
}

/// Every model id the class table names.
const CLASSES: [(&str, ClaudeClass); 13] = [
    ("claude-haiku-4-5", ClaudeClass::A0),
    ("claude-sonnet-4-5", ClaudeClass::A0),
    ("claude-opus-4-5", ClaudeClass::A1),
    ("claude-opus-4-6", ClaudeClass::A2),
    ("claude-sonnet-4-6", ClaudeClass::A2),
    ("claude-opus-4-7", ClaudeClass::A3),
    ("claude-opus-4-8", ClaudeClass::A3),
    ("claude-sonnet-5-5", ClaudeClass::A6),
    ("claude-sonnet-5", ClaudeClass::A4),
    ("claude-opus-5-5", ClaudeClass::A7),
    ("claude-opus-5", ClaudeClass::A5),
    ("claude-fable-5-1", ClaudeClass::A8),
    ("claude-fable-5", ClaudeClass::A8),
];

/// The class of the Claude `model`, spelled as Anthropic
/// (`claude-opus-5-5`), OpenRouter (`anthropic/claude-opus-5.5`) or Bedrock
/// (`us.anthropic.claude-opus-5-5-v1:0`) spell it, or one of its dated
/// snapshots. `None` for a model the table does not name.
#[doc(hidden)]
pub fn claude_class(model: &str) -> Option<ClaudeClass> {
    let model = model
        .rsplit_once("anthropic.")
        .map_or(model, |(_, rest)| rest);
    let model = model.strip_prefix("anthropic/").unwrap_or(model);
    let model = model.split_once("-v1:").map_or(model, |(id, _)| id);
    let model = model.replace('.', "-");
    CLASSES.iter().find_map(|(id, class)| {
        let rest = model.strip_prefix(id)?;
        (rest.is_empty() || rest.starts_with("-20")).then_some(*class)
    })
}

/// The `thinking` and `output_config` Anthropic's Messages API takes for
/// `reasoning` on a model of `class` (`None`: a model the table does not
/// name, which gets the newest models' shapes), with `max_tokens` the
/// request's output cap.
fn claude_reasoning(
    class: Option<ClaudeClass>,
    reasoning: &Reasoning,
    max_tokens: Option<u64>,
) -> Mapping {
    use ClaudeClass::{A0, A1, A2, A3, A6, A7, A8};
    let adaptive = |effort: &Effort| {
        Mapping::Send(json!({
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort.as_str()},
        }))
    };
    match reasoning {
        Reasoning::Off => match class {
            Some(A7 | A8) => Mapping::unsupported("thinking cannot be disabled on this model"),
            Some(A6) => Mapping::Send(json!({"thinking": {"type": "between_tools"}})),
            _ => Mapping::Send(json!({"thinking": {"type": "disabled"}})),
        },
        Reasoning::Effort(Effort::Minimal) => {
            Mapping::unsupported("Claude has no `minimal` effort level")
        }
        Reasoning::Effort(effort) => match (class, effort) {
            (Some(A0), _) => {
                Mapping::unsupported("this model takes a thinking budget, not an effort level")
            }
            (Some(A1), Effort::Low | Effort::Medium | Effort::High) => {
                Mapping::Send(json!({"output_config": {"effort": effort.as_str()}}))
            }
            (Some(A1 | A2), Effort::XHigh) | (Some(A1), Effort::Max) => Mapping::unsupported(
                format!("this model has no `{}` effort level", effort.as_str()),
            ),
            (_, effort) => adaptive(effort),
        },
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
            _ => Mapping::Send(json!({
                "thinking": {"type": "enabled", "budget_tokens": tokens},
            })),
        },
    }
}

/// `top_p` on a model of `class`, which some classes fix and the rest take
/// only without `temperature`.
fn claude_top_p(class: Option<ClaudeClass>, top_p: f64, temperature: bool) -> Mapping {
    match class {
        Some(class) if class >= ClaudeClass::A3 => {
            Mapping::unsupported("this model does not take `top_p`")
        }
        _ if temperature => {
            Mapping::unsupported("this model takes `temperature` or `top_p`, not both")
        }
        _ => Mapping::Send(json!({"top_p": top_p})),
    }
}

/// `parallel_tool_calls` on the Messages API: `false` sets
/// `disable_parallel_tool_use` on the request's tool choice, the default
/// `auto` when it names none.
fn parallel_tool_calls(request: &CompletionRequest, parallel: bool) -> Mapping {
    if parallel {
        return Mapping::Omit("parallel tool use is the default");
    }
    if request.tools.is_empty() {
        return Mapping::Omit("the request declares no tools to call in parallel");
    }
    match &request.tool_choice {
        None | Some(ToolChoice::Auto) => Mapping::Send(json!({
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
        })),
        Some(ToolChoice::Required | ToolChoice::Specific { .. }) => Mapping::Send(json!({
            "tool_choice": {"disable_parallel_tool_use": true},
        })),
        Some(ToolChoice::None) => Mapping::Omit("the request calls no tools"),
    }
}

/// The `stop_sequences` a dialect that takes them sends.
fn stop_sequences(stop: &[String]) -> Mapping {
    Mapping::Send(json!({"stop_sequences": stop}))
}

/// How `wire` answers `fields` for `request`.
pub(super) fn map_options(
    wire: &Messages,
    request: &CompletionRequest,
    fields: OptionFields<'_>,
) -> OptionMap {
    let model = request.model.as_deref().unwrap_or(&wire.model);
    let max_tokens = request.max_tokens.or_else(|| {
        if model == wire.model {
            wire.default_max_tokens
        } else {
            wire.provider.dialect.default_max_tokens(model)
        }
    });
    match wire.provider.dialect.name {
        name if name == ANTHROPIC.name => anthropic(wire, request, model, max_tokens, fields),
        name if name == MINIMAX.name => minimax(model, fields),
        name if name == XIAOMIMIMO.name => xiaomimimo(request, fields),
        name if name == ZAI.name || name == MOONSHOT.name => implicit_cache_gateway(fields),
        _ => unknown(fields),
    }
}

/// Anthropic's own API (section 6.1 of `TYPED_OPTIONS.md`).
fn anthropic(
    wire: &Messages,
    request: &CompletionRequest,
    model: &str,
    max_tokens: Option<u64>,
    fields: OptionFields<'_>,
) -> OptionMap {
    let OptionFields {
        reasoning,
        cache,
        service_tier,
        verbosity,
        parallel_tool_calls: parallel,
        top_p,
        seed,
        stop,
    } = fields;
    let class = claude_class(model);
    let places = wire.prompt_caching || wire.static_prefix_cache_ttl.is_some();
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| {
            claude_reasoning(class, reasoning, max_tokens)
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::None if places => Mapping::unsupported(
                "the wire places cache markers (`with_prompt_caching` or \
                 `with_static_prefix_cache_ttl`)",
            ),
            CacheRetention::None => Mapping::Omit("no cache marker is sent"),
            CacheRetention::Short => Mapping::Send(json!({"cache_control": {"type": "ephemeral"}})),
            CacheRetention::Long => {
                Mapping::Send(json!({"cache_control": {"type": "ephemeral", "ttl": "1h"}}))
            }
        }),
        service_tier: Mapping::of(service_tier, |tier| match tier {
            ServiceTier::Auto => Mapping::Send(json!({"service_tier": "auto"})),
            ServiceTier::Default => Mapping::Send(json!({"service_tier": "standard_only"})),
            ServiceTier::Priority => Mapping::unsupported(
                "Anthropic has no priority-only tier; `auto` uses Priority capacity under a \
                 commitment",
            ),
            ServiceTier::Flex => Mapping::unsupported("Anthropic has no flex tier"),
        }),
        verbosity: Mapping::of(verbosity, |_| {
            Mapping::unsupported("Anthropic has no verbosity setting")
        }),
        parallel_tool_calls: Mapping::of(parallel, |parallel| {
            parallel_tool_calls(request, parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| {
            claude_top_p(class, top_p, request.temperature.is_some())
        }),
        seed: Mapping::of(seed, |_| {
            Mapping::unsupported("Anthropic has no seed parameter")
        }),
        stop: Mapping::of_stop(stop, stop_sequences),
    }
}

/// MiniMax's Messages endpoint: block-level cache markers, and reasoning
/// only on the M3 models.
fn minimax(model: &str, fields: OptionFields<'_>) -> OptionMap {
    let OptionFields {
        reasoning,
        cache,
        service_tier,
        verbosity,
        parallel_tool_calls: parallel,
        top_p,
        seed,
        stop,
    } = fields;
    let model = model.to_ascii_lowercase();
    let flash = model.starts_with("minimax-m3.1");
    let m3 = !flash && model.starts_with("minimax-m3");
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off if m3 => Mapping::Send(json!({"thinking": {"type": "disabled"}})),
            Reasoning::Off => Mapping::unsupported("thinking cannot be disabled on this model"),
            Reasoning::Effort(Effort::Minimal) => {
                Mapping::unsupported("MiniMax has no `minimal` effort level")
            }
            Reasoning::Effort(effort) if flash => {
                Mapping::Send(json!({"output_config": {"effort": effort.as_str()}}))
            }
            Reasoning::Effort(_) => Mapping::unsupported("this model takes no effort level"),
            Reasoning::Budget { .. } => Mapping::unsupported("unverified for minimax"),
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::None => {
                Mapping::Omit("passive caching cannot be turned off; no marker is sent")
            }
            CacheRetention::Short => Mapping::Place,
            CacheRetention::Long => Mapping::unsupported("MiniMax caches for five minutes only"),
        }),
        service_tier: Mapping::of(service_tier, |tier| match tier {
            ServiceTier::Default => Mapping::Send(json!({"service_tier": "standard"})),
            ServiceTier::Priority => Mapping::Send(json!({"service_tier": "priority"})),
            ServiceTier::Auto | ServiceTier::Flex => {
                Mapping::unsupported("MiniMax takes only the standard and priority tiers")
            }
        }),
        verbosity: Mapping::of(verbosity, |_| {
            Mapping::unsupported("MiniMax has no verbosity setting")
        }),
        parallel_tool_calls: Mapping::of(parallel, |parallel| match parallel {
            true => Mapping::Omit("parallel tool use is the default"),
            false => Mapping::unsupported("unverified for minimax"),
        }),
        top_p: Mapping::of(top_p, |top_p| Mapping::Send(json!({"top_p": top_p}))),
        seed: Mapping::of(seed, |_| {
            Mapping::unsupported("MiniMax has no seed parameter")
        }),
        stop: Mapping::of_stop(stop, |_| {
            Mapping::unsupported("MiniMax documents `stop_sequences` as ignored")
        }),
    }
}

/// Xiaomi MiMo's Messages endpoint: thinking on or off, implicit caching.
fn xiaomimimo(request: &CompletionRequest, fields: OptionFields<'_>) -> OptionMap {
    let OptionFields {
        reasoning,
        cache,
        service_tier,
        verbosity,
        parallel_tool_calls: parallel,
        top_p,
        seed,
        stop,
    } = fields;
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off => Mapping::Send(json!({"thinking": {"type": "disabled"}})),
            _ => Mapping::unsupported("MiMo takes only thinking enabled or disabled"),
        }),
        cache: Mapping::of(cache, implicit_cache),
        service_tier: Mapping::of(service_tier, |_| {
            Mapping::unsupported("unverified for xiaomimimo")
        }),
        verbosity: Mapping::of(verbosity, |_| {
            Mapping::unsupported("unverified for xiaomimimo")
        }),
        parallel_tool_calls: Mapping::of(parallel, |parallel| {
            parallel_tool_calls(request, parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| {
            if (0.01..=1.0).contains(&top_p) {
                Mapping::Send(json!({"top_p": top_p}))
            } else {
                Mapping::unsupported("MiMo takes `top_p` from 0.01 to 1.0")
            }
        }),
        seed: Mapping::of(seed, |_| Mapping::unsupported("unverified for xiaomimimo")),
        stop: Mapping::of_stop(stop, stop_sequences),
    }
}

/// A cache that is implicit and cannot be turned off or lengthened.
fn implicit_cache(cache: &CacheRetention) -> Mapping {
    match cache {
        CacheRetention::None => {
            Mapping::Omit("caching is implicit and cannot be turned off; no marker is sent")
        }
        CacheRetention::Short => Mapping::Omit("caching is implicit"),
        CacheRetention::Long => Mapping::unsupported("the provider has no retention control"),
    }
}

/// Z.AI and Moonshot's Messages endpoints: implicit caching, and no other
/// option a vendor page or recording confirms.
fn implicit_cache_gateway(fields: OptionFields<'_>) -> OptionMap {
    let OptionFields {
        reasoning,
        cache,
        service_tier,
        verbosity,
        parallel_tool_calls: parallel,
        top_p,
        seed,
        stop,
    } = fields;
    const UNVERIFIED: &str = "unverified for this Messages-format provider";
    OptionMap {
        reasoning: Mapping::of(reasoning, |_| Mapping::unsupported(UNVERIFIED)),
        cache: Mapping::of(cache, implicit_cache),
        service_tier: Mapping::of(service_tier, |_| Mapping::unsupported(UNVERIFIED)),
        verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(UNVERIFIED)),
        parallel_tool_calls: Mapping::of(parallel, |parallel| match parallel {
            true => Mapping::Omit("parallel tool use is the default"),
            false => Mapping::unsupported(UNVERIFIED),
        }),
        top_p: Mapping::of(top_p, |_| Mapping::unsupported(UNVERIFIED)),
        seed: Mapping::of(seed, |_| Mapping::unsupported(UNVERIFIED)),
        stop: Mapping::of_stop(stop, |_| Mapping::unsupported(UNVERIFIED)),
    }
}

/// A Messages-format dialect this build does not know: no option is known
/// to reach it.
fn unknown(fields: OptionFields<'_>) -> OptionMap {
    let OptionFields {
        reasoning,
        cache,
        service_tier,
        verbosity,
        parallel_tool_calls: parallel,
        top_p,
        seed,
        stop,
    } = fields;
    let reason = "no option mapping is known for this Messages-format provider";
    let refuse = |set: bool| match set {
        true => Mapping::unsupported(reason),
        false => Mapping::Nothing,
    };
    OptionMap {
        reasoning: refuse(reasoning.is_some()),
        cache: refuse(cache.is_some()),
        service_tier: refuse(service_tier.is_some()),
        verbosity: refuse(verbosity.is_some()),
        parallel_tool_calls: refuse(parallel.is_some()),
        top_p: refuse(top_p.is_some()),
        seed: refuse(seed.is_some()),
        stop: refuse(!stop.is_empty()),
    }
}

#[cfg(test)]
mod tests;
