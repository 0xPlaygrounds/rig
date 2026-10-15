//! How each Messages-format dialect answers [`GenerationOptions`]: the JSON
//! it merges into the request body, the default it relies on, or the reason
//! it refuses. Anthropic's own answers depend on the model's catalog entry:
//! the effort levels and budget it takes, whether thinking turns off and
//! how, and whether it fixes its sampling.
//!
//! [`GenerationOptions`]: crate::completion::GenerationOptions

use serde_json::json;

use crate::catalog::ModelSpec;
use crate::completion::options::{Mapping, OptionFields, OptionMap};
use crate::completion::{CacheRetention, CompletionRequest, Effort, Reasoning, ServiceTier};
use crate::message::ToolChoice;

use super::wire::{ANTHROPIC, MINIMAX, MOONSHOT, Messages, XIAOMIMIMO, ZAI};

/// The `thinking` and `output_config` Anthropic's Messages API takes for
/// `reasoning` on the model `spec` describes (`None`: a model the catalog
/// does not list, which gets the newest models' shapes), with `max_tokens`
/// the request's output cap.
fn claude_reasoning(
    spec: Option<&ModelSpec>,
    reasoning: &Reasoning,
    max_tokens: Option<u64>,
) -> Mapping {
    let support = spec.map(|spec| &spec.reasoning);
    // A model the catalog says does not reason is answered by the shared
    // rule: nothing to turn off, and nothing else to take.
    if let Some(support) = support.filter(|support| !support.supported()) {
        return match support.refusal(reasoning) {
            Some(reason) => Mapping::unsupported(reason),
            None => Mapping::Omit("the model does not reason"),
        };
    }
    match reasoning {
        Reasoning::Off => match spec {
            Some(spec) if spec.reasoning.can_disable() == Some(false) => {
                Mapping::unsupported("thinking cannot be disabled on this model")
            }
            _ => {
                let off = spec
                    .and_then(|spec| spec.compat.thinking_off.as_deref())
                    .unwrap_or("disabled");
                Mapping::Send(json!({"thinking": {"type": off}}))
            }
        },
        Reasoning::Effort(Effort::Minimal) => {
            Mapping::unsupported("Claude has no `minimal` effort level")
        }
        Reasoning::Effort(effort) => match spec {
            Some(spec) if spec.reasoning.levels().is_some_and(<[Effort]>::is_empty) => {
                Mapping::unsupported("this model takes a thinking budget, not an effort level")
            }
            Some(spec)
                if spec
                    .reasoning
                    .levels()
                    .is_some_and(|levels| !levels.contains(effort)) =>
            {
                Mapping::unsupported(format!(
                    "this model has no `{}` effort level",
                    effort.as_str()
                ))
            }
            Some(spec) if !spec.compat.adaptive_thinking => {
                Mapping::Send(json!({"output_config": {"effort": effort.as_str()}}))
            }
            _ => Mapping::Send(json!({
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort.as_str()},
            })),
        },
        Reasoning::Budget { tokens } => match support {
            // Known levels with no budget: the catalog says it takes none.
            Some(support) if support.levels().is_some() && support.budget().is_none() => {
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

/// `top_p` on the model `spec` describes: refused by its catalog entry's
/// sampling rule (the one [`ModelSpec::refusals`] applies) for a model
/// that fixes its sampling, and otherwise taken only without
/// `temperature`.
fn claude_top_p(
    spec: Option<&ModelSpec>,
    reasoning: Option<&Reasoning>,
    top_p: f64,
    temperature: bool,
) -> Mapping {
    if let Some(reason) =
        spec.and_then(|spec| spec.sampling_refusal("top_p", spec.reasons_with(reasoning)))
    {
        return Mapping::unsupported(reason);
    }
    match temperature {
        true => Mapping::unsupported("this model takes `temperature` or `top_p`, not both"),
        false => Mapping::Send(json!({"top_p": top_p})),
    }
}

/// `parallel_tool_calls` on the Messages API: `false` sets
/// `disable_parallel_tool_use` on the request's tool choice, the default
/// `auto` when it names none. `has_tools` counts the request's tools and the
/// raw `additional_params.tools` the base appends to them.
fn parallel_tool_calls(request: &CompletionRequest, has_tools: bool, parallel: bool) -> Mapping {
    if parallel {
        return Mapping::Omit("parallel tool use is the default");
    }
    if !has_tools {
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
            wire.provider.dialect.default_max_tokens(&wire.facts, model)
        }
    });
    // Not `options::param`, which calls `map_options`.
    let has_tools = crate::completion::history::declares_tools(request);
    match wire.provider.dialect.name {
        name if name == ANTHROPIC.name => {
            anthropic(wire, request, model, max_tokens, has_tools, fields)
        }
        name if name == MINIMAX.name => minimax(model, fields),
        name if name == XIAOMIMIMO.name => xiaomimimo(request, has_tools, fields),
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
    has_tools: bool,
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
        cache_key,
    } = fields;
    let spec = super::completion::spec(&wire.facts, model);
    let places = wire.prompt_caching || wire.static_prefix_cache_ttl.is_some();
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| {
            claude_reasoning(spec, reasoning, max_tokens)
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
            parallel_tool_calls(request, has_tools, parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| {
            claude_top_p(spec, reasoning, top_p, request.temperature.is_some())
        }),
        seed: Mapping::of(seed, |_| {
            Mapping::unsupported("Anthropic has no seed parameter")
        }),
        stop: Mapping::of_stop(stop, stop_sequences),
        cache_key: Mapping::unrouted(cache_key),
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
        cache_key,
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
        cache_key: Mapping::unrouted(cache_key),
    }
}

/// Xiaomi MiMo's Messages endpoint: thinking on or off, implicit caching.
fn xiaomimimo(request: &CompletionRequest, has_tools: bool, fields: OptionFields<'_>) -> OptionMap {
    let OptionFields {
        reasoning,
        cache,
        service_tier,
        verbosity,
        parallel_tool_calls: parallel,
        top_p,
        seed,
        stop,
        cache_key,
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
            parallel_tool_calls(request, has_tools, parallel)
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
        cache_key: Mapping::unrouted(cache_key),
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
        cache_key,
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
        cache_key: Mapping::unrouted(cache_key),
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
        cache_key,
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
        cache_key: Mapping::unrouted(cache_key),
    }
}

#[cfg(test)]
mod tests;
