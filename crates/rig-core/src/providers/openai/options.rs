//! How the OpenAI-shaped wires answer
//! [`GenerationOptions`](crate::completion::GenerationOptions): Chat
//! Completions per dialect, and the OpenAI and xAI model facts Chat and
//! Responses share, which the catalog holds: whether a model reasons, the
//! effort levels it takes, its sampling rule and its prompt cache.

use serde_json::json;

use crate::catalog::{ModelSpec, Sampling};
use crate::completion::options::{FinalBody, Mapping, OptionFields, OptionMap};
use crate::completion::{CacheRetention, CompletionRequest, Effort, Reasoning, ServiceTier};
use crate::error::EncodeError;

use super::wire::Chat;

/// The catalog entry of OpenAI's `model`, read past a `vendor/` prefix and
/// a dated snapshot suffix. `None` for a model the catalog does not list (an
/// Azure deployment name, a fine-tune).
pub(crate) fn openai_spec(model: &str) -> Option<&'static ModelSpec> {
    let name = model.rsplit('/').next().unwrap_or(model);
    crate::catalog::lookup(super::wire::OPENAI.name, name)
}

/// Whether OpenAI's `model` reasons, or `None` for a model the catalog does
/// not list.
pub(crate) fn reasons(model: &str) -> Option<bool> {
    openai_spec(model).map(|spec| spec.reasoning.supported)
}

/// Whether `model`'s prompt cache takes `prompt_cache_options` (GPT-5.6 and
/// later, which keep one 30 minute retention).
pub(crate) fn caches_by_options(model: &str) -> bool {
    openai_spec(model).is_some_and(|spec| spec.compat.prompt_cache_options)
}

/// OpenAI's prompt cache retention for `model`, from the retentions its
/// catalog entry honours. Where it takes `prompt_cache_retention`, `Short`
/// is `in_memory` and `Long` is `24h`; a model the catalog does not list
/// takes `Short` only. Where it takes `prompt_cache_options`, one 30 minute
/// retention is all there is: `Short` sends nothing, `Long` is `long`
/// (Responses sends its 30 minute `ttl`; Chat refuses), and `None` stops
/// caching through explicit mode with no breakpoints.
pub(crate) fn openai_cache(model: &str, cache: &CacheRetention, long: Mapping) -> Mapping {
    let spec = openai_spec(model);
    let options = spec.is_some_and(|spec| spec.compat.prompt_cache_options);
    let honours = |retention: CacheRetention| match spec.map(|spec| &spec.caching.retention) {
        Some(listed) if !listed.is_empty() => listed.contains(&retention),
        _ => retention == CacheRetention::Short,
    };
    match cache {
        CacheRetention::None if options => {
            Mapping::Send(json!({"prompt_cache_options": {"mode": "explicit"}}))
        }
        CacheRetention::None => Mapping::unsupported("prompt caching is always on before GPT-5.6"),
        CacheRetention::Short if options => {
            Mapping::Omit("30 minutes is the only retention from GPT-5.6 on")
        }
        CacheRetention::Short if !honours(CacheRetention::Short) => {
            Mapping::unsupported("this model keeps its prompt cache for 24 hours only")
        }
        CacheRetention::Short => Mapping::Send(json!({"prompt_cache_retention": "in_memory"})),
        CacheRetention::Long if options => long,
        CacheRetention::Long if honours(CacheRetention::Long) => {
            Mapping::Send(json!({"prompt_cache_retention": "24h"}))
        }
        CacheRetention::Long => Mapping::unsupported("this model has no extended cache retention"),
    }
}

/// The OpenAI endpoint a request body is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Endpoint {
    ChatCompletions,
    Responses,
}

/// Refuse a body the model's catalog entry says the API rejects, by its
/// exact id (an OpenRouter `openai/` id is the gateway's to check). While
/// the model reasons, a model whose sampling rule is
/// [`Sampling::ReasoningOff`] takes no `temperature`, `top_p` or
/// `top_logprobs` (nor, on Chat Completions, `logprobs`), and one marked
/// `chat_tools_need_reasoning_off` whose reasoning cannot be turned off takes
/// no tools on Chat Completions at all. (One that can turn it off takes them
/// at effort `none`; the API's own error names that fix, and a recorded
/// session pins it.) The model reasons unless the body sets effort `none`;
/// with no effort it reasons when the catalog names its default level, and
/// is not checked otherwise. These are fields `additional_params` can set, so the check
/// reads the final body.
pub(crate) fn check_body(body: &FinalBody, endpoint: Endpoint) -> Result<(), EncodeError> {
    let Some(model) = body.get("model").and_then(serde_json::Value::as_str) else {
        return Ok(());
    };
    let Some(spec) = crate::catalog::lookup(super::wire::OPENAI.name, model) else {
        return Ok(());
    };
    let (effort, effort_field) = match endpoint {
        Endpoint::ChatCompletions => (body.get("reasoning_effort"), "reasoning_effort"),
        Endpoint::Responses => (body.pointer("/reasoning/effort"), "reasoning.effort"),
    };
    let reasons = match effort.and_then(serde_json::Value::as_str) {
        Some(effort) => effort != "none",
        None => spec.reasoning.default.is_some(),
    };
    let present = |field: &str| body.get(field).is_some_and(|value| !value.is_null());
    if reasons && spec.sampling == Some(Sampling::ReasoningOff) {
        let sampling: &[&str] = match endpoint {
            Endpoint::ChatCompletions => &["temperature", "top_p", "top_logprobs", "logprobs"],
            Endpoint::Responses => &["temperature", "top_p", "top_logprobs"],
        };
        if let Some(field) = sampling.iter().find(|field| present(field)) {
            let fix = match spec.reasoning.can_disable {
                true => format!("remove `{field}` or set `{effort_field}` to `none`"),
                false => format!("remove `{field}`"),
            };
            return Err(EncodeError::request(format!(
                "`{model}` rejects `{field}` while it reasons, which it does unless \
                 `{effort_field}` is `none`; {fix}"
            )));
        }
    }
    let carries_tools = body
        .get("tools")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    if endpoint == Endpoint::ChatCompletions
        && carries_tools
        && spec.compat.chat_tools_need_reasoning_off
        && !spec.reasoning.can_disable
    {
        return Err(EncodeError::request(format!(
            "`{model}` cannot call tools through Chat Completions; use the Responses API"
        )));
    }
    Ok(())
}

/// Why the model `spec` describes cannot take effort `effort`, from its
/// catalog levels, or `None` when it can or the catalog does not list it.
fn effort_refusal(spec: Option<&ModelSpec>, effort: &Effort) -> Option<Mapping> {
    let levels = &spec?.reasoning.levels;
    (!levels.contains(effort)).then(|| {
        Mapping::unsupported(format!(
            "this model has no `{}` effort level",
            effort.as_str()
        ))
    })
}

/// A `stop` list sent under `key`, refused past `limit` sequences.
fn stop_list(stop: &[String], limit: Option<usize>, key: &str) -> Mapping {
    match limit {
        Some(limit) if stop.len() > limit => {
            Mapping::unsupported(format!("at most {limit} stop sequences are taken"))
        }
        _ => Mapping::Send(json!({ key: stop })),
    }
}

/// `value` sent under `key`.
fn send(key: &str, value: impl Into<serde_json::Value>) -> Mapping {
    Mapping::Send(json!({ key: value.into() }))
}

/// `reasoning_effort` set to the level's word.
fn reasoning_effort(effort: &Effort) -> Mapping {
    send("reasoning_effort", effort.as_str())
}

/// Every option refused for `reason`.
fn refuse_all(fields: OptionFields<'_>, reason: &str) -> OptionMap {
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
    let refuse = |set: bool| match set {
        true => Mapping::unsupported(reason),
        false => Mapping::Nothing,
    };
    OptionMap {
        reasoning: refuse(reasoning.is_some()),
        cache: refuse(cache.is_some()),
        service_tier: refuse(service_tier.is_some()),
        verbosity: refuse(verbosity.is_some()),
        parallel_tool_calls: refuse(parallel_tool_calls.is_some()),
        top_p: refuse(top_p.is_some()),
        seed: refuse(seed.is_some()),
        stop: refuse(!stop.is_empty()),
    }
}

/// A cache the provider keeps without being asked, for a fixed time:
/// `Short` is honoured by sending nothing, the rest refused.
fn automatic_cache(cache: &CacheRetention) -> Mapping {
    match cache {
        CacheRetention::Short => Mapping::Omit("the provider caches prompts automatically"),
        CacheRetention::None => {
            Mapping::unsupported("automatic caching cannot be turned off per request")
        }
        CacheRetention::Long => Mapping::unsupported("the provider has no longer retention"),
    }
}

/// How the Chat Completions wire answers `fields` for `request`, by
/// dialect (section 6.2 of `TYPED_OPTIONS.md`).
pub(crate) fn chat_options(
    chat: &Chat,
    request: &CompletionRequest,
    fields: OptionFields<'_>,
) -> OptionMap {
    use super::wire::{
        AZURE, COHERE, DEEPSEEK, DOUBLEWORD, GROQ, HUGGINGFACE, HYPERBOLIC, LLAMACPP, MINIMAX,
        MIRA, MISTRAL, MOONSHOT, OLLAMA, OPENAI, OPENROUTER, PERPLEXITY, TOGETHER, VENICE,
        XIAOMIMIMO, ZAI,
    };
    let model = request.model.as_deref().unwrap_or(&chat.model);
    let name = chat.provider.dialect.name;
    let is = |dialect: &super::wire::Dialect| name == dialect.name;
    if is(&OPENAI) {
        openai_chat(model, fields, None)
    } else if is(&AZURE) {
        openai_chat(model, fields, Some(chat.provider.api_version.as_deref()))
    } else if is(&OPENROUTER) {
        openrouter(model, fields)
    } else if is(&DEEPSEEK) {
        deepseek(fields)
    } else if is(&MISTRAL) {
        mistral(fields)
    } else if is(&GROQ) {
        groq(model, fields)
    } else if name == crate::providers::xai::DIALECT.name {
        xai(model, fields)
    } else if is(&TOGETHER) {
        together(fields)
    } else if is(&VENICE) {
        venice(fields)
    } else if is(&MOONSHOT) {
        moonshot(model, fields)
    } else if is(&ZAI) {
        zai(model, fields)
    } else if is(&LLAMACPP) {
        llamacpp(fields)
    } else if is(&OLLAMA) {
        ollama(fields)
    } else if is(&COHERE) {
        cohere(fields)
    } else if is(&PERPLEXITY) {
        perplexity(model, fields)
    } else if is(&MINIMAX) {
        minimax(model, fields)
    } else if is(&XIAOMIMIMO) {
        xiaomimimo(fields)
    } else if is(&MIRA) {
        refuse_all(fields, "Mira rejects every pass-through parameter")
    } else if is(&HUGGINGFACE) || is(&HYPERBOLIC) || is(&DOUBLEWORD) {
        refuse_all(fields, "unverified for this provider")
    } else if name == crate::providers::copilot::wire::DIALECT.name {
        refuse_all(fields, "unverified for copilot")
    } else {
        refuse_all(
            fields,
            "no option mapping is known for this gateway; send it through `additional_params`",
        )
    }
}

/// OpenAI's own Chat Completions, and Azure's when `azure` names the
/// deployment's `api-version`.
fn openai_chat(model: &str, fields: OptionFields<'_>, azure: Option<Option<&str>>) -> OptionMap {
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
    // rig's default Azure `api-version` predates these fields.
    let dated = azure
        .flatten()
        .is_some_and(|version| version <= super::wire::AZURE_DEFAULT_API_VERSION);
    let refuse_dated = || {
        Mapping::unsupported(format!(
            "unverified on Azure `api-version` {}; set a later one",
            super::wire::AZURE_DEFAULT_API_VERSION
        ))
    };
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| {
            if dated {
                return refuse_dated();
            }
            let spec = openai_spec(model);
            match reasoning {
                Reasoning::Budget { .. } => {
                    Mapping::unsupported("Chat Completions takes an effort level, not a budget")
                }
                Reasoning::Off if reasons(model) == Some(false) => {
                    Mapping::Omit("the model does not reason")
                }
                _ if reasons(model) == Some(false) => {
                    Mapping::unsupported("the model does not reason")
                }
                Reasoning::Off if spec.is_some_and(|spec| !spec.reasoning.can_disable) => {
                    Mapping::unsupported("this model cannot turn reasoning off")
                }
                Reasoning::Off => send("reasoning_effort", "none"),
                Reasoning::Effort(Effort::Max) if azure.is_some() => {
                    Mapping::unsupported("Azure takes `max` effort only on Responses")
                }
                Reasoning::Effort(effort) => {
                    effort_refusal(spec, effort).unwrap_or_else(|| reasoning_effort(effort))
                }
            }
        }),
        cache: Mapping::of(cache, |cache| {
            if dated {
                return refuse_dated();
            }
            openai_cache(
                model,
                cache,
                Mapping::unsupported("Chat Completions has no longer retention from GPT-5.6 on"),
            )
        }),
        service_tier: Mapping::of(service_tier, |tier| match (tier, azure) {
            (_, Some(_)) => Mapping::unsupported("unverified for azure.openai"),
            (ServiceTier::Auto, _) => send("service_tier", "auto"),
            (ServiceTier::Default, _) => send("service_tier", "default"),
            (ServiceTier::Flex, _) => send("service_tier", "flex"),
            (ServiceTier::Priority, _) => send("service_tier", "priority"),
        }),
        verbosity: Mapping::of(verbosity, |verbosity| {
            if dated {
                return refuse_dated();
            }
            send("verbosity", verbosity.as_str())
        }),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| {
            send("parallel_tool_calls", parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |seed| send("seed", seed)),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, Some(4), "stop")),
    }
}

/// The OpenRouter upstreams that cache automatically, by `vendor/` prefix.
const AUTOMATIC_UPSTREAMS: [&str; 4] = ["openai/", "deepseek/", "x-ai/", "groq/"];

/// OpenRouter's `cache` for `model`, shared by its Chat and Responses
/// routes: a top-level marker for an upstream that takes one, nothing for
/// one that caches by itself.
pub(crate) fn openrouter_cache(model: &str, cache: &CacheRetention) -> Mapping {
    let automatic = AUTOMATIC_UPSTREAMS
        .iter()
        .any(|prefix| model.starts_with(prefix));
    match (cache, automatic) {
        (CacheRetention::None, true) => {
            Mapping::unsupported("this upstream caches automatically and cannot be stopped")
        }
        (CacheRetention::None, false) => Mapping::Omit("no cache marker is sent"),
        (CacheRetention::Short, true) => Mapping::Omit("this upstream caches automatically"),
        (CacheRetention::Short, false) => send("cache_control", json!({"type": "ephemeral"})),
        (CacheRetention::Long, true) => {
            Mapping::unsupported("this upstream has no one-hour cache retention")
        }
        (CacheRetention::Long, false) => {
            send("cache_control", json!({"type": "ephemeral", "ttl": "1h"}))
        }
    }
}

/// OpenRouter's `reasoning` object for `model`, shared by its two routes.
pub(crate) fn openrouter_reasoning(model: &str, reasoning: &Reasoning) -> Mapping {
    match reasoning {
        Reasoning::Off => send("reasoning", json!({"effort": "none"})),
        Reasoning::Effort(effort) => send("reasoning", json!({"effort": effort.as_str()})),
        Reasoning::Budget { .. } if model.starts_with("openai/") || model.starts_with("x-ai/") => {
            Mapping::unsupported("this upstream takes an effort level, not a budget")
        }
        Reasoning::Budget { tokens } => send("reasoning", json!({"max_tokens": tokens})),
    }
}

/// OpenRouter's `service_tier`, which has no `auto` value.
pub(crate) fn openrouter_tier(tier: &ServiceTier) -> Mapping {
    match tier {
        ServiceTier::Auto => Mapping::Omit("requests use the standard tier unless one is named"),
        ServiceTier::Default => send("service_tier", "default"),
        ServiceTier::Flex => send("service_tier", "flex"),
        ServiceTier::Priority => send("service_tier", "priority"),
    }
}

fn openrouter(model: &str, fields: OptionFields<'_>) -> OptionMap {
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
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| {
            openrouter_reasoning(model, reasoning)
        }),
        cache: Mapping::of(cache, |cache| openrouter_cache(model, cache)),
        service_tier: Mapping::of(service_tier, openrouter_tier),
        verbosity: Mapping::of(verbosity, |verbosity| send("verbosity", verbosity.as_str())),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| {
            send("parallel_tool_calls", parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |seed| send("seed", seed)),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, Some(4), "stop")),
    }
}

fn deepseek(fields: OptionFields<'_>) -> OptionMap {
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
    const UNDOCUMENTED: &str = "DeepSeek does not document this parameter";
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off => send("thinking", json!({"type": "disabled"})),
            Reasoning::Effort(effort @ (Effort::Low | Effort::High | Effort::Max)) => {
                Mapping::Send(json!({
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": effort.as_str(),
                }))
            }
            Reasoning::Effort(effort) => Mapping::unsupported(format!(
                "DeepSeek takes `low`, `high` and `max`, not `{}`",
                effort.as_str()
            )),
            Reasoning::Budget { .. } => {
                Mapping::unsupported("DeepSeek takes an effort level, not a budget")
            }
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::Short => Mapping::Omit("the disk cache is always on"),
            CacheRetention::None | CacheRetention::Long => {
                Mapping::unsupported("the disk cache is always on, with a fixed retention")
            }
        }),
        service_tier: Mapping::of(service_tier, |_| Mapping::unsupported(UNDOCUMENTED)),
        verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(UNDOCUMENTED)),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| {
            Mapping::unsupported(UNDOCUMENTED)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |_| Mapping::unsupported(UNDOCUMENTED)),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, Some(16), "stop")),
    }
}

fn mistral(fields: OptionFields<'_>) -> OptionMap {
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
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off => send("reasoning_effort", "none"),
            Reasoning::Effort(effort) => reasoning_effort(effort),
            Reasoning::Budget { .. } => {
                Mapping::unsupported("Mistral takes an effort level, not a budget")
            }
        }),
        cache: Mapping::of(cache, automatic_cache),
        service_tier: Mapping::of(service_tier, |tier| match tier {
            ServiceTier::Auto => send("service_tier", "auto"),
            ServiceTier::Default => send("service_tier", "standard_only"),
            ServiceTier::Flex | ServiceTier::Priority => {
                Mapping::unsupported("Mistral takes the `auto` and standard tiers only")
            }
        }),
        verbosity: Mapping::of(verbosity, |_| {
            Mapping::unsupported("Mistral has no verbosity setting")
        }),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| {
            send("parallel_tool_calls", parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |seed| send("random_seed", seed)),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, None, "stop")),
    }
}

fn groq(model: &str, fields: OptionFields<'_>) -> OptionMap {
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
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off
                if crate::catalog::lookup(super::wire::GROQ.name, model).is_some_and(|spec| {
                    spec.reasoning.supported && !spec.reasoning.can_disable
                }) =>
            {
                Mapping::unsupported("this model cannot turn reasoning off")
            }
            Reasoning::Off => send("reasoning_effort", "none"),
            Reasoning::Effort(effort) => reasoning_effort(effort),
            Reasoning::Budget { .. } => {
                Mapping::unsupported("Groq takes an effort level, not a budget")
            }
        }),
        cache: Mapping::of(cache, automatic_cache),
        service_tier: Mapping::of(service_tier, |tier| match tier {
            ServiceTier::Auto => send("service_tier", "auto"),
            ServiceTier::Default => send("service_tier", "on_demand"),
            ServiceTier::Flex => send("service_tier", "flex"),
            ServiceTier::Priority => send("service_tier", "performance"),
        }),
        verbosity: Mapping::of(verbosity, |_| {
            Mapping::unsupported("Groq has no verbosity setting")
        }),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| {
            send("parallel_tool_calls", parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |seed| send("seed", seed)),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, Some(4), "stop")),
    }
}

/// The catalog entry of xAI's `model`.
fn xai_spec(model: &str) -> Option<&'static ModelSpec> {
    crate::catalog::lookup(crate::providers::xai::DIALECT.name, model)
}

/// Whether xAI's `model` reasons, as its catalog entry says; a Grok model
/// the catalog does not list is taken to reason, as every Grok model since
/// Grok 4 does.
fn grok_reasons(model: &str) -> bool {
    xai_spec(model).is_none_or(|spec| spec.reasoning.supported)
}

/// xAI's `reasoning` on either route: `effort` spells the level as the
/// route takes it. The levels a model takes are its catalog entry's.
pub(crate) fn xai_reasoning(
    model: &str,
    reasoning: &Reasoning,
    effort: impl FnOnce(&Effort) -> Mapping,
) -> Mapping {
    match reasoning {
        Reasoning::Off if grok_reasons(model) => {
            Mapping::unsupported("a reasoning Grok model cannot turn reasoning off")
        }
        Reasoning::Off => Mapping::Omit("the model does not reason"),
        Reasoning::Effort(level @ (Effort::Minimal | Effort::Max)) => {
            Mapping::unsupported(format!("xAI has no `{}` effort level", level.as_str()))
        }
        Reasoning::Effort(level) => {
            effort_refusal(xai_spec(model), level).unwrap_or_else(|| effort(level))
        }
        Reasoning::Budget { .. } => Mapping::unsupported("xAI takes an effort level, not a budget"),
    }
}

/// xAI's `service_tier` on either route.
pub(crate) fn xai_tier(tier: &ServiceTier) -> Mapping {
    match tier {
        ServiceTier::Default => send("service_tier", "default"),
        ServiceTier::Priority => send("service_tier", "priority"),
        ServiceTier::Auto | ServiceTier::Flex => {
            Mapping::unsupported("xAI takes the default and priority tiers only")
        }
    }
}

fn xai(model: &str, fields: OptionFields<'_>) -> OptionMap {
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
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| {
            xai_reasoning(model, reasoning, reasoning_effort)
        }),
        cache: Mapping::of(cache, automatic_cache),
        service_tier: Mapping::of(service_tier, xai_tier),
        verbosity: Mapping::of(verbosity, |_| {
            Mapping::unsupported("xAI has no verbosity setting")
        }),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| {
            send("parallel_tool_calls", parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |_| Mapping::unsupported("unverified for xai")),
        stop: Mapping::of_stop(stop, |stop| match grok_reasons(model) {
            true => Mapping::unsupported("a reasoning Grok model takes no stop sequences"),
            false => stop_list(stop, None, "stop"),
        }),
    }
}

fn together(fields: OptionFields<'_>) -> OptionMap {
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
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off => send("reasoning", json!({"enabled": false})),
            Reasoning::Effort(effort @ (Effort::Low | Effort::Medium | Effort::High)) => {
                Mapping::Send(json!({
                    "reasoning": {"enabled": true},
                    "reasoning_effort": effort.as_str(),
                }))
            }
            Reasoning::Effort(effort) => Mapping::unsupported(format!(
                "Together has no `{}` effort level",
                effort.as_str()
            )),
            Reasoning::Budget { .. } => {
                Mapping::unsupported("Together takes an effort level, not a budget")
            }
        }),
        cache: Mapping::of(cache, automatic_cache),
        service_tier: Mapping::of(service_tier, |_| {
            Mapping::unsupported("Together has no service tiers")
        }),
        verbosity: Mapping::of(verbosity, |_| {
            Mapping::unsupported("Together has no verbosity setting")
        }),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| {
            Mapping::unsupported("unverified for together")
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |seed| send("seed", seed)),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, None, "stop")),
    }
}

fn venice(fields: OptionFields<'_>) -> OptionMap {
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
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off => send("reasoning_effort", "none"),
            Reasoning::Effort(effort) => reasoning_effort(effort),
            Reasoning::Budget { .. } => {
                Mapping::unsupported("Venice takes an effort level, not a budget")
            }
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::None => {
                Mapping::unsupported("Venice caching cannot be turned off per request")
            }
            CacheRetention::Short => send("prompt_cache_retention", "default"),
            CacheRetention::Long => send("prompt_cache_retention", "24h"),
        }),
        service_tier: Mapping::of(service_tier, |_| {
            Mapping::unsupported("Venice has no service tiers")
        }),
        verbosity: Mapping::of(verbosity, |_| {
            Mapping::unsupported("Venice has no verbosity setting")
        }),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| {
            send("parallel_tool_calls", parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |seed| send("seed", seed)),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, Some(4), "stop")),
    }
}

fn moonshot(model: &str, fields: OptionFields<'_>) -> OptionMap {
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
    const UNVERIFIED: &str = "unverified for moonshot";
    let name = model.rsplit('/').next().unwrap_or(model);
    let k3 = name.starts_with(crate::providers::moonshot::KIMI_K3);
    let k2_6 = name.starts_with(crate::providers::moonshot::KIMI_K2_6);
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off if k2_6 => send("thinking", json!({"type": "disabled"})),
            Reasoning::Off => Mapping::unsupported("this model cannot turn thinking off"),
            Reasoning::Effort(effort @ (Effort::Low | Effort::High | Effort::Max)) if k3 => {
                reasoning_effort(effort)
            }
            Reasoning::Effort(_) => Mapping::unsupported("this model takes no such effort level"),
            Reasoning::Budget { .. } => {
                Mapping::unsupported("Moonshot takes an effort level, not a budget")
            }
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::None => {
                Mapping::unsupported("Moonshot caching cannot be turned off per request")
            }
            CacheRetention::Short => send(
                "prompt_cache_options",
                json!({"mode": "implicit", "ttl": "5m"}),
            ),
            CacheRetention::Long => send(
                "prompt_cache_options",
                json!({"mode": "implicit", "ttl": "1h"}),
            ),
        }),
        service_tier: Mapping::of(service_tier, |_| Mapping::unsupported(UNVERIFIED)),
        verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(UNVERIFIED)),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| Mapping::unsupported(UNVERIFIED)),
        top_p: Mapping::of(top_p, |_| Mapping::unsupported(UNVERIFIED)),
        seed: Mapping::of(seed, |_| Mapping::unsupported(UNVERIFIED)),
        stop: Mapping::of_stop(stop, |stop| {
            if stop.iter().any(|sequence| sequence.len() > 32) {
                Mapping::unsupported("Moonshot takes stop sequences of at most 32 bytes")
            } else {
                stop_list(stop, Some(5), "stop")
            }
        }),
    }
}

fn zai(model: &str, fields: OptionFields<'_>) -> OptionMap {
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
    const UNSUPPORTED: &str = "Z.AI has no such parameter";
    // GLM 4 models switch thinking on and off and take no level.
    let toggle_only = model.starts_with("glm-4");
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off => send("thinking", json!({"type": "disabled"})),
            Reasoning::Effort(effort @ (Effort::Low | Effort::High | Effort::Max))
                if !toggle_only =>
            {
                Mapping::Send(json!({
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": effort.as_str(),
                }))
            }
            Reasoning::Effort(_) => Mapping::unsupported("this model takes no such effort level"),
            Reasoning::Budget { .. } => {
                Mapping::unsupported("Z.AI takes an effort level, not a budget")
            }
        }),
        cache: Mapping::of(cache, automatic_cache),
        service_tier: Mapping::of(service_tier, |_| Mapping::unsupported(UNSUPPORTED)),
        verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(UNSUPPORTED)),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| {
            Mapping::unsupported(UNSUPPORTED)
        }),
        top_p: Mapping::of(top_p, |top_p| {
            if (0.01..=1.0).contains(&top_p) {
                send("top_p", top_p)
            } else {
                Mapping::unsupported("Z.AI takes `top_p` from 0.01 to 1.0")
            }
        }),
        seed: Mapping::of(seed, |_| Mapping::unsupported(UNSUPPORTED)),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, Some(4), "stop")),
    }
}

fn llamacpp(fields: OptionFields<'_>) -> OptionMap {
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
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off => send("reasoning_effort", "none"),
            Reasoning::Effort(effort) => reasoning_effort(effort),
            Reasoning::Budget { .. } => Mapping::unsupported(
                "llama.cpp sets a reasoning budget per server, not per request",
            ),
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::None => send("cache_prompt", false),
            CacheRetention::Short => Mapping::Omit("llama-server reuses the prompt cache"),
            CacheRetention::Long => Mapping::unsupported("llama.cpp has no cache retention"),
        }),
        service_tier: Mapping::of(service_tier, |_| {
            Mapping::unsupported("llama.cpp has no service tiers")
        }),
        verbosity: Mapping::of(verbosity, |_| {
            Mapping::unsupported("llama.cpp has no verbosity setting")
        }),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| {
            send("parallel_tool_calls", parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |seed| send("seed", seed)),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, None, "stop")),
    }
}

fn ollama(fields: OptionFields<'_>) -> OptionMap {
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
    const UNSUPPORTED: &str = "Ollama's OpenAI-compatible API does not support it";
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off => send("reasoning_effort", "none"),
            Reasoning::Effort(effort) => reasoning_effort(effort),
            Reasoning::Budget { .. } => {
                Mapping::unsupported("Ollama takes an effort level, not a budget")
            }
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::Short => Mapping::Omit("Ollama reuses the loaded prompt"),
            CacheRetention::None | CacheRetention::Long => Mapping::unsupported(UNSUPPORTED),
        }),
        service_tier: Mapping::of(service_tier, |_| Mapping::unsupported(UNSUPPORTED)),
        verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(UNSUPPORTED)),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| {
            Mapping::unsupported(UNSUPPORTED)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |seed| send("seed", seed)),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, None, "stop")),
    }
}

fn cohere(fields: OptionFields<'_>) -> OptionMap {
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
    const UNSUPPORTED: &str = "Cohere's Compatibility API does not support it";
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off => send("reasoning_effort", "none"),
            Reasoning::Effort(Effort::High) => send("reasoning_effort", "high"),
            Reasoning::Effort(_) => {
                Mapping::unsupported("the Compatibility API takes only `none` and `high`")
            }
            Reasoning::Budget { .. } => {
                Mapping::unsupported("the Compatibility API takes no thinking budget")
            }
        }),
        cache: Mapping::of(cache, |_| {
            Mapping::unsupported("the Compatibility API has no cache control")
        }),
        service_tier: Mapping::of(service_tier, |_| Mapping::unsupported(UNSUPPORTED)),
        verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(UNSUPPORTED)),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| {
            Mapping::unsupported(UNSUPPORTED)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |seed| send("seed", seed)),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, None, "stop")),
    }
}

fn perplexity(model: &str, fields: OptionFields<'_>) -> OptionMap {
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
    const UNSUPPORTED: &str = "Perplexity does not support it";
    let deep_research = model.starts_with("sonar-deep-research");
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Effort(effort @ (Effort::Low | Effort::Medium | Effort::High))
                if deep_research =>
            {
                reasoning_effort(effort)
            }
            _ => Mapping::unsupported(
                "only `sonar-deep-research` takes a reasoning effort, `low` to `high`",
            ),
        }),
        cache: Mapping::of(cache, |_| Mapping::unsupported(UNSUPPORTED)),
        service_tier: Mapping::of(service_tier, |_| Mapping::unsupported(UNSUPPORTED)),
        verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(UNSUPPORTED)),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| {
            Mapping::unsupported(UNSUPPORTED)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |_| Mapping::unsupported("unverified for perplexity")),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, None, "stop")),
    }
}

fn minimax(model: &str, fields: OptionFields<'_>) -> OptionMap {
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
    const UNSUPPORTED: &str = "MiniMax does not support it";
    let model = model.to_ascii_lowercase();
    let flash = model.starts_with("minimax-m3.1");
    let m3 = !flash && model.starts_with("minimax-m3");
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off if m3 => send("thinking", json!({"type": "disabled"})),
            Reasoning::Off => Mapping::unsupported("thinking cannot be disabled on this model"),
            Reasoning::Effort(Effort::Minimal) => {
                Mapping::unsupported("MiniMax has no `minimal` effort level")
            }
            Reasoning::Effort(effort) if flash => Mapping::Send(json!({
                "thinking": {"type": "adaptive"},
                "reasoning_effort": effort.as_str(),
            })),
            Reasoning::Effort(_) => Mapping::unsupported("this model takes no effort level"),
            Reasoning::Budget { .. } => Mapping::unsupported(UNSUPPORTED),
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::Short => Mapping::Omit("MiniMax caches prompts passively"),
            CacheRetention::None | CacheRetention::Long => Mapping::unsupported(UNSUPPORTED),
        }),
        service_tier: Mapping::of(service_tier, |_| Mapping::unsupported(UNSUPPORTED)),
        verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(UNSUPPORTED)),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| {
            Mapping::unsupported("unverified for minimax")
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |_| Mapping::unsupported("unverified for minimax")),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, None, "stop")),
    }
}

fn xiaomimimo(fields: OptionFields<'_>) -> OptionMap {
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
    const UNSUPPORTED: &str = "MiMo does not support it";
    let thinking_off = matches!(reasoning, Some(Reasoning::Off));
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off => send("thinking", json!({"type": "disabled"})),
            _ => Mapping::unsupported("MiMo takes only thinking enabled or disabled"),
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::Short => Mapping::Omit("MiMo caches prompts implicitly"),
            CacheRetention::None | CacheRetention::Long => Mapping::unsupported(UNSUPPORTED),
        }),
        service_tier: Mapping::of(service_tier, |_| Mapping::unsupported(UNSUPPORTED)),
        verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(UNSUPPORTED)),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| {
            Mapping::unsupported("MiMo has no `parallel_tool_calls` parameter")
        }),
        top_p: Mapping::of(top_p, |top_p| match thinking_off {
            true => send("top_p", top_p),
            false => Mapping::unsupported("MiMo fixes `top_p` at 0.95 while thinking"),
        }),
        seed: Mapping::of(seed, |_| {
            Mapping::unsupported("MiMo has no `seed` parameter")
        }),
        stop: Mapping::of_stop(stop, |stop| stop_list(stop, Some(4), "stop")),
    }
}

/// `verbosity` for the Responses API, inside `text`.
fn text_verbosity(verbosity: &crate::completion::Verbosity) -> Mapping {
    send("text", json!({"verbosity": verbosity.as_str()}))
}

/// How the Responses wire answers `fields` for `request`, by dialect
/// (section 6.3 of `TYPED_OPTIONS.md`). Responses has no `seed` and no
/// `stop`.
pub(crate) fn responses_options(
    responses: &super::responses_api::wire::Responses,
    request: &CompletionRequest,
    fields: OptionFields<'_>,
) -> OptionMap {
    use super::responses_api::wire::ResponsesContract;
    use super::wire::{OPENAI, OPENROUTER};
    let model = request.model.as_deref().unwrap_or(&responses.model);
    let dialect = &responses.provider.dialect;
    match dialect.quirks.responses.contract {
        ResponsesContract::Codex => codex(fields),
        ResponsesContract::Xai => xai_responses(model, fields),
        ResponsesContract::OpenAi if dialect.name == OPENAI.name => openai_responses(model, fields),
        ResponsesContract::OpenAi if dialect.name == OPENROUTER.name => {
            openrouter_responses(model, fields)
        }
        ResponsesContract::OpenAi
            if dialect.name == crate::providers::copilot::wire::DIALECT.name =>
        {
            copilot_responses(fields)
        }
        ResponsesContract::OpenAi => refuse_all(
            fields,
            "no Responses option mapping is known for this provider; send it through \
             `additional_params`",
        ),
    }
}

/// `reasoning` on the Responses API, as `reasoning.effort`.
fn reasoning_object(effort: &str) -> Mapping {
    send("reasoning", json!({"effort": effort}))
}

fn openai_responses(model: &str, fields: OptionFields<'_>) -> OptionMap {
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
    let spec = openai_spec(model);
    let reasoning_off = matches!(reasoning, Some(Reasoning::Off));
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Budget { .. } => {
                Mapping::unsupported("Responses takes an effort level, not a budget")
            }
            Reasoning::Off if reasons(model) == Some(false) => {
                Mapping::Omit("the model does not reason")
            }
            _ if reasons(model) == Some(false) => Mapping::unsupported("the model does not reason"),
            Reasoning::Off if spec.is_some_and(|spec| !spec.reasoning.can_disable) => {
                Mapping::unsupported("this model cannot turn reasoning off")
            }
            Reasoning::Off => reasoning_object("none"),
            Reasoning::Effort(effort) => {
                effort_refusal(spec, effort).unwrap_or_else(|| reasoning_object(effort.as_str()))
            }
        }),
        cache: Mapping::of(cache, |cache| {
            openai_cache(
                model,
                cache,
                send("prompt_cache_options", json!({"ttl": "30m"})),
            )
        }),
        service_tier: Mapping::of(service_tier, |tier| match tier {
            ServiceTier::Auto => send("service_tier", "auto"),
            ServiceTier::Default => send("service_tier", "default"),
            ServiceTier::Flex => send("service_tier", "flex"),
            ServiceTier::Priority => send("service_tier", "priority"),
        }),
        verbosity: Mapping::of(verbosity, text_verbosity),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| {
            send("parallel_tool_calls", parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| match spec {
            Some(spec) if spec.reasoning.supported => match spec.sampling {
                Some(Sampling::Any) => send("top_p", top_p),
                Some(Sampling::ReasoningOff) if reasoning_off => send("top_p", top_p),
                Some(Sampling::ReasoningOff) => {
                    Mapping::unsupported("a reasoning model takes `top_p` only at effort `none`")
                }
                Some(Sampling::Never) => {
                    Mapping::unsupported("this model takes no sampling parameters")
                }
                None => Mapping::unsupported(
                    "unverified for this model, whose sampling rule the catalog lacks",
                ),
            },
            _ => send("top_p", top_p),
        }),
        seed: Mapping::of(seed, |_| {
            Mapping::unsupported("Responses has no seed parameter")
        }),
        stop: Mapping::of_stop(stop, |_| {
            Mapping::unsupported("Responses has no stop parameter")
        }),
    }
}

fn xai_responses(model: &str, fields: OptionFields<'_>) -> OptionMap {
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
    const UNSUPPORTED: &str = "xAI's Responses API does not support it";
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off if xai_spec(model).is_some_and(|spec| spec.reasoning.can_disable) => {
                reasoning_object("none")
            }
            reasoning => {
                xai_reasoning(model, reasoning, |effort| reasoning_object(effort.as_str()))
            }
        }),
        cache: Mapping::of(cache, automatic_cache),
        service_tier: Mapping::of(service_tier, xai_tier),
        verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(UNSUPPORTED)),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| {
            send("parallel_tool_calls", parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |_| Mapping::unsupported(UNSUPPORTED)),
        stop: Mapping::of_stop(stop, |_| Mapping::unsupported(UNSUPPORTED)),
    }
}

fn openrouter_responses(model: &str, fields: OptionFields<'_>) -> OptionMap {
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
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| {
            openrouter_reasoning(model, reasoning)
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::None if model.starts_with("openai/") && caches_by_options(model) => {
                send("prompt_cache_options", json!({"mode": "explicit"}))
            }
            cache => openrouter_cache(model, cache),
        }),
        service_tier: Mapping::of(service_tier, openrouter_tier),
        verbosity: Mapping::of(verbosity, text_verbosity),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| {
            send("parallel_tool_calls", parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |_| {
            Mapping::unsupported("Responses has no seed parameter")
        }),
        stop: Mapping::of_stop(stop, |_| {
            Mapping::unsupported("Responses has no stop parameter")
        }),
    }
}

/// The ChatGPT (Codex) backend: no public reference; the cells follow the
/// Codex CLI's requests and the recorded echoes.
fn codex(fields: OptionFields<'_>) -> OptionMap {
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
    const UNSUPPORTED: &str = "the ChatGPT backend does not take it";
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off => reasoning_object("none"),
            Reasoning::Effort(effort) => reasoning_object(effort.as_str()),
            Reasoning::Budget { .. } => {
                Mapping::unsupported("the ChatGPT backend takes an effort level, not a budget")
            }
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::Long => Mapping::Omit("the backend keeps prompts for 24 hours"),
            CacheRetention::None | CacheRetention::Short => {
                Mapping::unsupported("the ChatGPT backend keeps prompts for 24 hours")
            }
        }),
        service_tier: Mapping::of(service_tier, |tier| match tier {
            ServiceTier::Flex => send("service_tier", "flex"),
            ServiceTier::Priority => send("service_tier", "priority"),
            ServiceTier::Auto | ServiceTier::Default => Mapping::unsupported(UNSUPPORTED),
        }),
        verbosity: Mapping::of(verbosity, text_verbosity),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| {
            send("parallel_tool_calls", parallel)
        }),
        top_p: Mapping::of(top_p, |_| Mapping::unsupported(UNSUPPORTED)),
        seed: Mapping::of(seed, |_| Mapping::unsupported(UNSUPPORTED)),
        stop: Mapping::of_stop(stop, |_| Mapping::unsupported(UNSUPPORTED)),
    }
}

/// Copilot's Responses route (`*codex*` models): no public reference; the
/// cells follow its recorded streams.
fn copilot_responses(fields: OptionFields<'_>) -> OptionMap {
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
    const UNVERIFIED: &str = "unverified for copilot";
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Effort(Effort::Minimal) => {
                Mapping::unsupported("Copilot's codex models have no `minimal` effort")
            }
            Reasoning::Effort(effort) => reasoning_object(effort.as_str()),
            Reasoning::Off => {
                Mapping::unsupported("Copilot's codex models cannot turn reasoning off")
            }
            Reasoning::Budget { .. } => {
                Mapping::unsupported("Copilot takes an effort level, not a budget")
            }
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::Long => Mapping::Omit("Copilot echoes a 24 hour retention"),
            CacheRetention::None | CacheRetention::Short => Mapping::unsupported(UNVERIFIED),
        }),
        service_tier: Mapping::of(service_tier, |_| Mapping::unsupported(UNVERIFIED)),
        verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(UNVERIFIED)),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| {
            send("parallel_tool_calls", parallel)
        }),
        top_p: Mapping::of(top_p, |top_p| send("top_p", top_p)),
        seed: Mapping::of(seed, |_| {
            Mapping::unsupported("Responses has no seed parameter")
        }),
        stop: Mapping::of_stop(stop, |_| {
            Mapping::unsupported("Responses has no stop parameter")
        }),
    }
}

#[cfg(test)]
mod tests;
