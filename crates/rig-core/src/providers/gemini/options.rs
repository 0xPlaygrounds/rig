//! How the Gemini wires answer
//! [`GenerationOptions`](crate::completion::GenerationOptions): one mapping
//! for every GenerateContent route (the Gemini API, Vertex AI, gRPC), and
//! one for Interactions. Which thinking a model takes is its catalog
//! entry's.

use serde_json::json;

use crate::completion::options::{Mapping, OptionFields, OptionMap};
use crate::completion::{CacheRetention, Effort, Reasoning, ServiceTier};

/// The GenerateContent route a request takes, which decides the service
/// tiers it can name and whether an unverified field reaches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Route {
    /// The Gemini API's REST endpoint.
    Rest,
    /// Vertex AI, through its SDK.
    Vertex,
    /// The Gemini API over gRPC.
    Grpc,
}

/// The thinking a Gemini model takes, from its catalog entry.
enum Thinking {
    /// These levels (Gemini 3), and no way to turn thinking off.
    Levels(&'static [Effort]),
    /// A token budget in this range (Gemini 2.5), and whether `0` turns it
    /// off.
    Budget {
        range: std::ops::RangeInclusive<u32>,
        can_disable: bool,
    },
    /// A model that does not think.
    None,
    /// A model the catalog does not list.
    Unknown,
}

/// The thinking `model` takes, by its id past a `models/` prefix.
fn thinking(model: &str) -> Thinking {
    let model = model.to_ascii_lowercase();
    let model = model.strip_prefix("models/").unwrap_or(&model);
    let Some(spec) = crate::catalog::lookup(super::PROVIDER_NAME, model) else {
        return Thinking::Unknown;
    };
    let support = &spec.reasoning;
    match &support.budget {
        _ if !support.supported => Thinking::None,
        Some(range) if support.levels.is_empty() => Thinking::Budget {
            range: range.clone(),
            can_disable: support.can_disable,
        },
        _ => Thinking::Levels(&support.levels),
    }
}

/// `value` under `generationConfig`.
fn config(value: serde_json::Value) -> Mapping {
    Mapping::Send(json!({ "generationConfig": value }))
}

/// How a GenerateContent wire on `route` answers `fields` for `model`
/// (section 6.4 of `TYPED_OPTIONS.md`). Every GenerateContent wire calls it,
/// so the REST, Vertex AI and gRPC wires agree.
pub fn generate_content_options(model: &str, route: Route, fields: OptionFields<'_>) -> OptionMap {
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
    let thinking = thinking(model);
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match (reasoning, &thinking) {
            (
                Reasoning::Off,
                Thinking::Budget {
                    can_disable: true, ..
                },
            ) => config(json!({"thinkingConfig": {"thinkingBudget": 0}})),
            (Reasoning::Off, Thinking::None) => Mapping::Omit("the model does not think"),
            (Reasoning::Off, _) => Mapping::unsupported("this model cannot turn thinking off"),
            (Reasoning::Effort(Effort::XHigh | Effort::Max), _) => {
                Mapping::unsupported("Gemini's thinking levels stop at `high`")
            }
            (Reasoning::Effort(_), _) if route == Route::Grpc => Mapping::unsupported(
                "unverified for gemini-grpc: Google's proto has no thinking level",
            ),
            (Reasoning::Effort(effort), Thinking::Levels(levels)) if levels.contains(effort) => {
                config(json!({"thinkingConfig": {"thinkingLevel": effort.as_str()}}))
            }
            (Reasoning::Effort(effort), Thinking::Levels(_)) => Mapping::unsupported(format!(
                "this model has no `{}` thinking level",
                effort.as_str()
            )),
            (Reasoning::Effort(_), Thinking::Budget { .. }) => Mapping::unsupported(
                "Gemini 2.5 takes a thinking budget, and rig has no effort-to-budget table",
            ),
            (Reasoning::Effort(_), Thinking::None) => {
                Mapping::unsupported("the model does not think")
            }
            (Reasoning::Effort(effort), Thinking::Unknown) => {
                config(json!({"thinkingConfig": {"thinkingLevel": effort.as_str()}}))
            }
            (Reasoning::Budget { tokens }, Thinking::Budget { range, .. }) => {
                if range.contains(tokens) {
                    config(json!({"thinkingConfig": {"thinkingBudget": tokens}}))
                } else {
                    Mapping::unsupported(format!(
                        "this model takes a thinking budget from {} to {}",
                        range.start(),
                        range.end()
                    ))
                }
            }
            (Reasoning::Budget { .. }, Thinking::Levels(_)) => {
                Mapping::unsupported("Gemini 3 takes a thinking level, not a budget")
            }
            (Reasoning::Budget { .. }, Thinking::None) => {
                Mapping::unsupported("the model does not think")
            }
            (Reasoning::Budget { tokens }, Thinking::Unknown) => {
                config(json!({"thinkingConfig": {"thinkingBudget": tokens}}))
            }
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::Short => Mapping::Omit("Gemini caches prompts implicitly"),
            CacheRetention::None => {
                Mapping::unsupported("implicit caching cannot be turned off per request")
            }
            CacheRetention::Long => Mapping::unsupported(
                "explicit caching is a `cachedContents` resource, not a request field",
            ),
        }),
        service_tier: Mapping::of(service_tier, |tier| match (tier, route) {
            (ServiceTier::Auto, _) => Mapping::Omit("the standard tier is the default"),
            (_, Route::Grpc) => Mapping::unsupported("the gRPC request has no service tier"),
            (ServiceTier::Default, Route::Vertex) => {
                Mapping::Omit("Vertex AI serves standard pay-as-you-go by default")
            }
            (ServiceTier::Flex | ServiceTier::Priority, Route::Vertex) => Mapping::unsupported(
                "Vertex AI selects this tier by request headers its SDK transport cannot set",
            ),
            (ServiceTier::Default, Route::Rest) => {
                Mapping::Send(json!({"serviceTier": "standard"}))
            }
            (ServiceTier::Flex, Route::Rest) => Mapping::Send(json!({"serviceTier": "flex"})),
            (ServiceTier::Priority, Route::Rest) => {
                Mapping::Send(json!({"serviceTier": "priority"}))
            }
        }),
        verbosity: Mapping::of(verbosity, |_| {
            Mapping::unsupported("Gemini has no verbosity setting")
        }),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| match parallel {
            true => Mapping::Omit("parallel function calls are the default"),
            false => Mapping::unsupported("Gemini cannot turn parallel function calls off"),
        }),
        top_p: Mapping::of(top_p, |top_p| config(json!({"topP": top_p}))),
        seed: Mapping::of(seed, |seed| match i32::try_from(seed) {
            Ok(seed) => config(json!({"seed": seed})),
            Err(_) => Mapping::unsupported("Gemini's seed is a 32-bit integer"),
        }),
        stop: Mapping::of_stop(stop, |stop| match stop.len() {
            0..=5 => config(json!({"stopSequences": stop})),
            _ => Mapping::unsupported("Gemini takes at most 5 stop sequences"),
        }),
    }
}

/// `value` under `generation_config`, the Interactions spelling.
fn generation_config(value: serde_json::Value) -> Mapping {
    Mapping::Send(json!({ "generation_config": value }))
}

/// How the Interactions wire answers `fields` for `model`.
pub(super) fn interactions(model: &str, fields: OptionFields<'_>) -> OptionMap {
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
    const NO_FIELD: &str = "the Interactions API has no such field";
    let thinking = thinking(model);
    OptionMap {
        reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
            Reasoning::Off | Reasoning::Budget { .. } => Mapping::unsupported(
                "the Interactions API takes a thinking level: no budget, no off",
            ),
            Reasoning::Effort(Effort::XHigh | Effort::Max) => {
                Mapping::unsupported("Gemini's thinking levels stop at `high`")
            }
            Reasoning::Effort(effort) => match &thinking {
                Thinking::Levels(levels) if !levels.contains(effort) => Mapping::unsupported(
                    format!("this model has no `{}` thinking level", effort.as_str()),
                ),
                Thinking::Budget { .. } | Thinking::None => {
                    Mapping::unsupported("this model takes no thinking level")
                }
                Thinking::Levels(_) | Thinking::Unknown => {
                    generation_config(json!({"thinking_level": effort.as_str()}))
                }
            },
        }),
        cache: Mapping::of(cache, |cache| match cache {
            CacheRetention::Short => Mapping::Omit("Gemini caches prompts implicitly"),
            CacheRetention::None | CacheRetention::Long => {
                Mapping::unsupported("the Interactions API has no cache control")
            }
        }),
        service_tier: Mapping::of(service_tier, |tier| match tier {
            ServiceTier::Auto => Mapping::Omit("the standard tier is the default"),
            ServiceTier::Default => Mapping::Send(json!({"service_tier": "standard"})),
            ServiceTier::Flex => Mapping::Send(json!({"service_tier": "flex"})),
            ServiceTier::Priority => Mapping::Send(json!({"service_tier": "priority"})),
        }),
        verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(NO_FIELD)),
        parallel_tool_calls: Mapping::of(parallel_tool_calls, |parallel| match parallel {
            true => Mapping::Omit("parallel function calls are the default"),
            false => Mapping::unsupported(NO_FIELD),
        }),
        top_p: Mapping::of(top_p, |_| {
            Mapping::unsupported("unverified for the Interactions API, which no longer lists it")
        }),
        seed: Mapping::of(seed, |seed| match i32::try_from(seed) {
            Ok(seed) => generation_config(json!({"seed": seed})),
            Err(_) => Mapping::unsupported("Gemini's seed is a 32-bit integer"),
        }),
        stop: Mapping::of_stop(stop, |stop| match stop.len() {
            0..=5 => generation_config(json!({"stop_sequences": stop})),
            _ => Mapping::unsupported("Gemini takes at most 5 stop sequences"),
        }),
    }
}

#[cfg(test)]
mod tests;
