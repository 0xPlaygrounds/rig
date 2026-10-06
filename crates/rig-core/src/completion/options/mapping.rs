//! What a completion wire answers for each option: [`OptionFields`] is the
//! request's options as a wire reads them, and [`OptionMap`] holds one
//! [`Mapping`] per option. Neither has `..` or a default, so a new option
//! fails to compile in every wire until it answers for it.

use serde_json::Value;

use super::{CacheRetention, Reasoning, ServiceTier, Verbosity};

/// Every option but the policy, borrowed, as a wire's
/// [`map_options`](crate::completion::ReplayTarget::map_options) reads it.
/// Not `#[non_exhaustive]`, so wires in companion crates destructure it
/// whole, which `#[non_exhaustive]` forbids for
/// [`GenerationOptions`](super::GenerationOptions).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptionFields<'a> {
    /// How much the model reasons.
    pub reasoning: Option<&'a Reasoning>,
    /// How long the prompt prefix stays cached.
    pub cache: Option<&'a CacheRetention>,
    /// The processing tier.
    pub service_tier: Option<&'a ServiceTier>,
    /// How long the answer should be.
    pub verbosity: Option<&'a Verbosity>,
    /// Whether several tool calls may come in one turn.
    pub parallel_tool_calls: Option<bool>,
    /// Nucleus sampling probability mass.
    pub top_p: Option<f64>,
    /// Sampling seed.
    pub seed: Option<u64>,
    /// Stop sequences; empty means unset.
    pub stop: &'a [String],
}

impl OptionFields<'_> {
    /// Which options are set, in [`OptionMap`] field order.
    pub(super) fn set(&self) -> [bool; 8] {
        let OptionFields {
            reasoning,
            cache,
            service_tier,
            verbosity,
            parallel_tool_calls,
            top_p,
            seed,
            stop,
        } = self;
        [
            reasoning.is_some(),
            cache.is_some(),
            service_tier.is_some(),
            verbosity.is_some(),
            parallel_tool_calls.is_some(),
            top_p.is_some(),
            seed.is_some(),
            !stop.is_empty(),
        ]
    }
}

/// What a wire does with one option.
#[derive(Clone, Debug, PartialEq)]
pub enum Mapping {
    /// The option is unset. For a set option this is an encode error.
    Nothing,
    /// Deep-merge this JSON object into the body, above the wire's own
    /// encoding of the request and below `additional_params`.
    Send(Value),
    /// Honour the option by sending nothing: the provider default already
    /// does what was asked. Logged at `debug` with the reason.
    Omit(&'static str),
    /// Honour the option through markers the wire's base builder writes
    /// inside the body's arrays. Valid for `cache` only.
    Place,
    /// The wire or model cannot honour it: an error under
    /// [`OnUnsupported::Error`](super::OnUnsupported::Error), a warning
    /// under [`OnUnsupported::Ignore`](super::OnUnsupported::Ignore).
    Unsupported(String),
}

impl Mapping {
    /// `Nothing` for an unset option, otherwise what `map` answers for it.
    pub fn of<T>(value: Option<T>, map: impl FnOnce(T) -> Mapping) -> Mapping {
        value.map_or(Mapping::Nothing, map)
    }

    /// `Nothing` for an empty stop list, otherwise what `map` answers.
    pub fn of_stop(stop: &[String], map: impl FnOnce(&[String]) -> Mapping) -> Mapping {
        if stop.is_empty() {
            Mapping::Nothing
        } else {
            map(stop)
        }
    }

    /// The refusal for `reason`.
    pub fn unsupported(reason: impl Into<String>) -> Mapping {
        Mapping::Unsupported(reason.into())
    }
}

/// One [`Mapping`] per option, by field name. Not `#[non_exhaustive]` and no
/// `Default`, so a wire writes every field.
#[derive(Clone, Debug, PartialEq)]
pub struct OptionMap {
    /// [`GenerationOptions::reasoning`](super::GenerationOptions::reasoning).
    pub reasoning: Mapping,
    /// [`GenerationOptions::cache`](super::GenerationOptions::cache).
    pub cache: Mapping,
    /// [`GenerationOptions::service_tier`](super::GenerationOptions::service_tier).
    pub service_tier: Mapping,
    /// [`GenerationOptions::verbosity`](super::GenerationOptions::verbosity).
    pub verbosity: Mapping,
    /// [`GenerationOptions::parallel_tool_calls`](super::GenerationOptions::parallel_tool_calls).
    pub parallel_tool_calls: Mapping,
    /// [`GenerationOptions::top_p`](super::GenerationOptions::top_p).
    pub top_p: Mapping,
    /// [`GenerationOptions::seed`](super::GenerationOptions::seed).
    pub seed: Mapping,
    /// [`GenerationOptions::stop`](super::GenerationOptions::stop).
    pub stop: Mapping,
}

impl OptionMap {
    /// Every option's name and answer, in field order.
    pub(super) fn into_slots(self) -> [(&'static str, Mapping); 8] {
        let OptionMap {
            reasoning,
            cache,
            service_tier,
            verbosity,
            parallel_tool_calls,
            top_p,
            seed,
            stop,
        } = self;
        [
            ("reasoning", reasoning),
            ("cache", cache),
            ("service_tier", service_tier),
            ("verbosity", verbosity),
            ("parallel_tool_calls", parallel_tool_calls),
            ("top_p", top_p),
            ("seed", seed),
            ("stop", stop),
        ]
    }
}

/// Every set option refused: the answer of a wire whose mapping has not
/// landed yet.
#[doc(hidden)]
pub fn unmapped(fields: OptionFields<'_>) -> OptionMap {
    let [
        reasoning,
        cache,
        service_tier,
        verbosity,
        parallel_tool_calls,
        top_p,
        seed,
        stop,
    ] = fields.set().map(|set| match set {
        true => Mapping::unsupported("this wire maps no options yet"),
        false => Mapping::Nothing,
    });
    OptionMap {
        reasoning,
        cache,
        service_tier,
        verbosity,
        parallel_tool_calls,
        top_p,
        seed,
        stop,
    }
}
