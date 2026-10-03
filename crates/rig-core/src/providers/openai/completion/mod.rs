//! Chat Completions model identifiers, the tool choice the OpenAI wires
//! share, and the usage counters of OpenAI's replies.
//!
//! ```
//! use rig_core::providers::openai::completion::{GPT_4O, ToolChoice};
//! assert_eq!(serde_json::to_value(ToolChoice::Required)?, "required");
//! # let _ = GPT_4O;
//! # Ok::<(), serde_json::Error>(())
//! ```

use crate::json_utils;
use serde::{Deserialize, Serialize, Serializer};
use std::fmt;

/// GPT-6 Astra, API ID `gpt-6-astra`: a reasoning model. Chat Completions
/// takes its function tools only at `reasoning_effort: "none"`, which it does
/// not support, so a Chat request with tools (the extractor's included) is
/// refused before it is sent: use the Responses wire.
pub const GPT_6_ASTRA: &str = "gpt-6-astra";

/// GPT-6.1 Sol, API ID `gpt-6.1-sol`: a reasoning model. Chat Completions
/// takes its function tools only at `reasoning_effort: "none"`, which it does
/// not support, so a Chat request with tools (the extractor's included) is
/// refused before it is sent: use the Responses wire.
pub const GPT_6_1_SOL: &str = "gpt-6.1-sol";

/// GPT-6 Sol, API ID `gpt-6-sol`: a reasoning model. Chat Completions takes
/// its function tools only at `reasoning_effort: "none"`: a Chat request with
/// tools is refused before it is sent unless `additional_params` carries
/// `"reasoning_effort": "none"`. Responses takes them at any effort.
pub const GPT_6_SOL: &str = "gpt-6-sol";

/// GPT-6 Luna, API ID `gpt-6-luna`: a reasoning model. Chat Completions takes
/// its function tools only at `reasoning_effort: "none"`: a Chat request with
/// tools is refused before it is sent unless `additional_params` carries
/// `"reasoning_effort": "none"`. Responses takes them at any effort.
pub const GPT_6_LUNA: &str = "gpt-6-luna";

/// `gpt-5.6` completion model (alias that routes to GPT-5.6 Sol)
pub const GPT_5_6: &str = "gpt-5.6";

/// `gpt-5.6-sol` completion model
pub const GPT_5_6_SOL: &str = "gpt-5.6-sol";

/// `gpt-5.6-terra` completion model
pub const GPT_5_6_TERRA: &str = "gpt-5.6-terra";

/// `gpt-5.6-luna` completion model
pub const GPT_5_6_LUNA: &str = "gpt-5.6-luna";

/// `gpt-5.5` completion model
pub const GPT_5_5: &str = "gpt-5.5";

/// GPT-5.4, API ID `gpt-5.4`: a reasoning model.
pub const GPT_5_4: &str = "gpt-5.4";

/// GPT-5.4 mini, API ID `gpt-5.4-mini`: a reasoning model.
pub const GPT_5_4_MINI: &str = "gpt-5.4-mini";

/// GPT-5.4 nano, API ID `gpt-5.4-nano`: a reasoning model.
pub const GPT_5_4_NANO: &str = "gpt-5.4-nano";

/// `gpt-5.2` completion model
pub const GPT_5_2: &str = "gpt-5.2";

/// GPT-5.2 Pro, API ID `gpt-5.2-pro`: a reasoning model served on the
/// Responses API only, without structured outputs.
pub const GPT_5_2_PRO: &str = "gpt-5.2-pro";

/// `gpt-5.1` completion model
pub const GPT_5_1: &str = "gpt-5.1";

/// `gpt-5` completion model
pub const GPT_5: &str = "gpt-5";
/// `gpt-5-mini` completion model.
pub const GPT_5_MINI: &str = "gpt-5-mini";
/// `gpt-5-nano` completion model.
pub const GPT_5_NANO: &str = "gpt-5-nano";

/// `gpt-4.5-preview` completion model
pub const GPT_4_5_PREVIEW: &str = "gpt-4.5-preview";
/// `gpt-4.5-preview-2025-02-27` completion model
pub const GPT_4_5_PREVIEW_2025_02_27: &str = "gpt-4.5-preview-2025-02-27";
/// `gpt-4o-2024-11-20` completion model.
pub const GPT_4O_2024_11_20: &str = "gpt-4o-2024-11-20";
/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-mini` completion model
pub const GPT_4O_MINI: &str = "gpt-4o-mini";
/// `gpt-4o-2024-05-13` completion model
pub const GPT_4O_2024_05_13: &str = "gpt-4o-2024-05-13";
/// `gpt-4-turbo` completion model
pub const GPT_4_TURBO: &str = "gpt-4-turbo";
/// `gpt-4-turbo-2024-04-09` completion model
pub const GPT_4_TURBO_2024_04_09: &str = "gpt-4-turbo-2024-04-09";
/// `gpt-4-turbo-preview` completion model
pub const GPT_4_TURBO_PREVIEW: &str = "gpt-4-turbo-preview";
/// `gpt-4-0125-preview` completion model
pub const GPT_4_0125_PREVIEW: &str = "gpt-4-0125-preview";
/// `gpt-4-1106-preview` completion model
pub const GPT_4_1106_PREVIEW: &str = "gpt-4-1106-preview";
/// `gpt-4-vision-preview` completion model
pub const GPT_4_VISION_PREVIEW: &str = "gpt-4-vision-preview";
/// `gpt-4-1106-vision-preview` completion model
pub const GPT_4_1106_VISION_PREVIEW: &str = "gpt-4-1106-vision-preview";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-0613` completion model
pub const GPT_4_0613: &str = "gpt-4-0613";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k-0613` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k-0613";

/// `o4-mini-2025-04-16` completion model
pub const O4_MINI_2025_04_16: &str = "o4-mini-2025-04-16";
/// `o4-mini` completion model
pub const O4_MINI: &str = "o4-mini";
/// `o3` completion model
pub const O3: &str = "o3";
/// `o3-mini` completion model
pub const O3_MINI: &str = "o3-mini";
/// `o3-mini-2025-01-31` completion model
pub const O3_MINI_2025_01_31: &str = "o3-mini-2025-01-31";
/// `o1-pro` completion model
pub const O1_PRO: &str = "o1-pro";
/// `o1` completion model.
pub const O1: &str = "o1";
/// `o1-2024-12-17` completion model
pub const O1_2024_12_17: &str = "o1-2024-12-17";
/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-preview-2024-09-12` completion model
pub const O1_PREVIEW_2024_09_12: &str = "o1-preview-2024-09-12";
/// `o1-mini` completion model.
pub const O1_MINI: &str = "o1-mini";
/// `o1-mini-2024-09-12` completion model
pub const O1_MINI_2024_09_12: &str = "o1-mini-2024-09-12";

/// `gpt-4.1-mini` completion model
pub const GPT_4_1_MINI: &str = "gpt-4.1-mini";
/// `gpt-4.1-nano` completion model
pub const GPT_4_1_NANO: &str = "gpt-4.1-nano";
/// `gpt-4.1-2025-04-14` completion model
pub const GPT_4_1_2025_04_14: &str = "gpt-4.1-2025-04-14";
/// `gpt-4.1` completion model
pub const GPT_4_1: &str = "gpt-4.1";

#[derive(Default, Clone, Debug, PartialEq)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    /// Force the model to call one specific function:
    /// `{"type": "function", "function": {"name": "..."}}`.
    Function {
        name: String,
    },
}

#[derive(Deserialize, Serialize)]
struct ToolChoiceFunctionName {
    name: String,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ToolChoiceFunctionRepr {
    Function { function: ToolChoiceFunctionName },
}

impl Serialize for ToolChoice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function { name } => ToolChoiceFunctionRepr::Function {
                function: ToolChoiceFunctionName { name: name.clone() },
            }
            .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Mode(String),
            Function(ToolChoiceFunctionRepr),
        }

        match Repr::deserialize(deserializer)? {
            Repr::Mode(mode) => match mode.as_str() {
                "auto" => Ok(Self::Auto),
                "none" => Ok(Self::None),
                "required" => Ok(Self::Required),
                other => Err(serde::de::Error::custom(format!(
                    "unknown tool_choice mode {other:?}"
                ))),
            },
            Repr::Function(ToolChoiceFunctionRepr::Function {
                function: ToolChoiceFunctionName { name },
            }) => Ok(Self::Function { name }),
        }
    }
}

impl ToolChoice {
    /// Force a call to the named function.
    pub fn function(name: impl Into<String>) -> Self {
        Self::Function { name: name.into() }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default)]
pub struct PromptTokensDetails {
    /// Cached tokens from prompt caching
    #[serde(default)]
    pub cached_tokens: usize,
    /// Audio input tokens, defaulting null or missing values to zero.
    /// Zero is omitted from serialization. [`Usage::to_normalized`] uses the
    /// reported total to determine whether audio is additional to prompt tokens.
    #[serde(
        default,
        deserialize_with = "json_utils::null_or_default",
        skip_serializing_if = "is_zero"
    )]
    pub audio_tokens: usize,
    /// Tokens written to cache on this call. `None` means unreported, not zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<usize>,
}

/// Whether a counter is absent-as-zero, for `skip_serializing_if`.
fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default)]
pub struct CompletionTokensDetails {
    /// Reasoning tokens reported by reasoning-capable providers.
    #[serde(default)]
    pub reasoning_tokens: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<usize>,
    pub total_tokens: usize,
    // Not aliased to Mistral's singular `prompt_token_details`: Mistral's
    // embeddings reply carries *both* keys (the singular always `null`), and
    // an alias makes serde reject the document as a duplicate field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// Mistral's top-level cached-prompt count, reported beside (or instead
    /// of) `prompt_tokens_details.cached_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_cached_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_time: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_time: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_time: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_time: Option<f64>,
}

impl Usage {
    pub fn new() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: None,
            total_tokens: 0,
            prompt_tokens_details: None,
            completion_tokens_details: None,
            num_cached_tokens: None,
            queue_time: None,
            prompt_time: None,
            completion_time: None,
            total_time: None,
        }
    }
}

impl Default for Usage {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Usage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Usage {
            prompt_tokens,
            total_tokens,
            ..
        } = self;
        write!(
            f,
            "Prompt tokens: {prompt_tokens} Total tokens: {total_tokens}"
        )
    }
}

impl From<&Usage> for crate::completion::Usage {
    fn from(value: &Usage) -> crate::completion::Usage {
        value.to_normalized()
    }
}

impl From<Usage> for crate::completion::Usage {
    fn from(value: Usage) -> crate::completion::Usage {
        value.to_normalized()
    }
}

impl Usage {
    /// Return prompt tokens plus audio only when that sum and output match the total.
    /// Missing output counts are treated as zero for this comparison.
    fn input_tokens(&self) -> usize {
        let audio = self
            .prompt_tokens_details
            .map_or(0, |details| details.audio_tokens);
        let beside = self.prompt_tokens.saturating_add(audio);
        let accounted = beside.saturating_add(self.completion_tokens.unwrap_or(0));
        if audio != 0 && accounted == self.total_tokens {
            beside
        } else {
            self.prompt_tokens
        }
    }

    /// Normalize token accounting, deriving absent output counts from the total.
    /// Cached input prefers prompt details and falls back to `num_cached_tokens`.
    pub fn to_normalized(&self) -> crate::completion::Usage {
        let input_tokens = self.input_tokens();
        let details = self.prompt_tokens_details.as_ref();
        crate::completion::Usage {
            input_tokens: Some(input_tokens as u64),
            // Gateways that omit `completion_tokens` still send the total, so
            // the completion count is the remainder.
            output_tokens: Some(
                self.completion_tokens
                    .unwrap_or_else(|| self.total_tokens.saturating_sub(input_tokens))
                    as u64,
            ),
            total_tokens: Some(self.total_tokens as u64),
            cached_input_tokens: details
                .map(|d| d.cached_tokens as u64)
                .or(self.num_cached_tokens),
            cache_creation_input_tokens: details
                .and_then(|d| d.cache_write_tokens)
                .map(|tokens| tokens as u64),
            reasoning_tokens: self
                .completion_tokens_details
                .as_ref()
                .map(|d| d.reasoning_tokens as u64),
            ..Default::default()
        }
    }
}

/// Whether the model matches the GPT-5 through GPT-9 or numeric o-series rules
/// used to select `max_completion_tokens`.
pub(crate) fn is_openai_reasoning_model(model: &str) -> bool {
    /// Match a single-digit GPT major version at least `lowest`, allowing dot
    /// and hyphen suffixes. Multi-digit major versions do not match.
    fn is_numbered_gpt_family(model: &str, lowest: u32) -> bool {
        model
            .strip_prefix("gpt-")
            .and_then(|rest| rest.split(['.', '-']).next())
            .filter(|major| major.len() == 1)
            .and_then(|major| major.parse::<u32>().ok())
            .is_some_and(|major| major >= lowest)
    }

    /// Match `o` followed by a digit and then an end, hyphen, or another digit.
    fn is_o_series(model: &str) -> bool {
        let mut chars = model.chars();
        chars.next() == Some('o')
            && chars.next().is_some_and(|digit| digit.is_ascii_digit())
            && chars
                .next()
                .is_none_or(|next| next == '-' || next.is_ascii_digit())
    }

    is_numbered_gpt_family(model, 5) || is_o_series(model)
}

#[cfg(test)]
pub(crate) mod tests;
