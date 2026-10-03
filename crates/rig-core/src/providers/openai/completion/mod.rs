//! Chat Completions model identifiers, the tool choice the OpenAI wires
//! share, and the usage counters of OpenAI's replies.
//!
//! ```
//! use rig_core::providers::openai::completion::{GPT_4O, ToolChoice};
//! assert_eq!(serde_json::to_value(ToolChoice::Required)?, "required");
//! # let _ = GPT_4O;
//! # Ok::<(), serde_json::Error>(())
//! ```

use serde::{Deserialize, Serialize, Serializer};

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
