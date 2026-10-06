//! OpenAI's completion model identifiers.
//!
//! ```
//! use rig_core::providers::openai::completion::GPT_4O;
//! assert_eq!(GPT_4O, "gpt-4o");
//! ```

/// GPT-6 Astra, API ID `gpt-6-astra`: a reasoning model that cannot turn
/// reasoning off. Its catalog entry holds what the encoders enforce: no
/// sampling parameters, and tools only through Responses.
pub const GPT_6_ASTRA: &str = "gpt-6-astra";

/// GPT-6.1 Sol, API ID `gpt-6.1-sol`: a reasoning model that cannot turn
/// reasoning off. Its catalog entry holds what the encoders enforce: no
/// sampling parameters, and tools only through Responses.
pub const GPT_6_1_SOL: &str = "gpt-6.1-sol";

/// GPT-6 Sol, API ID `gpt-6-sol`: a reasoning model. Its catalog entry
/// holds what the encoders enforce: sampling parameters, and Chat
/// Completions tools, only at effort `none`
/// ([`GenerationOptions::reasoning`](crate::completion::GenerationOptions::reasoning)
/// `(Reasoning::Off)`). Responses takes tools at any effort.
pub const GPT_6_SOL: &str = "gpt-6-sol";

/// GPT-6 Luna, API ID `gpt-6-luna`: a reasoning model. Its catalog entry
/// holds what the encoders enforce: sampling parameters, and Chat
/// Completions tools, only at effort `none`
/// ([`GenerationOptions::reasoning`](crate::completion::GenerationOptions::reasoning)
/// `(Reasoning::Off)`). Responses takes tools at any effort.
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

#[cfg(test)]
pub(crate) mod tests;
