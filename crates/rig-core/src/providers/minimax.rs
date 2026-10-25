//! MiniMax's endpoints and model identifiers.
//!
//! Global and China configurations support chat-completions and Messages wires,
//! using `MINIMAX_API_KEY`. [`from_env`] and [`new`] build a client on the
//! chat-completions [`MINIMAX`](crate::providers::openai::wire::MINIMAX)
//! dialect; [`anthropic_from_env`] and [`anthropic_new`] build a Messages-format
//! client. The chat-completions client also generates images and speech, and
//! `voice_clone` clones a voice from uploaded audio, and `.responses(model)`
//! speaks MiniMax's Responses API on either endpoint.
//!
//! ```no_run
//! use rig_core::providers::minimax;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = minimax::from_env()?.chat(minimax::MINIMAX_M3);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

/// Global OpenAI-compatible base URL.
pub const GLOBAL_API_BASE_URL: &str = "https://api.minimax.io/v1";
/// China OpenAI-compatible base URL.
pub const CHINA_API_BASE_URL: &str = "https://api.minimax.cn/v1";
/// Global Anthropic-compatible base URL.
pub const GLOBAL_ANTHROPIC_API_BASE_URL: &str = "https://api.minimax.io/anthropic";
/// China Anthropic-compatible base URL.
pub const CHINA_ANTHROPIC_API_BASE_URL: &str = "https://api.minimax.cn/anthropic";

/// `MiniMax-M3`
pub const MINIMAX_M3: &str = "MiniMax-M3";
/// `MiniMax-M2.7`
pub const MINIMAX_M2_7: &str = "MiniMax-M2.7";
/// `MiniMax-M2.7-highspeed`
pub const MINIMAX_M2_7_HIGHSPEED: &str = "MiniMax-M2.7-highspeed";
/// `MiniMax-M2.5`
pub const MINIMAX_M2_5: &str = "MiniMax-M2.5";
/// `MiniMax-M2.5-highspeed`
pub const MINIMAX_M2_5_HIGHSPEED: &str = "MiniMax-M2.5-highspeed";

/// The MiniMax model families whose options differ on every wire.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    M31Flash,
    M3,
    Other,
}

/// `model`'s family, matched in any case; `minimax-m3.1` wins over `minimax-m3`.
pub(crate) fn family(model: &str) -> Family {
    let model = model.to_ascii_lowercase();
    match model.as_str() {
        id if id.starts_with("minimax-m3.1") => Family::M31Flash,
        id if id.starts_with("minimax-m3") => Family::M3,
        _ => Family::Other,
    }
}

/// Whether MiniMax `model` reads images: M2 models are text only, and M3
/// reads images. The name is matched in any case, as gateways lowercase it.
/// Every wire MiniMax serves applies this rule to an id the catalog does not
/// list.
pub(crate) fn reads_images(model: &str) -> bool {
    !model.to_ascii_lowercase().starts_with("minimax-m2")
}

#[cfg(feature = "audio")]
pub mod audio_generation;
pub mod extension;
#[cfg(feature = "image")]
pub mod image_generation;
#[cfg(feature = "audio")]
pub mod voice_clone;

#[cfg(feature = "audio")]
pub use audio_generation::*;
#[cfg(feature = "image")]
pub use image_generation::*;
#[cfg(feature = "audio")]
pub use voice_clone::voice_cloning;

/// The provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "minimax";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::MINIMAX, "MiniMax");
crate::client::macros::anthropic_vendor!(
    crate::providers::anthropic::wire::MINIMAX,
    "MiniMax",
    anthropic_from_env,
    anthropic_new
);

/// Decode a MiniMax media reply once its `base_resp` status reports success.
/// MiniMax can answer a failure with HTTP 200, so a nonzero `status_code` is
/// the body verbatim as [`ProviderError::ProviderResponse`].
#[cfg(any(feature = "image", feature = "audio"))]
pub(crate) fn decode_reply<T: serde::de::DeserializeOwned>(
    body: &[u8],
) -> Result<T, crate::error::ProviderError> {
    #[derive(serde::Deserialize)]
    struct Status {
        #[serde(default)]
        base_resp: Option<BaseResp>,
    }
    #[derive(serde::Deserialize)]
    struct BaseResp {
        #[serde(default)]
        status_code: i64,
    }

    let status: Status = serde_json::from_slice(body)?;
    if status
        .base_resp
        .is_some_and(|status| status.status_code != 0)
    {
        return Err(crate::error::ProviderError::from_provider_body(
            String::from_utf8_lossy(body),
        ));
    }
    Ok(serde_json::from_slice(body)?)
}

#[cfg(all(test, any(feature = "image", feature = "audio")))]
mod tests;

