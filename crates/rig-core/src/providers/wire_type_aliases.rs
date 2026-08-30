//! Compile-time guard that every provider's public wire-type alias still
//! names what that provider's driver actually returns.
//!
//! The aliases exist so users need not reach into the `#[doc(hidden)]`
//! compatible trees. They are type aliases, so pointing one at the wrong
//! target (a `StreamingCompletionResponse` carrying another provider's
//! `Usage`, say) compiles fine and silently publishes the wrong type; these
//! coercions fail instead.
#![allow(dead_code)]

#[cfg(feature = "azure")]
fn azure_aliases(
    raw: super::azure::CompletionResponse,
    terminal: super::azure::StreamingCompletionResponse,
) -> (
    <super::azure::AzureExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::azure::AzureExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "deepseek")]
fn deepseek_aliases(
    raw: super::deepseek::CompletionResponse,
    terminal: super::deepseek::StreamingCompletionResponse,
) -> (
    <super::deepseek::DeepSeekExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::deepseek::DeepSeekExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "doubleword")]
fn doubleword_aliases(
    raw: super::doubleword::CompletionResponse,
    terminal: super::doubleword::StreamingCompletionResponse,
) -> (
    <super::doubleword::client::DoublewordExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::doubleword::client::DoublewordExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "groq")]
fn groq_aliases(
    raw: super::groq::CompletionResponse,
    terminal: super::groq::StreamingCompletionResponse,
) -> (
    <super::groq::GroqExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::groq::GroqExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "huggingface")]
fn huggingface_aliases(
    raw: super::huggingface::CompletionResponse,
    terminal: super::huggingface::StreamingCompletionResponse,
) -> (
    <super::huggingface::client::HuggingFaceExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::huggingface::client::HuggingFaceExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "hyperbolic")]
fn hyperbolic_aliases(
    raw: super::hyperbolic::CompletionResponse,
    terminal: super::hyperbolic::StreamingCompletionResponse,
) -> (
    <super::hyperbolic::HyperbolicExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::hyperbolic::HyperbolicExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "llamacpp")]
fn llamacpp_aliases(
    raw: super::llamacpp::CompletionResponse,
    terminal: super::llamacpp::StreamingCompletionResponse,
) -> (
    <super::llamacpp::LlamacppExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::llamacpp::LlamacppExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "minimax")]
fn minimax_aliases(
    raw: super::minimax::CompletionResponse,
    terminal: super::minimax::StreamingCompletionResponse,
) -> (
    <super::minimax::MiniMaxExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::minimax::MiniMaxExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "mira")]
fn mira_aliases(
    raw: super::mira::CompletionResponse,
    terminal: super::mira::StreamingCompletionResponse,
) -> (
    <super::mira::MiraExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::mira::MiraExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "mistral")]
fn mistral_aliases(
    raw: super::mistral::CompletionResponse,
    terminal: super::mistral::StreamingCompletionResponse,
) -> (
    <super::mistral::MistralExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::mistral::MistralExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "moonshot")]
fn moonshot_aliases(
    raw: super::moonshot::CompletionResponse,
    terminal: super::moonshot::StreamingCompletionResponse,
) -> (
    <super::moonshot::MoonshotExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::moonshot::MoonshotExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "openrouter")]
fn openrouter_aliases(
    raw: super::openrouter::CompletionResponse,
    terminal: super::openrouter::StreamingCompletionResponse,
) -> (
    <super::openrouter::OpenRouterExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::openrouter::OpenRouterExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "perplexity")]
fn perplexity_aliases(
    raw: super::perplexity::CompletionResponse,
    terminal: super::perplexity::StreamingCompletionResponse,
) -> (
    <super::perplexity::PerplexityExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::perplexity::PerplexityExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "together")]
fn together_aliases(
    raw: super::together::CompletionResponse,
    terminal: super::together::StreamingCompletionResponse,
) -> (
    <super::together::client::TogetherExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::together::client::TogetherExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "venice")]
fn venice_aliases(
    raw: super::venice::CompletionResponse,
    terminal: super::venice::StreamingCompletionResponse,
) -> (
    <super::venice::VeniceExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::venice::VeniceExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "xiaomimimo")]
fn xiaomimimo_aliases(
    raw: super::xiaomimimo::CompletionResponse,
    terminal: super::xiaomimimo::StreamingCompletionResponse,
) -> (
    <super::xiaomimimo::XiaomiMimoExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::xiaomimimo::XiaomiMimoExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

#[cfg(feature = "zai")]
fn zai_aliases(
    raw: super::zai::CompletionResponse,
    terminal: super::zai::StreamingCompletionResponse,
) -> (
    <super::zai::ZAiExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::Response,
    super::openai_compatible::StreamingCompletionResponse<
        <super::zai::ZAiExt as crate::providers::openai_compatible::completion::OpenAICompatibleProvider>::StreamingUsage,
    >,
){
    (raw, terminal)
}

// The Anthropic-dialect aliases on the four dual-dialect providers.
#[cfg(any(
    feature = "minimax",
    feature = "moonshot",
    feature = "xiaomimimo",
    feature = "zai"
))]
mod anthropic_dialect {
    macro_rules! assert_anthropic_aliases {
        ($feature:literal, $provider:ident) => {
            #[cfg(feature = $feature)]
            fn $provider(
                raw: crate::providers::$provider::AnthropicCompletionResponse,
                terminal: crate::providers::$provider::AnthropicStreamingCompletionResponse,
            ) -> (
                crate::providers::anthropic_compatible::completion::CompletionResponse,
                crate::providers::anthropic_compatible::streaming::StreamingCompletionResponse,
            ) {
                (raw, terminal)
            }
        };
    }

    assert_anthropic_aliases!("minimax", minimax);
    assert_anthropic_aliases!("moonshot", moonshot);
    assert_anthropic_aliases!("xiaomimimo", xiaomimimo);
    assert_anthropic_aliases!("zai", zai);
}

// The Responses-dialect aliases.
#[cfg(feature = "chatgpt")]
fn chatgpt_aliases(
    raw: super::chatgpt::CompletionResponse,
    terminal: super::chatgpt::StreamingCompletionResponse,
) -> (
    super::openai_compatible::responses_api::CompletionResponse,
    super::openai_compatible::responses_api::streaming::StreamingCompletionResponse,
) {
    (raw, terminal)
}

// Copilot wraps both routes' payloads in its own enums; the aliases are what
// a caller needs to spell a variant's contents.
#[cfg(feature = "copilot")]
fn copilot_aliases(
    chat: super::copilot::ChatCompletionResponse,
    responses: super::copilot::ResponsesCompletionResponse,
    chat_terminal: super::copilot::ChatStreamingCompletionResponse,
    responses_terminal: super::copilot::ResponsesStreamingCompletionResponse,
) -> (
    super::copilot::CopilotCompletionResponse,
    super::copilot::CopilotStreamingResponse,
) {
    let _ = super::copilot::CopilotCompletionResponse::Responses(Box::new(responses));
    let _ = super::copilot::CopilotStreamingResponse::Responses(responses_terminal);
    (
        super::copilot::CopilotCompletionResponse::Chat(Box::new(chat)),
        super::copilot::CopilotStreamingResponse::Chat(chat_terminal),
    )
}

// The shared transcription types the audio-capable providers re-export.
#[cfg(all(feature = "audio", any(feature = "azure", feature = "groq")))]
fn transcription_aliases() {
    #[cfg(feature = "azure")]
    fn azure(
        r: super::azure::TranscriptionResponse,
    ) -> super::internal::transcription::TranscriptionResponse {
        r
    }
    #[cfg(feature = "groq")]
    fn groq(
        r: super::groq::TranscriptionResponse,
    ) -> super::internal::transcription::TranscriptionResponse {
        r
    }
}
