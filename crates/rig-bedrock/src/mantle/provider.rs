//! Mantle-specific Rig provider extensions (OpenAI wire format, Bedrock identity).
//!
//! These are **not** type aliases of the OpenAI provider. Using
//! [`ResponsesClient::builder`] / [`CompletionsClient::builder`] defaults to a
//! Mantle base URL, not `api.openai.com`, so a Bedrock bearer is never sent to
//! OpenAI by accident. Capabilities are limited to chat completion; embeddings,
//! image, transcription, and other OpenAI surfaces are [`Nothing`].

use rig_core::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
};
use rig_core::http_client::{self, HttpClientExt};
use rig_core::providers::openai::completion::OpenAICompatibleProvider;
use rig_core::providers::openai::responses_api::{
    GenericResponsesCompletionModel, ResponsesProviderExt, SystemInstructionsPlacement,
};
use rig_core::providers::openai::{self, CompletionResponse, Usage};

/// GenAI / telemetry provider name shared with Converse Bedrock spans.
pub const PROVIDER_NAME: &str = "aws_bedrock";

/// Default Mantle base URL when no region-specific override is supplied.
///
/// Prefer [`super::openai_base_url`] (or [`super::openai_gpt5_base_url`]) with an
/// explicit region when building production clients.
pub const DEFAULT_MANTLE_BASE_URL: &str = "https://bedrock-mantle.us-east-1.api.aws/v1";

// ---------------------------------------------------------------------------
// Responses (default Mantle surface)
// ---------------------------------------------------------------------------

/// Provider extension for Mantle Responses API clients.
#[derive(Debug, Default, Clone, Copy)]
pub struct MantleResponsesExt {
    system_instructions_placement: SystemInstructionsPlacement,
}

/// Builder for [`MantleResponsesExt`].
#[derive(Debug, Default, Clone, Copy)]
pub struct MantleResponsesBuilder;

/// Mantle Responses client (OpenAI wire format, Bedrock defaults).
pub type ResponsesClient<H = reqwest::Client> = client::Client<MantleResponsesExt, H>;
/// Builder for [`ResponsesClient`].
pub type ResponsesClientBuilder<H = rig_core::markers::Missing> =
    client::ClientBuilder<MantleResponsesBuilder, BearerAuth, H>;

impl Provider for MantleResponsesExt {
    type Builder = MantleResponsesBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl ResponsesProviderExt for MantleResponsesExt {
    const PROVIDER_NAME: &'static str = PROVIDER_NAME;

    fn system_instructions_placement(&self) -> SystemInstructionsPlacement {
        self.system_instructions_placement
    }
}

impl<H> Capabilities<H> for MantleResponsesExt {
    type Completion = Capable<GenericResponsesCompletionModel<MantleResponsesExt, H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    // Enabled on our rig-core dep so these associated types always exist; Mantle
    // does not implement them.
    type ImageGeneration = Nothing;
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for MantleResponsesExt {}

impl ProviderBuilder for MantleResponsesBuilder {
    type Extension<H>
        = MantleResponsesExt
    where
        H: HttpClientExt;
    type ApiKey = BearerAuth;

    const BASE_URL: &'static str = DEFAULT_MANTLE_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(MantleResponsesExt::default())
    }
}

// ---------------------------------------------------------------------------
// Completions
// ---------------------------------------------------------------------------

/// Provider extension for Mantle Chat Completions clients.
#[derive(Debug, Default, Clone, Copy)]
pub struct MantleCompletionsExt;

/// Builder for [`MantleCompletionsExt`].
#[derive(Debug, Default, Clone, Copy)]
pub struct MantleCompletionsBuilder;

/// Mantle Chat Completions client (OpenAI wire format, Bedrock defaults).
pub type CompletionsClient<H = reqwest::Client> = client::Client<MantleCompletionsExt, H>;
/// Builder for [`CompletionsClient`].
pub type CompletionsClientBuilder<H = rig_core::markers::Missing> =
    client::ClientBuilder<MantleCompletionsBuilder, BearerAuth, H>;

impl Provider for MantleCompletionsExt {
    type Builder = MantleCompletionsBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl<H> Capabilities<H> for MantleCompletionsExt {
    type Completion = Capable<openai::completion::GenericCompletionModel<MantleCompletionsExt, H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    // Enabled on our rig-core dep so these associated types always exist; Mantle
    // does not implement them.
    type ImageGeneration = Nothing;
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for MantleCompletionsExt {}

impl ProviderBuilder for MantleCompletionsBuilder {
    type Extension<H>
        = MantleCompletionsExt
    where
        H: HttpClientExt;
    type ApiKey = BearerAuth;

    const BASE_URL: &'static str = DEFAULT_MANTLE_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(MantleCompletionsExt)
    }
}

impl OpenAICompatibleProvider for MantleCompletionsExt {
    const PROVIDER_NAME: &'static str = PROVIDER_NAME;

    type StreamingUsage = Usage;
    type Response = CompletionResponse;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::client::ProviderBuilder;
    use rig_core::providers::openai::completion::OpenAICompatibleProvider;
    use rig_core::providers::openai::responses_api::ResponsesProviderExt;

    #[test]
    fn responses_defaults_to_mantle_not_openai() {
        assert_eq!(
            MantleResponsesBuilder::BASE_URL,
            "https://bedrock-mantle.us-east-1.api.aws/v1"
        );
        assert_ne!(MantleResponsesBuilder::BASE_URL, "https://api.openai.com/v1");
        assert_eq!(
            <MantleResponsesExt as ResponsesProviderExt>::PROVIDER_NAME,
            "aws_bedrock"
        );
    }

    #[test]
    fn completions_provider_name_is_bedrock() {
        assert_eq!(
            <MantleCompletionsExt as OpenAICompatibleProvider>::PROVIDER_NAME,
            "aws_bedrock"
        );
        assert_eq!(
            MantleCompletionsBuilder::BASE_URL,
            "https://bedrock-mantle.us-east-1.api.aws/v1"
        );
    }
}
