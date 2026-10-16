//! Histories built to break handle round-trips: a long reasoning
//! ciphertext, a signature on a reasoning block with no text, and a history
//! carried across three providers and back to the first.
//!
//! Each case continues live and checks both the provider's answer and the
//! recorded request: every handle must sit in its own slot, beside the value
//! it belongs to.

use serde_json::Value;

use rig_core::completion::{CompletionRequest, CompletionResponse, ToolDefinition};
use rig_core::message::{AssistantContent, CallId, Message, ToolResultContent, UserContent};
use rig_core::providers::anthropic::wire::AnthropicConfig;
use rig_core::providers::gemini::GeminiConfig;
use rig_core::providers::gemini::completion::GenerateContent;
use rig_core::providers::openai::wire::OpenAIConfig;

use super::portability::{FOLLOW_UP, Source, decode_whole_reply};
use super::{Dialect, Params, lost_tokens, response_tokens};

/// The code each record resolves to.
pub const CODES: [(&str, &str); 2] = [("alpha", "amber-5521"), ("beta", "cobalt-7730")];

fn code(record: &str) -> &'static str {
    CODES
        .iter()
        .find(|(name, _)| *name == record)
        .map_or("unknown-record", |(_, code)| code)
}

fn lookup() -> ToolDefinition {
    ToolDefinition {
        name: rig_core::message::ToolName::new("lookup_code").expect("tool name"),
        description: "Return the code stored for one record.".to_owned(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "record": { "type": "string" } },
            "required": ["record"]
        }),
        annotations: Default::default(),
    }
}

/// A request over `history` with the lookup tool and `params`.
pub fn request(
    history: Vec<Message>,
    params: impl Into<Params>,
    max_tokens: u64,
) -> CompletionRequest {
    let mut request = CompletionRequest::from(history);
    request.tools = vec![lookup().into()];
    request.max_tokens = Some(max_tokens);
    params.into().apply(&mut request);
    request
}

fn text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

fn result(call: CallId, record: &str) -> UserContent {
    UserContent::tool_result(
        call,
        rig_core::message::ToolName::new("lookup_code").expect("tool name"),
        vec![ToolResultContent::text(format!(
            "record {record}: code {}",
            code(record)
        ))],
    )
}

fn assistant(reply: &CompletionResponse) -> Message {
    Message::Assistant(reply.head().with_content(reply.choice.clone()))
}

/// The continuation after `turn` loses nothing turn one delivered, and
/// turn one delivered `kind` at least `min_len` bytes long.
pub fn assert_carried(provider: &str, scenario: &str, kind: &str, min_len: usize) {
    let paths = crate::cassettes::recorded_request_paths(provider, scenario);
    let bodies = crate::cassettes::recorded_interaction_bodies(provider, scenario);
    let dialect = Dialect::from_path(&paths[0]);
    let next: Value = serde_json::from_str(&bodies[1].0).expect("the continuation is JSON");
    let lost = lost_tokens(dialect, &bodies[0].1, &next);
    assert!(lost.is_empty(), "[{provider}] lost: {lost:?}");
    let delivered = response_tokens(dialect, &bodies[0].1);
    let longest = delivered
        .iter()
        .filter(|token| token.kind == kind)
        .map(|token| token.value.len())
        .max()
        .unwrap_or_default();
    assert!(
        longest >= min_len,
        "[{provider}] turn one delivered a {kind} of at least {min_len} bytes, the longest was {longest}"
    );
}

async fn complete(
    model: &rig_core::DynModel<rig_core::operation::Completion>,
    request: CompletionRequest,
    streamed: bool,
) -> Result<CompletionResponse, rig_core::error::ProviderError> {
    if !streamed {
        return model.call(request).await;
    }
    use futures::StreamExt;
    let mut stream = model.stream(request)?;
    while let Some(item) = stream.next().await {
        if let Err(error) = item {
            return Err(rig_core::error::ProviderError::Response(error.to_string()));
        }
    }
    stream.finish().await
}

/// A reasoning turn with a tool call, streamed or not, answered and
/// continued. Returns the first reply so a cell can check the shape of what
/// was delivered.
pub async fn reasoning_round_trip(
    model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    prompt: &str,
    params: impl Into<Params>,
    max_tokens: u64,
    streamed: bool,
) -> CompletionResponse {
    let params = params.into();
    let model: rig_core::DynModel<rig_core::operation::Completion> = model.into();
    let prompt = Message::user(prompt);
    let first = complete(
        &model,
        request(vec![prompt.clone()], params.clone(), max_tokens),
        streamed,
    )
    .await
    .expect("turn one");
    let call = first
        .choice
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("turn one calls lookup_code: {:?}", first.choice));
    let record = call.function.arguments["record"]
        .as_str()
        .unwrap_or("alpha");
    let history = vec![
        prompt,
        assistant(&first),
        Message::User {
            content: vec![result(call.id.clone(), record)],
        },
    ];
    let reply = complete(&model, request(history, params, max_tokens), streamed)
        .await
        .expect("the provider accepts the replayed reasoning");
    assert!(
        text(&reply.choice).contains(code(record)),
        "the answer uses the result: {:?}",
        reply.choice
    );
    first
}

/// Whether a reply holds a signed reasoning block with no text.
pub fn has_empty_signed_reasoning(reply: &CompletionResponse) -> bool {
    reply.choice.iter().any(|content| match content {
        AssistantContent::Reasoning(reasoning) => {
            reasoning.text.is_empty() && reasoning.native.is_some()
        }
        _ => false,
    })
}

/// The three foreign hops of the round trip and the home hop that closes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hop {
    /// Continue the Anthropic source on OpenAI Responses.
    OpenAiResponses,
    /// Continue that on Gemini.
    Gemini,
    /// Return to Anthropic.
    Anthropic,
}

/// The round trip's scenario in each hop's cassette directory.
pub const ROUND_TRIP: &str = "adversarial/three_provider_round_trip";

const HOPS: [Hop; 3] = [Hop::OpenAiResponses, Hop::Gemini, Hop::Anthropic];

impl Hop {
    /// The hop's cassette directory.
    pub fn provider(self) -> &'static str {
        match self {
            Self::OpenAiResponses => "openai",
            Self::Gemini => "gemini",
            Self::Anthropic => "anthropic",
        }
    }

    fn prompt(self) -> &'static str {
        match self {
            Self::OpenAiResponses => FOLLOW_UP,
            Self::Gemini => "Restate that advice in different words, in one sentence.",
            Self::Anthropic => "Summarize the whole conversation in one sentence.",
        }
    }

    fn reply(self) -> CompletionResponse {
        let body = crate::cassettes::recorded_interaction_bodies(self.provider(), ROUND_TRIP)
            .into_iter()
            .next()
            .map(|(_, response)| response)
            .unwrap_or_else(|| panic!("{} records its hop", self.provider()));
        let decoded = match self {
            Self::OpenAiResponses => decode_whole_reply(
                &OpenAIConfig::new("decode-only")
                    .connect(crate::cassettes::local_http())
                    .responses("gpt-5-mini")
                    .wire,
                &body,
            ),
            Self::Gemini => decode_whole_reply(
                &GenerateContent::new(GeminiConfig::new("decode-only"), "gemini-3-flash-preview"),
                &body,
            ),
            Self::Anthropic => decode_whole_reply(
                &AnthropicConfig::new("decode-only")
                    .connect(crate::cassettes::local_http())
                    .completion("claude-sonnet-4-6")
                    .wire,
                &body,
            ),
        };
        decoded.unwrap_or_else(|error| panic!("{} hop decodes: {error}", self.provider()))
    }

    /// The history this hop continues: the Anthropic source with its tool
    /// result, then every earlier hop's prompt and decoded reply, then this
    /// hop's prompt.
    pub fn history(self) -> Vec<Message> {
        let mut history = Source::Anthropic.history();
        for hop in HOPS.into_iter().take_while(|hop| *hop != self) {
            history.push(Message::user(hop.prompt()));
            history.push(assistant(&hop.reply()));
        }
        history.push(Message::user(self.prompt()));
        history
    }
}

/// Send one hop of the round trip.
pub async fn round_trip_hop(
    model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    hop: Hop,
    params: impl Into<Params>,
) {
    let model: rig_core::DynModel<rig_core::operation::Completion> = model.into();
    let reply = model
        .call(request(hop.history(), params, 4096))
        .await
        .unwrap_or_else(|error| panic!("{hop:?} accepts the carried history: {error}"));
    assert!(!text(&reply.choice).trim().is_empty(), "{hop:?} answers");
}

/// The Anthropic signature reaches no foreign hop and returns to Anthropic
/// in its thinking block, byte for byte; the OpenAI ciphertext reaches
/// neither later hop.
pub fn assert_round_trip_recorded(hop: Hop) {
    let sent = |hop: Hop| {
        crate::cassettes::recorded_interaction_bodies(hop.provider(), ROUND_TRIP)
            .into_iter()
            .next()
            .map(|(request, _)| request)
            .unwrap_or_else(|| panic!("{} records its hop", hop.provider()))
    };
    let source = crate::cassettes::recorded_interaction_bodies(
        "anthropic",
        super::portability::SOURCE_SCENARIO,
    )
    .into_iter()
    .next()
    .map(|(_, response)| response)
    .expect("the Anthropic source reply");
    let signatures: Vec<_> = response_tokens(Dialect::AnthropicMessages, &source)
        .into_iter()
        .filter(|token| token.kind == "signature")
        .collect();
    assert!(!signatures.is_empty(), "the source turn is signed");
    let request = sent(hop);
    match hop {
        Hop::Anthropic => {
            let next: Value = serde_json::from_str(&request).expect("the home request is JSON");
            let lost = lost_tokens(Dialect::AnthropicMessages, &source, &next);
            assert!(
                lost.is_empty(),
                "the source turn returns home intact: {lost:?}"
            );
        }
        foreign => {
            for signature in &signatures {
                assert!(
                    !request.contains(&signature.value),
                    "{foreign:?} receives no Anthropic signature"
                );
            }
        }
    }
    if hop != Hop::OpenAiResponses {
        let openai = crate::cassettes::recorded_interaction_bodies("openai", ROUND_TRIP)
            .into_iter()
            .next()
            .map(|(_, response)| response)
            .expect("the OpenAI hop reply");
        for token in response_tokens(Dialect::OpenAiResponses, &openai)
            .into_iter()
            .filter(|token| token.kind == "encrypted_content")
        {
            assert!(
                !request.contains(&token.value),
                "{hop:?} receives no OpenAI ciphertext"
            );
        }
    }
}
