//! The Bedrock Converse completion wire and model identifiers.
//! Model availability and inference-profile support depend on the AWS region.
//!
//! A request is built as Converse JSON and a reply is read as the JSON
//! Bedrock sent. The AWS SDK signs and transports both; its typed request
//! and response shapes play no part.
//!
//! ```no_run
//! use rig_bedrock::{client::BedrockRuntime, completion::{AMAZON_NOVA_LITE, Converse}};
//! use rig_core::Model;
//!
//! let model = BedrockRuntime::from_env().completion(AMAZON_NOVA_LITE);
//! # let _ = model;
//! ```

use aws_sdk_bedrockruntime::config::http::HttpResponse;
use aws_sdk_bedrockruntime::error::{ProvideErrorMetadata, SdkError};
use aws_sdk_bedrockruntime::operation::RequestId;
use rig_core::catalog::{Catalog, ModelSpec};
use rig_core::completion::options::FinalBody;
use rig_core::completion::{Accepts, CompletionRequest, Media, Pairing, ReplayTarget};
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::json_utils::Lenient;
use rig_core::message::{Api, DocumentSourceKind, Origin, ToolChoice};
use rig_core::operation::Completion;
use rig_core::providers::registry::ProviderId;
use rig_core::wire::{Descriptor, Mode, Wire};
use serde_json::Value;

use crate::capture::{self, Capture, Events};
use crate::client::BedrockRuntime;
use crate::request;
use crate::streaming::StreamState;
use crate::types::errors::sdk_error;

/// Stable descriptor name reported on normalized Bedrock responses.
pub const PROVIDER_NAME: &str = "aws_bedrock";

// Profile identifiers with a us. prefix route inference within the US region
// family; callers elsewhere must select a supported regional profile.

// Profile identifiers with a us. prefix route inference within the US region
// family; callers elsewhere must select a supported regional profile.

/// `amazon.nova-lite-v1:0`
pub const AMAZON_NOVA_LITE: &str = "amazon.nova-lite-v1:0";
/// `amazon.nova-micro-v1:0`
pub const AMAZON_NOVA_MICRO: &str = "amazon.nova-micro-v1:0";
/// `amazon.nova-pro-v1:0`
pub const AMAZON_NOVA_PRO: &str = "amazon.nova-pro-v1:0";

/// `us.anthropic.claude-haiku-4-5-20251001-v1:0` (cross-region profile)
pub const ANTHROPIC_CLAUDE_HAIKU_4_5: &str = "us.anthropic.claude-haiku-4-5-20251001-v1:0";
/// `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (cross-region profile)
pub const ANTHROPIC_CLAUDE_SONNET_4_5: &str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0";
/// `us.anthropic.claude-opus-4-5-20251101-v1:0` (cross-region profile)
pub const ANTHROPIC_CLAUDE_OPUS_4_5: &str = "us.anthropic.claude-opus-4-5-20251101-v1:0";
/// `us.anthropic.claude-sonnet-4-6` (cross-region profile)
pub const ANTHROPIC_CLAUDE_SONNET_4_6: &str = "us.anthropic.claude-sonnet-4-6";
/// `us.anthropic.claude-sonnet-5` (cross-region profile)
pub const ANTHROPIC_CLAUDE_SONNET_5: &str = "us.anthropic.claude-sonnet-5";
/// `us.anthropic.claude-opus-5` (cross-region profile)
pub const ANTHROPIC_CLAUDE_OPUS_5: &str = "us.anthropic.claude-opus-5";

/// `us.deepseek.r1-v1:0` (cross-region profile)
pub const DEEPSEEK_R1: &str = "us.deepseek.r1-v1:0";

/// `meta.llama3-8b-instruct-v1:0`
pub const LLAMA_3_8B_INSTRUCT: &str = "meta.llama3-8b-instruct-v1:0";
/// `meta.llama3-70b-instruct-v1:0`
pub const LLAMA_3_70B_INSTRUCT: &str = "meta.llama3-70b-instruct-v1:0";
/// `meta.llama3-1-8b-instruct-v1:0`
pub const LLAMA_3_1_8B_INSTRUCT: &str = "meta.llama3-1-8b-instruct-v1:0";
/// `meta.llama3-1-70b-instruct-v1:0`
pub const LLAMA_3_1_70B_INSTRUCT: &str = "meta.llama3-1-70b-instruct-v1:0";
/// `us.meta.llama3-3-70b-instruct-v1:0` (cross-region profile)
pub const META_LLAMA_3_3_70B_INSTRUCT: &str = "us.meta.llama3-3-70b-instruct-v1:0";
/// `us.meta.llama4-maverick-17b-instruct-v1:0` (cross-region profile)
pub const META_LLAMA_4_MAVERICK_17B_INSTRUCT: &str = "us.meta.llama4-maverick-17b-instruct-v1:0";
/// `us.meta.llama4-scout-17b-instruct-v1:0` (cross-region profile)
pub const META_LLAMA_4_SCOUT_17B_INSTRUCT: &str = "us.meta.llama4-scout-17b-instruct-v1:0";

/// `mistral.mistral-7b-instruct-v0:2`
pub const MISTRAL_7B_INSTRUCT: &str = "mistral.mistral-7b-instruct-v0:2";
/// `mistral.mistral-large-2402-v1:0`
pub const MISTRAL_LARGE_24_02: &str = "mistral.mistral-large-2402-v1:0";
/// `mistral.mistral-small-2402-v1:0`
pub const MISTRAL_SMALL_24_02: &str = "mistral.mistral-small-2402-v1:0";
/// `mistral.mixtral-8x7b-instruct-v0:1`
pub const MISTRAL_MIXTRAL_8X7B_INSTRUCT_V0: &str = "mistral.mixtral-8x7b-instruct-v0:1";
/// `us.mistral.pixtral-large-2502-v1:0` (cross-region profile)
pub const MISTRAL_PIXTRAL_LARGE_2502: &str = "us.mistral.pixtral-large-2502-v1:0";

/// `us.writer.palmyra-x4-v1:0` (cross-region profile)
pub const WRITER_PALMYRA_X4: &str = "us.writer.palmyra-x4-v1:0";
/// `us.writer.palmyra-x5-v1:0` (cross-region profile)
pub const WRITER_PALMYRA_X5: &str = "us.writer.palmyra-x5-v1:0";

/// The model family behind a Converse model id. It decides what history a
/// model reads back: Claude reads reasoning signatures, rejects unsigned
/// reasoning, and reads images; Nova reads S3 objects and video; both read
/// a tool result's status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Anthropic Claude.
    Claude,
    /// Amazon Nova.
    Nova,
    /// Any other model.
    Other,
}

impl Family {
    /// The family of the provider a Bedrock model id names. A base model id
    /// or a system inference profile is `[geography.]provider.model`, also
    /// the last part of a foundation-model or inference-profile ARN. An
    /// application inference profile or provisioned model ARN names no
    /// provider, so it is [`Family::Other`] unless the caller states its
    /// family with [`Converse::with_family`].
    ///
    /// ```
    /// use rig_bedrock::completion::{AMAZON_NOVA_PRO, ANTHROPIC_CLAUDE_SONNET_4_5, Family};
    ///
    /// assert_eq!(Family::of(ANTHROPIC_CLAUDE_SONNET_4_5), Family::Claude);
    /// assert_eq!(Family::of(AMAZON_NOVA_PRO), Family::Nova);
    /// assert_eq!(
    ///     Family::of("arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3"),
    ///     Family::Other
    /// );
    /// ```
    pub fn of(model: &str) -> Self {
        let id = model.rsplit('/').next().unwrap_or(model);
        let mut parts = id.rsplit('.');
        match (parts.next(), parts.next()) {
            (_, Some("anthropic")) => Self::Claude,
            (Some(name), Some("amazon")) if name.starts_with("nova") => Self::Nova,
            _ => Self::Other,
        }
    }
}

/// The Converse endpoint for one model: `Converse` for a unary call,
/// `ConverseStream` for a streamed one.
#[derive(Clone, Debug)]
pub struct Converse {
    pub model: String,
    /// The family of `model` when the caller states it, which an
    /// application inference profile ARN needs: its id names no provider.
    /// Set through [`Converse::with_family`]; otherwise [`Family::of`]
    /// the model id decides.
    pub family: Option<Family>,
}

impl Converse {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            family: None,
        }
    }

    /// State the family of this wire's model, for a model id that names no
    /// provider, such as an application inference profile ARN.
    ///
    /// ```
    /// use rig_bedrock::completion::{Converse, Family};
    ///
    /// let profile = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3";
    /// let wire = Converse::new(profile).with_family(Family::Claude);
    /// assert_eq!(wire.family(profile), Family::Claude);
    /// ```
    pub fn with_family(mut self, family: Family) -> Self {
        self.family = Some(family);
        self
    }

    /// The family of `model`: the stated one for this wire's own model,
    /// otherwise [`Family::of`] its id.
    pub fn family(&self, model: &str) -> Family {
        match self.family {
            Some(family) if model == self.model => family,
            _ => Family::of(model),
        }
    }
}

/// One Converse request: the model it addresses and its JSON body.
#[derive(Clone, Debug)]
pub struct ConverseRequest {
    pub model: String,
    pub body: FinalBody,
}

/// One unit of a Converse reply, as the JSON Bedrock sent.
#[derive(Clone, Debug)]
pub enum ConverseFrame {
    /// A whole unary reply. It is also the response's `raw`.
    Whole(Value),
    /// One stream event or in-band exception, as `{"<type>": <payload>}`.
    Event(Value),
}

impl Wire for Converse {
    type Op = Completion;
    type Payload = ConverseRequest;
    type Frame = ConverseFrame;
    type Decoder<'id> = StreamState;
    type Reassembler = crate::streaming::document::TerminalRecord;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .replay(self)
    }

    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<ConverseRequest, EncodeError> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let body = request::body(self, &request, &model)?;
        Ok(ConverseRequest { model, body })
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        StreamState::default()
    }
}

/// Model families pi's catalog lists as text-only on Bedrock, separated by
/// spaces: the rule for a model the catalog does not list under Bedrock (a
/// region inference profile, a dated or `-v2` revision).
const TEXT_ONLY: &str = "amazon.nova-micro deepseek. meta.llama3-8b meta.llama3-70b \
    meta.llama3-1- meta.llama3-3- minimax. mistral.devstral mistral.mistral-7b \
    mistral.mistral-large-2402 mistral.mistral-small-2402 mistral.mixtral mistral.voxtral \
    moonshot.kimi-k2-thinking nvidia.nemotron-nano-3 nvidia.nemotron-nano-9b \
    nvidia.nemotron-super openai.gpt-oss qwen.qwen3-2 qwen.qwen3-3 qwen.qwen3-coder \
    qwen.qwen3-next writer.palmyra zai.glm";

/// The catalog entry of the Bedrock `model`. A Claude id is read as every
/// wire that serves Claude reads it ([`claude_spec`]: the Anthropic model's
/// entry, past a region prefix, a `-v1:N` revision or a dated snapshot), so
/// its reasoning, sampling and context binding match Anthropic's own API.
/// Any other id is a base model id or system inference profile, or the last
/// part of its ARN, listed under Bedrock. `None` for a model the catalog
/// does not list.
///
/// [`claude_spec`]: rig_core::providers::anthropic::completion::claude_spec
pub fn spec(model: &str) -> Option<&'static ModelSpec> {
    let id = model.rsplit('/').next().unwrap_or(model);
    if id.contains("anthropic.") || id.starts_with("claude") {
        return rig_core::providers::anthropic::completion::claude_spec(id);
    }
    ProviderId::catalog(PROVIDER_NAME).and_then(|provider| Catalog::builtin().get(provider, id))
}

impl ReplayTarget for Converse {
    /// Section 6.5 of the typed-options design: reasoning in the model's
    /// own request fields, by family and Claude class.
    fn map_options(
        &self,
        request: &CompletionRequest,
        fields: rig_core::completion::options::OptionFields<'_>,
    ) -> rig_core::completion::options::OptionMap {
        let model = request.model.as_deref().unwrap_or(&self.model);
        crate::options::converse(self.family(model), model, request, fields)
    }

    fn api(&self) -> Api {
        Api::from_static("bedrock.converse")
    }

    fn provider(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// Converse reads images in user turns and tool results, never in
    /// assistant turns. Claude reads them; so does every other model but
    /// those the catalog lists as reading no images, or, for a model the
    /// catalog does not list (a region inference profile such as
    /// `eu.meta.llama3-3-70b-instruct-v1:0`), the text-only families.
    fn accepts(&self, model: &str) -> Accepts {
        let images = self.family(model) == Family::Claude
            || spec(model).map_or_else(
                || {
                    !TEXT_ONLY
                        .split_whitespace()
                        .any(|family| model.contains(family))
                },
                |spec| spec.input.image,
            );
        Accepts {
            user_images: images,
            assistant_images: false,
            tool_result_images: images,
            tools: true,
        }
    }

    /// Claude binds its thinking to the request's tools and system prompt
    /// on Converse as on Anthropic's own API.
    fn binds_context(&self, model: &str) -> bool {
        self.family(model) == Family::Claude
            && spec(model).is_some_and(|spec| spec.compat.binds_context)
    }

    /// Converse rejects a conversation that does not start with a user
    /// message.
    fn starts_with_user(&self) -> bool {
        true
    }

    /// A later system message goes as user text where it stands, so adding
    /// one never changes the cached prefix before it.
    fn later_system(&self, _model: &str) -> rig_core::completion::LaterSystem {
        rig_core::completion::LaterSystem::UserText
    }

    /// Converse takes user and assistant messages only in alternation.
    fn alternates_roles(&self) -> bool {
        true
    }

    /// Converse takes a hosted tool's use and result only beside a
    /// `toolConfig`, which only the request's own tools make.
    fn hosted_needs_tools(&self) -> bool {
        true
    }

    /// Only the request's tools reach Converse's `toolConfig`;
    /// `additional_params` go to `additionalModelRequestFields`. Converse
    /// has no `none` tool choice, so `ToolChoice::None` sends no
    /// `toolConfig`, and Converse rejects tool blocks without one.
    fn declares_tools(&self, request: &CompletionRequest) -> bool {
        !request.tools.is_empty() && !matches!(request.tool_choice, Some(ToolChoice::None))
    }

    fn sends_alone(&self, block: &rig_core::message::AssistantContent) -> bool {
        crate::request::sends(block, self)
    }

    /// Converse carries images in its four formats, documents in a format it
    /// lists, and inline data, which must be valid base64. Only Nova reads S3
    /// objects and video. Converse rejects a document's text source and
    /// takes no audio, so a string document goes as its text.
    fn encodes(&self, model: &str, media: Media<'_>) -> bool {
        let nova = self.family(model) == Family::Nova;
        let stored = |data: &DocumentSourceKind| {
            nova || !matches!(data, DocumentSourceKind::Url(url) if url.starts_with("s3://"))
        };
        match media {
            Media::Image(image, _) => stored(&image.data) && request::image(image).is_ok(),
            Media::Document(document) => {
                stored(&document.data) && request::document(document).is_ok()
            }
            Media::Video(video) => {
                nova && self.accepts(model).user_images && request::video(video).is_ok()
            }
            Media::Audio(_) => false,
        }
    }

    /// Converse tool-use ids match `[a-zA-Z0-9_-]{1,64}`.
    fn normalize_tool_call_id(&self, id: &str, _model: &str, _source: Option<&Origin>) -> String {
        rig_core::providers::internal::wire_ids::legal_call_id(id, 64)
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/toolUse/toolUseId")
    }

    /// A hosted tool's use is a typed `toolUse`; its result is the
    /// `toolResult` in the same turn. Converse rejects one without the other.
    fn hosted_pair(&self, item: &Value) -> Option<(Pairing, String)> {
        let (side, body) = match item.get("toolUse") {
            Some(body) => (Pairing::Use, body),
            None => (Pairing::Result, item.get("toolResult")?),
        };
        Some((side, body.str("toolUseId")?.to_owned()))
    }
}

/// The request id of a sent Converse call, and its failure, if the SDK
/// read one.
fn sent<O: RequestId, E: ProvideErrorMetadata>(
    sent: Result<O, SdkError<E, HttpResponse>>,
) -> (Option<String>, Option<ProviderError>) {
    match sent {
        Ok(output) => (output.request_id().map(str::to_owned), None),
        Err(error) => (
            error.request_id().map(str::to_owned),
            Some(sdk_error(error)),
        ),
    }
}

impl Transport<Converse> for BedrockRuntime {
    fn send(&self, payload: ConverseRequest, exchange: Exchange) -> Opening<ConverseFrame> {
        let ConverseRequest { model, body } = payload;
        let capture = match serde_json::to_vec(&body) {
            Ok(body) => Capture::new(body),
            Err(error) => return Opening::failed(ProviderError::request(error)),
        };
        let runtime = self.clone();
        Opening::new(async move {
            let client = runtime.inner().await;
            let unary = exchange.mode == Mode::Unary;
            let interceptor = capture.clone();
            let (request_id, failure) = if unary {
                let call = client.converse().model_id(model).customize();
                sent(call.interceptor(interceptor).send().await)
            } else {
                let call = client.converse_stream().model_id(model).customize();
                sent(call.interceptor(interceptor).send().await)
            };
            // A success's body is never the SDK's to read, so the SDK fails a
            // unary call it could not deserialize; the reply is decoded here.
            let Some(mut body) = capture.reply() else {
                let failure = failure.unwrap_or_else(|| {
                    ProviderError::Response("Converse sent no reply".to_owned())
                });
                return Ok(Opened::failed(failure));
            };
            if unary {
                let mut bytes = Vec::new();
                while let Some(chunk) = capture::chunk(&mut body).await {
                    bytes.extend(chunk?);
                }
                let document: Value = serde_json::from_slice(&bytes)
                    .map_err(|error| ProviderError::Json(error.into()))?;
                let frames = futures::stream::iter([Ok(ConverseFrame::Whole(document.clone()))]);
                return Ok(Opened::new(frames)
                    .with_request_id(request_id)
                    .with_document(document));
            }
            let frames = async_stream::stream! {
                let mut events = Events::default();
                while let Some(chunk) = capture::chunk(&mut body).await {
                    match chunk {
                        Ok(chunk) => {
                            for event in events.read(&chunk) {
                                yield Ok(ConverseFrame::Event(event));
                            }
                        }
                        Err(error) => {
                            yield Err(error);
                            break;
                        }
                    }
                }
            };
            Ok(Opened::new(frames).with_request_id(request_id))
        })
    }
}

#[cfg(test)]
mod tests;
