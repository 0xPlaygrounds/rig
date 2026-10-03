//! The Bedrock Converse completion wire and model identifiers.
//! Model availability and inference-profile support depend on the AWS region.
//!
//! ```no_run
//! use rig_bedrock::{client::BedrockRuntime, completion::{AMAZON_NOVA_LITE, Converse}};
//! use rig_core::Model;
//!
//! let model = BedrockRuntime::from_env().completion(AMAZON_NOVA_LITE);
//! # let _ = model;
//! ```

use crate::{
    capture::{Capture, Events},
    client::BedrockRuntime,
    streaming::StreamState,
    types::{
        assistant_content::PROVIDER_NAME,
        completion_request::AwsCompletionRequest,
        errors::{sdk_error, stream_error},
    },
};

use aws_sdk_bedrockruntime::operation::RequestId;
use aws_sdk_bedrockruntime::operation::converse::ConverseOutput;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::completion::CompletionRequest;
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::Completion;
use rig_core::wire::{Descriptor, Mode, Wire};

// Profile identifiers with a us. prefix route inference within the US region
// family; callers elsewhere must select a supported regional profile.

/// `amazon.nova-lite-v1:0`
pub const AMAZON_NOVA_LITE: &str = "amazon.nova-lite-v1:0";
/// `amazon.nova-micro-v1:0`
pub const AMAZON_NOVA_MICRO: &str = "amazon.nova-micro-v1:0";
/// `amazon.nova-pro-v1:0`
pub const AMAZON_NOVA_PRO: &str = "amazon.nova-pro-v1:0";
/// `amazon.nova-canvas-v1:0` image generation model
pub const AMAZON_NOVA_CANVAS: &str = "amazon.nova-canvas-v1:0";
/// `amazon.nova-reel-v1:0` video generation model
pub const AMAZON_NOVA_REEL_V1_0: &str = "amazon.nova-reel-v1:0";
/// `amazon.nova-reel-v1:1` video generation model
pub const AMAZON_NOVA_REEL_V1_1: &str = "amazon.nova-reel-v1:1";
/// `amazon.nova-sonic-v1:0` speech model
pub const AMAZON_NOVA_SONIC: &str = "amazon.nova-sonic-v1:0";
/// `amazon.rerank-v1:0` rerank model
pub const AMAZON_RERANK_1_0: &str = "amazon.rerank-v1:0";
/// `amazon.titan-embed-text-v1` embedding model
pub const AMAZON_TITAN_EMBEDDINGS_G1_TEXT: &str = "amazon.titan-embed-text-v1";
/// `amazon.titan-embed-image-v1` multimodal embedding model
pub const AMAZON_TITAN_MULTIMODAL_EMBEDDINGS_G1: &str = "amazon.titan-embed-image-v1";
/// `amazon.titan-embed-text-v2:0` embedding model
pub const AMAZON_TITAN_TEXT_EMBEDDINGS_V2: &str = "amazon.titan-embed-text-v2:0";

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

/// `cohere.embed-english-v3` embedding model
pub const COHERE_EMBED_ENGLISH: &str = "cohere.embed-english-v3";
/// `cohere.embed-multilingual-v3` embedding model
pub const COHERE_EMBED_MULTILINGUAL: &str = "cohere.embed-multilingual-v3";
/// `cohere.rerank-v3-5:0` rerank model
pub const COHERE_RERANK_V3_5: &str = "cohere.rerank-v3-5:0";

/// `us.deepseek.r1-v1:0` (cross-region profile)
pub const DEEPSEEK_R1: &str = "us.deepseek.r1-v1:0";

/// `luma.ray-v2:0` video generation model
pub const LUMA_RAY_V2_0: &str = "luma.ray-v2:0";

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

/// `stability.sd3-5-large-v1:0` image generation model
pub const STABILITY_SD3_5_LARGE: &str = "stability.sd3-5-large-v1:0";
/// `stability.stable-image-core-v1:1` image generation model
pub const STABILITY_STABLE_IMAGE_CORE_1_0: &str = "stability.stable-image-core-v1:1";
/// `stability.stable-image-ultra-v1:1` image generation model
pub const STABILITY_STABLE_IMAGE_ULTRA_1_0: &str = "stability.stable-image-ultra-v1:1";

/// `twelvelabs.pegasus-1-2-v1:0` video-understanding model
pub const TWELVELABS_PEGASUS_V1_2: &str = "twelvelabs.pegasus-1-2-v1:0";

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
    /// When enabled, cache checkpoints are inserted into Converse API requests
    /// to take advantage of [Bedrock prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html).
    /// Marks system content and, when history contains no reasoning, the final
    /// message. Disabled by default.
    pub prompt_caching: bool,
    /// Guardrail applied to unary Converse requests, if any.
    /// Set through [`Converse::with_guardrail`].
    pub guardrail: Option<aws_bedrock::GuardrailConfiguration>,
}

impl Converse {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            family: None,
            prompt_caching: false,
            guardrail: None,
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

    /// Enables checkpoints after system content and the final message.
    /// History containing reasoning suppresses the message checkpoint. Tool
    /// definitions are not marked; model-specific caching limits apply.
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }

    /// Apply a [Bedrock guardrail](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
    /// to unary Converse requests. Streaming requests do not apply this setting.
    ///
    /// `identifier` is the guardrail ID or ARN; `version` is a version or `DRAFT`.
    /// The normalized finish reason reports guardrail intervention as content
    /// filtering.
    pub fn with_guardrail(
        mut self,
        identifier: impl Into<String>,
        version: impl Into<String>,
        trace: aws_bedrock::GuardrailTrace,
    ) -> Self {
        self.guardrail = Some(
            aws_bedrock::GuardrailConfiguration::builder()
                .guardrail_identifier(identifier)
                .guardrail_version(version)
                .trace(trace)
                .build(),
        );
        self
    }

    fn request_model<'a>(&'a self, model: Option<&'a str>) -> &'a str {
        model.unwrap_or(&self.model)
    }
}

/// One Converse request: the model it addresses and the request prepared
/// for it.
pub struct ConverseRequest {
    pub model: String,
    pub request: AwsCompletionRequest,
    pub guardrail: Option<aws_bedrock::GuardrailConfiguration>,
}

/// One unit of a Converse reply.
#[derive(Clone, Debug)]
pub enum ConverseFrame {
    /// The reply opened, with the AWS request id from the SDK's response
    /// metadata.
    Opened { request_id: Option<String> },
    /// The whole unary reply. Its JSON body is the response's `raw`.
    Whole(Box<ConverseOutput>),
    /// One streamed event.
    Event(aws_bedrock::ConverseStreamOutput),
    /// The JSON Bedrock sent for the event that follows, as
    /// `{"<event type>": <payload>}`. The message-level events make up a
    /// stream's `raw`.
    Raw(serde_json::Value),
}

impl Wire for Converse {
    type Op = Completion;
    type Payload = ConverseRequest;
    type Frame = ConverseFrame;
    type Decoder<'id> = StreamState;

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
        let model = self.request_model(request.model.as_deref()).to_owned();
        Ok(ConverseRequest {
            request: AwsCompletionRequest::new(request, self.family(&model), self.prompt_caching),
            model,
            guardrail: self.guardrail.clone(),
        })
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        StreamState::default()
    }
}

/// Model families pi's catalog lists as text-only on Bedrock, separated by
/// spaces.
const TEXT_ONLY: &str = "amazon.nova-micro deepseek. meta.llama3-8b meta.llama3-70b \
    meta.llama3-1- meta.llama3-3- minimax. mistral.devstral mistral.mistral-7b \
    mistral.mistral-large-2402 mistral.mistral-small-2402 mistral.mixtral mistral.voxtral \
    moonshot.kimi-k2-thinking nvidia.nemotron-nano-3 nvidia.nemotron-nano-9b \
    nvidia.nemotron-super openai.gpt-oss qwen.qwen3-2 qwen.qwen3-3 qwen.qwen3-coder \
    qwen.qwen3-next writer.palmyra zai.glm";

impl rig_core::completion::ReplayTarget for Converse {
    fn api(&self) -> rig_core::message::Api {
        rig_core::message::Api::from_static("bedrock.converse")
    }

    fn provider(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// Converse reads images in user turns and tool results, never in
    /// assistant turns. Claude reads them; so does every other model but
    /// the text-only families.
    fn accepts(&self, model: &str) -> rig_core::completion::Accepts {
        let images = self.family(model) == Family::Claude
            || !TEXT_ONLY
                .split_whitespace()
                .any(|family| model.contains(family));
        rig_core::completion::Accepts {
            user_images: images,
            assistant_images: false,
            tool_result_images: images,
            tools: true,
        }
    }

    /// Converse tool-use ids match `[a-zA-Z0-9_-]{1,64}`.
    fn normalize_tool_call_id(
        &self,
        id: &str,
        _model: &str,
        _source: Option<&rig_core::message::Origin>,
    ) -> String {
        id.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .take(64)
            .collect()
    }
}

impl Transport<Converse> for BedrockRuntime {
    fn send(&self, payload: ConverseRequest, exchange: Exchange) -> Opening<ConverseFrame> {
        let mode = exchange.mode;
        let ConverseRequest {
            model,
            request,
            guardrail,
        } = payload;
        let additional_params = request.additional_params();
        let inference_config = request.inference_config();
        let prepared = (|| {
            Ok::<_, ProviderError>((
                request.tools_config()?,
                request.output_config()?,
                request.system_prompt()?,
                request.messages()?,
            ))
        })();
        let (tool_config, output_config, system_prompt, messages) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return Opening::failed(error),
        };
        let runtime = self.clone();
        let capture = Capture::default();
        Opening::new(async move {
            let client = runtime.inner().await;
            match mode {
                Mode::Unary => {
                    let sent = client
                        .converse()
                        .model_id(model)
                        .set_additional_model_request_fields(additional_params)
                        .set_inference_config(Some(inference_config))
                        .set_tool_config(tool_config)
                        .set_system(system_prompt)
                        .set_messages(Some(messages))
                        .set_output_config(output_config)
                        .set_guardrail_config(guardrail)
                        .customize()
                        .interceptor(capture.clone())
                        .send()
                        .await;
                    Ok(match sent {
                        Ok(output) => {
                            let request_id = output.request_id().map(str::to_owned);
                            let opened = Opened::new(futures::stream::iter([
                                Ok(ConverseFrame::Opened {
                                    request_id: request_id.clone(),
                                }),
                                Ok(ConverseFrame::Whole(Box::new(output))),
                            ]))
                            .with_request_id(request_id);
                            match capture.document() {
                                Some(document) => opened.with_document(document),
                                None => opened,
                            }
                        }
                        Err(error) => Opened::failed(sdk_error(error)),
                    })
                }
                Mode::Streaming => {
                    let sent = client
                        .converse_stream()
                        .model_id(model)
                        .set_additional_model_request_fields(additional_params)
                        .set_inference_config(Some(inference_config))
                        .set_tool_config(tool_config)
                        .set_system(system_prompt)
                        .set_messages(Some(messages))
                        .set_output_config(output_config)
                        .customize()
                        .interceptor(capture.clone())
                        .send()
                        .await;
                    let response = match sent {
                        Ok(response) => response,
                        Err(error) => {
                            return Ok(Opened::failed(sdk_error(error)));
                        }
                    };
                    // Events do not carry the request id the terminal record
                    // reports: it is the operation's metadata.
                    let request_id = response.request_id().map(str::to_owned);
                    let opened = ConverseFrame::Opened {
                        request_id: request_id.clone(),
                    };
                    let frames = async_stream::stream! {
                        yield Ok(opened);
                        let mut stream = response.stream;
                        let mut events = Events::default();
                        let mut read = std::collections::VecDeque::new();
                        loop {
                            match stream.recv().await {
                                Ok(Some(output)) => {
                                    // Each event the SDK yields is the next
                                    // event message of the body it read.
                                    read.extend(events.read(&capture.take()));
                                    if let Some(raw) = read.pop_front() {
                                        yield Ok(ConverseFrame::Raw(raw));
                                    }
                                    yield Ok(ConverseFrame::Event(output));
                                }
                                Ok(None) => break,
                                Err(error) => {
                                    yield Err(stream_error(error.into_service_error()));
                                    break;
                                }
                            }
                        }
                    };
                    Ok(Opened::new(frames).with_request_id(request_id))
                }
            }
        })
    }
}
