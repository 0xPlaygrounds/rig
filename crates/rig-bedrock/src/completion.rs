//! The Bedrock Converse completion wire and model identifiers.
//! Model availability and inference-profile support depend on the AWS region.
//!
//! ```no_run
//! use rig_bedrock::{client::BedrockRuntime, completion::{AMAZON_NOVA_LITE, Converse}};
//! use rig_core::Model;
//!
//! let model = Model::new(Converse::new(AMAZON_NOVA_LITE), BedrockRuntime::from_env()?);
//! # let _ = model;
//! # Ok::<(), rig_core::client::ProviderClientError>(())
//! ```

use crate::{
    client::BedrockRuntime,
    streaming::StreamState,
    types::{
        assistant_content::{PROVIDER_NAME, reasoning_issuer},
        completion_request::AwsCompletionRequest,
        converse_output::InternalConverseOutput,
        errors::{
            AwsSdkConverseError, AwsSdkConverseStreamError, converse_stream_output_completion_error,
        },
    },
};

use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::completion::CompletionRequest;
use rig_core::driver::{Observation, Opened, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::Completion;
use rig_core::wire::{Mode, Wire};

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

/// The Converse endpoint for one model: `Converse` for a unary call,
/// `ConverseStream` for a streamed one.
#[derive(Clone, Debug)]
pub struct Converse {
    pub model: String,
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
            prompt_caching: false,
            guardrail: None,
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
pub enum ConverseFrame {
    /// The reply opened: the model it answers for, and the AWS request id
    /// from the SDK's response metadata.
    Opened {
        model: String,
        request_id: Option<String>,
    },
    /// The whole unary reply.
    Whole(Box<InternalConverseOutput>),
    /// One streamed event.
    Event(aws_bedrock::ConverseStreamOutput),
}

impl Wire for Converse {
    type Op = Completion;
    type Payload = ConverseRequest;
    type Frame = ConverseFrame;
    type Decoder = StreamState;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    /// Claude reasoning on Bedrock is Anthropic's; other models' is Bedrock's.
    fn replay_issuers(&self, model: Option<&str>) -> Vec<String> {
        vec![reasoning_issuer(self.request_model(model)).to_owned()]
    }

    fn reasoning_issuer(&self, model: Option<&str>) -> Option<&str> {
        let issuer = reasoning_issuer(self.request_model(model));
        (issuer != PROVIDER_NAME).then_some(issuer)
    }

    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<ConverseRequest, EncodeError> {
        let model = self.request_model(request.model.as_deref()).to_owned();
        Ok(ConverseRequest {
            request: AwsCompletionRequest::new(request, self.prompt_caching),
            model,
            guardrail: self.guardrail.clone(),
        })
    }

    fn decoder(&self, _mode: Mode) -> StreamState {
        StreamState::default()
    }
}

impl Transport<Converse> for BedrockRuntime {
    fn send(
        &self,
        payload: ConverseRequest,
        mode: Mode,
        _observation: Option<Observation>,
    ) -> Result<
        impl Future<Output = Opened<ConverseRequest, ConverseFrame>> + Send + 'static + use<>,
        ProviderError,
    > {
        let ConverseRequest {
            model,
            request,
            guardrail,
        } = payload;
        let tool_config = request.tools_config()?;
        let output_config = request.output_config()?;
        let additional_params = request.additional_params();
        let inference_config = request.inference_config();
        let system_prompt = request.system_prompt()?;
        let messages = request.messages()?;
        let runtime = self.clone();
        Ok(async move {
            let client = runtime.inner().await;
            match mode {
                Mode::Unary => {
                    let sent = client
                        .converse()
                        .model_id(model.clone())
                        .set_additional_model_request_fields(additional_params)
                        .set_inference_config(Some(inference_config))
                        .set_tool_config(tool_config)
                        .set_system(system_prompt)
                        .set_messages(Some(messages))
                        .set_output_config(output_config)
                        .set_guardrail_config(guardrail)
                        .send()
                        .await
                        .map_err(|sdk_error| ProviderError::from(AwsSdkConverseError(sdk_error)))
                        .and_then(|response| {
                            InternalConverseOutput::try_from(response).map_err(|error| {
                                ProviderError::Provider(format!("Type conversion error: {error}"))
                            })
                        });
                    match sent {
                        Ok(output) => {
                            let request_id = output.request_id().map(str::to_owned);
                            Opened {
                                request_id: request_id.clone(),
                                ..Opened::new(futures::stream::iter([
                                    Ok(ConverseFrame::Opened { model, request_id }),
                                    Ok(ConverseFrame::Whole(Box::new(output))),
                                ]))
                            }
                        }
                        Err(error) => Opened::failed(error),
                    }
                }
                Mode::Streaming => {
                    let sent = client
                        .converse_stream()
                        .model_id(model.clone())
                        .set_additional_model_request_fields(additional_params)
                        .set_inference_config(Some(inference_config))
                        .set_tool_config(tool_config)
                        .set_system(system_prompt)
                        .set_messages(Some(messages))
                        .set_output_config(output_config)
                        .send()
                        .await;
                    let response = match sent {
                        Ok(response) => response,
                        Err(sdk_error) => {
                            return Opened::failed(AwsSdkConverseStreamError(sdk_error).into());
                        }
                    };
                    // Events do not carry the request id the terminal record
                    // reports: it is the operation's metadata.
                    let request_id =
                        aws_sdk_bedrockruntime::operation::RequestId::request_id(&response)
                            .map(str::to_owned);
                    let opened = ConverseFrame::Opened {
                        model,
                        request_id: request_id.clone(),
                    };
                    let frames = async_stream::stream! {
                        yield Ok(opened);
                        let mut stream = response.stream;
                        loop {
                            match stream.recv().await {
                                Ok(Some(output)) => yield Ok(ConverseFrame::Event(output)),
                                Ok(None) => break,
                                Err(error) => {
                                    yield Err(converse_stream_output_completion_error(
                                        error.into_service_error(),
                                    ));
                                    break;
                                }
                            }
                        }
                    };
                    Opened {
                        request_id,
                        ..Opened::new(frames)
                    }
                }
            }
        })
    }
}
