use crate::completion::{AssistantContent, CompletionRequest, Document, Message, ToolDefinition};
use crate::embedding::Embedding;
use crate::image_generation::ImageGenerationRequest;
use crate::tool::JsTool;
use crate::transcription::TranscriptionRequest;
use crate::vector_store::JsVectorStore;
use crate::{
    JsAgentOpts, JsCompletionOpts, JsImageGenerationOpts, JsModelOpts, JsResult, JsToolObject,
    JsTranscriptionOpts, JsVectorStoreShim, ModelOpts, StringIterable, log,
};
use futures::StreamExt;
use futures::TryStreamExt;
use rig::OneOrMany;
use rig::agent::{Agent, AgentBuilder};
use rig::client::embeddings::EmbeddingsClient;
use rig::client::image_generation::ImageGenerationClient;
use rig::client::{AudioGenerationClient, CompletionClient, TranscriptionClient};
use rig::completion::{Chat, CompletionModel, Prompt};
use rig::embeddings::EmbeddingModel;
use rig::image_generation::ImageGenerationModel;
use rig::streaming::StreamingPrompt;
use rig::transcription::{TranscriptionModelDyn, TranscriptionResponse};
use serde_json::Map;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::js_sys::{self, Array, Reflect};
use wasm_streams::ReadableStream;

#[wasm_bindgen]
pub struct OpenAIAgent(Agent<rig::providers::openai::responses_api::ResponsesCompletionModel>);

#[wasm_bindgen]
impl OpenAIAgent {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: JsAgentOpts) -> JsResult<Self> {
        let api_key = Reflect::get(&opts, &JsValue::from_str("apiKey"))
            .map_err(|_| JsError::new("failed to get apiKey"))?
            .as_string()
            .ok_or(JsError::new("apiKey property of Agent is required"))?;

        let model = Reflect::get(&opts, &JsValue::from_str("model"))
            .map_err(|_| JsError::new("failed to get model"))?
            .as_string()
            .ok_or(JsError::new("model property of Agent is required"))?;

        let preamble = Reflect::get(&opts, &JsValue::from_str("preamble"))
            .ok()
            .and_then(|v| v.as_string());

        let context = Reflect::get(&opts, &JsValue::from_str("context"))
            .ok()
            .and_then(|v| {
                if v.is_undefined() || v.is_null() {
                    None
                } else {
                    Some(
                        Array::from(&v)
                            .iter()
                            .filter_map(|x| x.as_string())
                            .collect::<Vec<_>>(),
                    )
                }
            });

        let temperature = Reflect::get(&opts, &JsValue::from_str("temperature"))
            .ok()
            .and_then(|v| v.as_f64());

        let tools = match Reflect::get(&opts, &JsValue::from_str("tools")) {
            Ok(v) if v.is_undefined() || v.is_null() => None,
            Ok(v) => {
                let array = Array::from(&v);
                let mut converted = Vec::new();
                for item in array.iter() {
                    let tool = JsTool::new(item).unwrap();
                    converted.push(tool);
                }
                Some(converted)
            }
            Err(e) => {
                return Err(JsError::new(&format!(
                    "Failed to get tools property: {e:?}"
                )));
            }
        };

        let dynamic_context = Reflect::get(&opts, &JsValue::from_str("dynamicContext"))
            .ok()
            .and_then(|v| {
                if v.is_undefined() || v.is_null() {
                    None
                } else {
                    let sample = Reflect::get(&v, &JsValue::from_str("sample"))
                        .ok()
                        .and_then(|v| v.as_f64())
                        .map(|v| v as usize)?;

                    let store = JsVectorStore::new(
                        Reflect::get(&v, &JsValue::from_str("dynamicTools")).ok()?,
                    )
                    .expect("dynamicTools should exist!");

                    Some((sample, store))
                }
            });

        let _dynamic_tools = Reflect::get(&opts, &JsValue::from_str("dynamicTools"))
            .ok()
            .and_then(|v| {
                if v.is_undefined() || v.is_null() {
                    None
                } else {
                    let sample = Reflect::get(&v, &JsValue::from_str("sample"))
                        .ok()
                        .and_then(|v| v.as_f64())
                        .map(|v| v as usize)?;

                    let store = JsVectorStore::new(
                        Reflect::get(&v, &JsValue::from_str("dynamicTools")).ok()?,
                    );

                    Some((sample, store))
                }
            });

        let mut agent = rig::providers::openai::Client::new(&api_key).agent(&model);

        if let Some(preamble) = preamble {
            agent = agent.preamble(&preamble);
        }

        if let Some(context) = context {
            for doc in context {
                agent = agent.context(&doc);
            }
        }

        if let Some(temperature) = temperature {
            agent = agent.temperature(temperature);
        }

        if let Some(tools) = tools {
            for tool in tools {
                agent = agent.tool(tool);
            }
        }

        if let Some((sample, ctx)) = dynamic_context {
            agent = agent.dynamic_context(sample, ctx);
        }

        Ok(Self(agent.build()))
    }

    pub async fn prompt(&self, prompt: &str) -> JsResult<String> {
        self.0
            .prompt(prompt)
            .await
            .map_err(|x| JsError::new(x.to_string().as_ref()))
    }

    pub async fn prompt_stream(&self, prompt: &str) -> JsResult<wasm_streams::ReadableStream> {
        let res = self
            .0
            .stream_prompt(prompt)
            .await
            .map_err(|x| JsError::new(format!("Error while streaming response: {x}").as_ref()))?
            .map(|x| {
                serde_wasm_bindgen::to_value(&x)
                    .map_err(|x| JsValue::from_str(format!("Failed streaming: {x}").as_ref()))
            });

        Ok(ReadableStream::from_stream(res))
    }

    pub async fn prompt_multi_turn(&self, prompt: &str, turns: u32) -> JsResult<String> {
        self.0
            .prompt(prompt)
            .multi_turn(turns as usize)
            .await
            .map_err(|x| JsError::new(x.to_string().as_ref()))
    }

    pub async fn chat(&self, prompt: &str, messages: Vec<Message>) -> JsResult<String> {
        let messages: Vec<rig::message::Message> = messages
            .into_iter()
            .map(rig::message::Message::from)
            .collect();
        self.0
            .chat(prompt, messages)
            .await
            .map_err(|x| JsError::new(x.to_string().as_ref()))
    }
}

/// The OpenAI Responses API, modelled as the Completions API.
#[wasm_bindgen]
pub struct OpenAIResponsesCompletionModel(
    rig::providers::openai::responses_api::ResponsesCompletionModel,
);

#[wasm_bindgen]
impl OpenAIResponsesCompletionModel {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::openai::Client::new(&model_opts.api_key);
        let model = client.completion_model(&model_opts.model_name);
        Ok(Self(model))
    }

    pub async fn completion(&self, opts: JsCompletionOpts) -> JsResult<JsValue> {
        let req: CompletionRequest = CompletionRequest::new(opts)?;
        let req: rig::completion::CompletionRequest = rig::completion::CompletionRequest::from(req);

        let res = self
            .0
            .completion(req)
            .await
            .map_err(|x| JsError::new(x.to_string().as_ref()))?;

        let res = crate::completion::CompletionResponse::from(res);

        let js_val =
            serde_wasm_bindgen::to_value(&res).map_err(|x| JsError::new(x.to_string().as_ref()))?;

        Ok(js_val)
    }
}

/// The OpenAI completions chat API.
#[wasm_bindgen]
pub struct OpenAICompletionsCompletionModel(rig::providers::openai::completion::CompletionModel);

#[wasm_bindgen]
impl OpenAICompletionsCompletionModel {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::openai::Client::new(&model_opts.api_key);
        let model = client
            .completion_model(&model_opts.model_name)
            .completions_api();
        Ok(Self(model))
    }

    pub async fn completion(&self, opts: JsCompletionOpts) -> JsResult<JsValue> {
        let req: CompletionRequest = CompletionRequest::new(opts)?;
        let req: rig::completion::CompletionRequest = rig::completion::CompletionRequest::from(req);

        let res = self
            .0
            .completion(req)
            .await
            .map_err(|x| JsError::new(x.to_string().as_ref()))?;

        let res = crate::completion::CompletionResponse::from(res);

        let js_val =
            serde_wasm_bindgen::to_value(&res).map_err(|x| JsError::new(x.to_string().as_ref()))?;

        Ok(js_val)
    }
}

#[wasm_bindgen]
pub struct OpenAIEmbeddingModel(rig::providers::openai::embedding::EmbeddingModel);

#[wasm_bindgen]
impl OpenAIEmbeddingModel {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::openai::Client::new(&model_opts.api_key);
        let model = client.embedding_model(&model_opts.model_name);
        Ok(Self(model))
    }

    pub async fn embed_text(&self, text: String) -> JsResult<Embedding> {
        let res = self
            .0
            .embed_text(&text)
            .await
            .map_err(|e| JsError::new(e.to_string().as_ref()))?;

        Ok(Embedding::from(res))
    }

    pub async fn embed_texts(&self, iter: StringIterable) -> JsResult<Vec<Embedding>> {
        let arr = js_sys::Array::from(&iter.obj);

        let val = arr
            .into_iter()
            .map(|x| {
                x.as_string()
                    .ok_or_else(|| JsError::new(format!("Expected string, got {x:?}").as_ref()))
            })
            .collect::<Result<Vec<String>, JsError>>()?;

        Ok(self
            .0
            .embed_texts(val)
            .await
            .map_err(|x| JsError::new(x.to_string().as_ref()))?
            .into_iter()
            .map(crate::embedding::Embedding::from)
            .collect())
    }
}

#[wasm_bindgen]
pub struct OpenAITranscriptionModel(rig::providers::openai::TranscriptionModel);

#[wasm_bindgen]
impl OpenAITranscriptionModel {
    #[wasm_bindgen]
    pub fn new(opts: JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::openai::Client::new(&model_opts.api_key);
        let model = client.transcription_model(&model_opts.model_name);
        Ok(Self(model))
    }

    #[wasm_bindgen]
    pub async fn transcription(
        &self,
        opts: JsTranscriptionOpts,
    ) -> JsResult<OpenAITranscriptionResponse> {
        let req =
            serde_wasm_bindgen::from_value::<crate::transcription::TranscriptionRequest>(opts.obj)
                .map_err(|x| {
                    JsError::new(
                        format!("Error while creating transcription options: {x}").as_ref(),
                    )
                })?;
        let req = rig::transcription::TranscriptionRequest::from(req);

        let res = self
            .0
            .transcription(req)
            .await
            .map_err(|x| JsError::new(format!("Error while transcribing: {x}").as_ref()))?;

        let transcription = OpenAITranscriptionResponse(res);

        Ok(transcription)
    }
}

#[wasm_bindgen]
pub struct OpenAITranscriptionResponse(TranscriptionResponse<()>);

#[wasm_bindgen]
impl OpenAITranscriptionResponse {
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.0.text.clone()
    }
}

#[wasm_bindgen]
pub struct OpenAIImageGenerationModel(rig::providers::openai::ImageGenerationModel);

#[wasm_bindgen]
impl OpenAIImageGenerationModel {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::openai::Client::new(&model_opts.api_key);
        let model = client.image_generation_model(&model_opts.model_name);
        Ok(Self(model))
    }

    pub async fn image_generation(
        &self,
        opts: JsImageGenerationOpts,
    ) -> JsResult<OpenAIImageGenerationResponse> {
        let req =
            serde_wasm_bindgen::from_value::<crate::image_generation::ImageGenerationRequest>(
                opts.obj,
            )
            .map_err(|x| {
                JsError::new(format!("Error while creating transcription options: {x}").as_ref())
            })?;
        let req = rig::image_generation::ImageGenerationRequest::from(req);
        let res = self
            .0
            .image_generation(req)
            .await
            .map_err(|x| JsError::new(format!("Error while creating image: {x}").as_ref()))?;

        Ok(OpenAIImageGenerationResponse(res))
    }
}

#[wasm_bindgen]
pub struct OpenAIImageGenerationResponse(
    rig::image_generation::ImageGenerationResponse<rig::providers::openai::ImageGenerationResponse>,
);

#[wasm_bindgen]
impl OpenAIImageGenerationResponse {
    pub fn image_bytes(&self) -> wasm_bindgen_futures::js_sys::Uint8Array {
        wasm_bindgen_futures::js_sys::Uint8Array::from(self.0.image.as_ref())
    }
}
