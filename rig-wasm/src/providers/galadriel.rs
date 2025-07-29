use crate::{JsResult, ModelOpts};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::js_sys::{Array, Reflect};

use crate::completion::{CompletionRequest, Message};
use crate::vector_store::JsVectorStore;
use futures::StreamExt;
use futures::TryStreamExt;
use rig::client::CompletionClient;
use rig::completion::{Chat, CompletionModel, Prompt};
use rig::streaming::StreamingPrompt;
use wasm_streams::ReadableStream;

#[wasm_bindgen]
pub struct GaladrielAgent(rig::agent::Agent<rig::providers::galadriel::CompletionModel>);

#[wasm_bindgen]
impl GaladrielAgent {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: crate::JsAgentOpts) -> JsResult<Self> {
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
                    let tool = crate::tool::JsTool::new(item).unwrap();
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

        let fine_tune_key = Reflect::get(&opts, &JsValue::from_str("fineTuneKey"))
            .ok()
            .and_then(|v| v.as_string());

        let mut agent = rig::providers::galadriel::Client::new(&api_key, fine_tune_key.as_deref())
            .agent(&model);

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

    pub async fn prompt_stream(
        &self,
        prompt: &str,
    ) -> Result<wasm_streams::readable::sys::ReadableStream, JsValue> {
        let stream =
            self.0.stream_prompt(prompt).await.map_err(|x| {
                JsError::new(format!("Error while streaming response: {x}").as_ref())
            })?;

        let js_stream = stream
            .map_ok(|x| {
                serde_wasm_bindgen::to_value(&x)
                    .map_err(|e| JsValue::from_str(&format!("Failed streaming: {e}")))
            })
            .map(|result| match result {
                Ok(Ok(js)) => Ok(js),
                Ok(Err(js_err)) => Err(js_err),
                Err(e) => Err(JsValue::from_str(&format!("Stream error: {e}"))),
            });

        Ok(ReadableStream::from_stream(js_stream).into_raw())
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

/// The Galadriel completions chat API.
#[wasm_bindgen]
pub struct GaladrielCompletionsModel(rig::providers::galadriel::CompletionModel);

#[wasm_bindgen]
impl GaladrielCompletionsModel {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: crate::JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let fine_tune_key = model_opts.additional_params.get("fineTuneKey").map(|x| {
            x.as_str()
                .expect("Fine tuning keys for the Galadriel API are expected to be strings")
        });

        let client = rig::providers::galadriel::Client::new(&model_opts.api_key, fine_tune_key);

        let model = client.completion_model(&model_opts.model_name);

        Ok(Self(model))
    }

    pub async fn completion(&self, opts: crate::JsCompletionOpts) -> JsResult<JsValue> {
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
