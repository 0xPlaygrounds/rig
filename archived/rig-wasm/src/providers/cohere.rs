use crate::completion::CompletionRequest;
use crate::embedding::Embedding;
use crate::tool::JsTool;
use crate::{JsCompletionOpts, JsModelOpts, StringIterable};
use crate::{JsResult, ModelOpts, vector_store::JsVectorStore};
use futures::{StreamExt, TryStreamExt};
use rig::client::completion::CompletionClient;
use rig::completion::CompletionModel;
use rig::completion::{Chat, Prompt};
use rig::embeddings::EmbeddingModel;
use rig::streaming::StreamingPrompt;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::js_sys::{Array, Reflect};

#[wasm_bindgen]
pub struct CohereAgent(rig::agent::Agent<rig::providers::cohere::CompletionModel>);

#[wasm_bindgen]
impl CohereAgent {
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

        let mut agent = rig::providers::cohere::Client::new(&api_key).agent(&model);

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

        Ok(wasm_streams::ReadableStream::from_stream(js_stream).into_raw())
    }

    pub async fn prompt_multi_turn(&self, prompt: &str, turns: u32) -> JsResult<String> {
        self.0
            .prompt(prompt)
            .multi_turn(turns as usize)
            .await
            .map_err(|x| JsError::new(x.to_string().as_ref()))
    }

    pub async fn chat(
        &self,
        prompt: &str,
        messages: Vec<crate::completion::Message>,
    ) -> JsResult<String> {
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

/// The Cohere Responses API, modelled as the Completions API.
#[wasm_bindgen]
pub struct CohereCompletionModel(rig::providers::cohere::CompletionModel);

#[wasm_bindgen]
impl CohereCompletionModel {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: crate::JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::cohere::Client::new(&model_opts.api_key);
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

#[wasm_bindgen]
pub struct CohereEmbeddingModel(rig::providers::cohere::EmbeddingModel);

#[wasm_bindgen]
impl CohereEmbeddingModel {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::cohere::Client::new(&model_opts.api_key);

        let input_type = model_opts
            .additional_params
            .get("inputType")
            .ok_or(JsError::new(
                "cohere embedding model requires an input type (use `inputType` field) at creation",
            ))?
            .as_str()
            .ok_or(JsError::new(
                "CohereEmbeddingModel.inputType input is expected to be a string",
            ))?;
        let model = client.embedding_model(&model_opts.model_name, input_type);
        Ok(Self(model))
    }

    #[wasm_bindgen(js_name = "embedText")]
    pub async fn embed_text(&self, text: String) -> JsResult<Embedding> {
        let res = self
            .0
            .embed_text(&text)
            .await
            .map_err(|e| JsError::new(e.to_string().as_ref()))?;

        Ok(Embedding::from(res))
    }

    #[wasm_bindgen(js_name = "embedTexts")]
    pub async fn embed_texts(&self, iter: StringIterable) -> JsResult<Vec<Embedding>> {
        let arr = wasm_bindgen_futures::js_sys::Array::from(&iter.obj);

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
