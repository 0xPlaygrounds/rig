use std::fmt;
use std::path::Path;

use tera::{Context, Tera};

#[derive(serde::Serialize)]
enum Provider {
    Anthropic,
    Azure,
    Cohere,
    Gemini,
    Huggingface,
    Mistral,
    OpenAI,
    Together,
    Xai,
    Groq,
    DeepSeek,
    Galadriel,
    Hyperbolic,
    Mira,
    Moonshot,
    Ollama,
    Perplexity,
    VoyageAI,
    OpenRouter,
}

impl TryFrom<&str> for Provider {
    type Error = String;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let res = match value {
            "openai" => Self::OpenAI,
            "cohere" => Self::Cohere,
            "gemini" => Self::Gemini,
            "huggingface" => Self::Huggingface,
            "mistral" => Self::Mistral,
            "together" => Self::Together,
            "xai" => Self::Xai,
            "openrouter" => Self::OpenRouter,
            "anthropic" => Self::Anthropic,
            "azure" => Self::Azure,
            "groq" => Self::Groq,
            "deepseek" => Self::DeepSeek,
            "galadriel" => Self::Galadriel,
            "hyperbolic" => Self::Hyperbolic,
            "mira" => Self::Mira,
            "moonshot" => Self::Moonshot,
            "ollama" => Self::Ollama,
            "perplexity" => Self::Perplexity,
            "voyageai" => Self::VoyageAI,
            err => return Err(format!("Not a valid provider: {err}")),
        };

        Ok(res)
    }
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cohere => write!(f, "Cohere"),
            Self::Gemini => write!(f, "Gemini"),
            Self::Huggingface => write!(f, "HuggingFace"),
            Self::Mistral => write!(f, "Mistral"),
            Self::Together => write!(f, "Together"),
            Self::Xai => write!(f, "Xai"),
            Self::Anthropic => write!(f, "Anthropic"),
            Self::OpenAI => write!(f, "OpenAI"),
            Self::Azure => write!(f, "Azure"),
            Self::Groq => write!(f, "Groq"),
            Self::DeepSeek => write!(f, "DeepSeek"),
            Self::Galadriel => write!(f, "Galadriel"),
            Self::Hyperbolic => write!(f, "Hyperbolic"),
            Self::Mira => write!(f, "Mira"),
            Self::Moonshot => write!(f, "Moonshot"),
            Self::Ollama => write!(f, "Ollama"),
            Self::OpenRouter => write!(f, "OpenRouter"),
            Self::Perplexity => write!(f, "Perplexity"),
            Self::VoyageAI => write!(f, "VoyageAI"),
        }
    }
}

fn main() {
    let input_dir = "../rig-core/src/providers";
    println!("cargo:rerun-if-changed={input_dir}");
    let input_dir = Path::new(input_dir);
    let input_dir = input_dir.read_dir().unwrap();

    let out_dir = Path::new("src/providers");
    let mut tera = Tera::default();

    for entry in input_dir {
        let entry = entry.expect("This should be OK");
        if !entry.path().is_dir() {
            let path = entry.path();
            let filename = entry.file_name().into_string().unwrap();
            let filename = filename
                .strip_suffix(".rs")
                .expect("stripping .rs should never panic as we are only dealing with .rs files");

            if filename == "mod" {
                continue;
            }

            let Ok(provider_name) = Provider::try_from(filename) else {
                let err = format!("Invalid provider name: {filename}");
                panic!("{err}");
            };

            let file_contents = std::fs::read_to_string(path)
                .expect("to read the filepath for a file that should exist");

            let mut context = Context::new();
            context.insert("provider", &provider_name.to_string());
            context.insert("module_name", &filename);

            if file_contents.contains(" CompletionClient ") {
                context.insert("completion", &true);
            }

            if file_contents.contains(" EmbeddingsClient ") {
                context.insert("embeddings", &true);
            }
            if file_contents.contains(" TranscriptionClient ") {
                context.insert("transcription", &true);
            }
            if file_contents.contains(" ImageGenerationClient ") {
                context.insert("image_gen", &true);
            }
            if file_contents.contains(" AudioGenerationClient ") {
                context.insert("audio_gen", &true);
            }

            let file_contents = tera
                .render_str(CODE, &context)
                .expect("This shouldn't fail!");
            std::fs::write(out_dir.join(entry.file_name()), &file_contents).unwrap();

            let modrs_file = out_dir.join("mod.rs");
            let mut modrs_file_contents = std::fs::read_to_string(&modrs_file)
                .expect("The mod.rs file should expect in the providers module");
            let str = format!("pub mod {filename};");

            if modrs_file_contents.is_empty() {
                modrs_file_contents.push_str(&str);
                std::fs::write(modrs_file, modrs_file_contents)
                    .expect("writing to the mod.rs file should work");
            } else if !modrs_file_contents.contains(&str) {
                modrs_file_contents.push('\n');
                modrs_file_contents.push_str(&str);
                std::fs::write(modrs_file, modrs_file_contents)
                    .expect("writing to the mod.rs file should work");
            }
        } else {
            let filenames = entry
                .path()
                .read_dir()
                .unwrap()
                .map(|x| x.unwrap().file_name().into_string().unwrap());
            let dirname = entry.file_name().into_string().unwrap();
            println!("Reading dirname: {dirname}");
            let dirname_as_provider =
                Provider::try_from(dirname.as_ref()).expect("to compile to a provider that exists");

            let mut context = Context::new();
            context.insert("provider", &dirname_as_provider.to_string());
            context.insert("module_name", &dirname);
            for file in filenames {
                match file.trim() {
                    "completion" | "audio_generation" | "embedding" | "image_generation"
                    | "transcription" => {
                        context.insert(&file, &true);
                    }
                    _ => {
                        // nothing to do here
                    }
                }
            }

            let file_contents = tera
                .render_str(CODE, &context)
                .expect("This shouldn't fail!");
            let mut final_filename = entry
                .file_name()
                .into_string()
                .expect("files in the build directory should always have a filename");
            final_filename.push_str(".rs");
            std::fs::write(out_dir.join(final_filename), &file_contents).unwrap();

            let modrs_file = out_dir.join("mod.rs");
            let mut modrs_file_contents = std::fs::read_to_string(&modrs_file)
                .expect("The mod.rs file should expect in the providers module");
            let str = format!("pub mod {dirname};");

            if modrs_file_contents.is_empty() {
                modrs_file_contents.push_str(&str);
                std::fs::write(modrs_file, modrs_file_contents)
                    .expect("writing to the mod.rs file should work");
            } else if !modrs_file_contents.contains(&str) {
                modrs_file_contents.push('\n');
                modrs_file_contents.push_str(&str);
                std::fs::write(modrs_file, modrs_file_contents)
                    .expect("writing to the mod.rs file should work");
            }
        }
    }
}

const CODE: &str = r#"
use crate::{JsResult, ModelOpts};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::js_sys::{self, Array, Reflect};

{% if completion %}
use futures::StreamExt;
use futures::TryStreamExt;
use rig::streaming::StreamingPrompt;
use wasm_streams::ReadableStream;
use rig::client::CompletionClient;
use crate::completion::{AssistantContent, CompletionRequest, Document, Message, ToolDefinition};
use rig::completion::{Chat, CompletionModel, Prompt};
use crate::vector_store::JsVectorStore;
{% if module_name == "openai" %}
#[wasm_bindgen]
pub struct {{provider}}Agent(rig::agent::Agent<rig::providers::{{module_name}}::responses_api::ResponsesCompletionModel>);
{% else %}
#[wasm_bindgen]
pub struct {{provider}}Agent(rig::agent::Agent<rig::providers::{{module_name}}::CompletionModel>);
{% endif %}

#[wasm_bindgen]
impl {{provider}}Agent {
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

        let mut agent = rig::providers::{{module_name}}::Client::new(&api_key).agent(&model);

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

{% if module_name == "openai" %}
/// The OpenAI Responses API, modelled as the Completions API.
#[wasm_bindgen]
pub struct OpenAIResponsesCompletionModel(
    rig::providers::openai::responses_api::ResponsesCompletionModel,
);

#[wasm_bindgen]
impl OpenAIResponsesCompletionModel {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: crate::JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::openai::Client::new(&model_opts.api_key);
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
{% endif %}

/// The {{provider}} completions chat API.
#[wasm_bindgen]
pub struct {{provider}}CompletionsCompletionModel(rig::providers::openai::completion::CompletionModel);

#[wasm_bindgen]
impl {{provider}}CompletionsCompletionModel {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: crate::JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::{{module_name}}::Client::new(&model_opts.api_key);

        {% if module_name == "openai" %}
        let model = client
            .completion_model(&model_opts.model_name)
            .completions_api();
            {% else %}
            let model = client
                .completion_model(&model_opts.model_name);
            {% endif %}

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
{% endif %}

{% if embedding %}
use crate::embedding::Embedding;
use rig::embeddings::EmbeddingModel;
use rig::client::embeddings::EmbeddingsClient;
#[wasm_bindgen]
pub struct {{provider}}EmbeddingModel(rig::providers::{{module_name}}::embedding::EmbeddingModel);

#[wasm_bindgen]
impl {{provider}}EmbeddingModel {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: crate::JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::{{module_name}}::Client::new(&model_opts.api_key);
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

    pub async fn embed_texts(&self, iter: crate::StringIterable) -> JsResult<Vec<Embedding>> {
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
{% endif %}


{% if transcription %}
use crate::transcription::TranscriptionRequest;
use rig::client::TranscriptionClient;
use rig::transcription::{TranscriptionModel, TranscriptionResponse};
#[wasm_bindgen]
pub struct {{provider}}TranscriptionModel(rig::providers::{{module_name}}::TranscriptionModel);

#[wasm_bindgen]
impl {{provider}}TranscriptionModel {
    #[wasm_bindgen]
    pub fn new(opts: crate::JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::{{module_name}}::Client::new(&model_opts.api_key);
        let model = client.transcription_model(&model_opts.model_name);
        Ok(Self(model))
    }

    #[wasm_bindgen]
    pub async fn transcription(
        &self,
        opts: crate::JsTranscriptionOpts,
    ) -> JsResult<{{provider}}TranscriptionResponse> {
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

        let transcription = {{provider}}TranscriptionResponse(res);

        Ok(transcription)
    }
}
#[wasm_bindgen]
pub struct {{provider}}TranscriptionResponse(TranscriptionResponse<()>);

#[wasm_bindgen]
impl {{provider}}TranscriptionResponse {
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.0.text.clone()
    }
}
{% endif %}

{% if image_gen %}
use rig::client::image_generation::ImageGenerationClient;
use crate::image_generation::ImageGenerationRequest;
use rig::image_generation::ImageGenerationModel;
#[wasm_bindgen]
pub struct {{provider}}ImageGenerationModel(rig::providers::{{module_name}}::ImageGenerationModel);

#[wasm_bindgen]
impl {{provider}}ImageGenerationModel {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: crate::JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::{{module_name}}::Client::new(&model_opts.api_key);
        let model = client.image_generation_model(&model_opts.model_name);
        Ok(Self(model))
    }

    pub async fn image_generation(
        &self,
        opts: crate::JsImageGenerationOpts,
    ) -> JsResult<{{provider}}ImageGenerationResponse> {
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

        Ok({{provider}}ImageGenerationResponse(res))
    }
}

#[wasm_bindgen]
pub struct {{provider}}ImageGenerationResponse(
    rig::image_generation::ImageGenerationResponse<rig::providers::{{module_name}}::ImageGenerationResponse>,
);

#[wasm_bindgen]
impl {{provider}}ImageGenerationResponse {
    pub fn image_bytes(&self) -> wasm_bindgen_futures::js_sys::Uint8Array {
        wasm_bindgen_futures::js_sys::Uint8Array::from(self.0.image.as_ref())
    }
}
{% endif %}
{% if audio_gen %}
use rig::client::AudioGenerationClient;
#[wasm_bindgen]
pub struct OpenAIAudioGenerationModel(
    rig::providers::openai::audio_generation::AudioGenerationModel,
);

#[wasm_bindgen]
impl OpenAIAudioGenerationModel {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: crate::JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::openai::Client::new(&model_opts.api_key);
        let model = client.audio_generation_model(&model_opts.model_name);
        Ok(Self(model))
    }

    pub async fn audio_generation(
        &self,
        opts: JsAudioGenerationOpts,
    ) -> JsResult<OpenAIAudioGenerationResponse> {
        let req =
            serde_wasm_bindgen::from_value::<crate::audio_generation::AudioGenerationRequest>(
                opts.obj,
            )
            .map_err(|x| {
                JsError::new(format!("Error while creating audio generation options: {x}").as_ref())
            })?;
        let req = rig::audio_generation::AudioGenerationRequest::from(req);
        let res = self
            .0
            .audi(req)
            .await
            .map_err(|x| JsError::new(format!("Error while creating audio: {x}").as_ref()))?;

        Ok(OpenAIAudioGenerationResponse(res))
    }
}
{% endif %}
"#;
