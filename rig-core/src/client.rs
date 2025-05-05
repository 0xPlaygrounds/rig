use crate::agent::AgentBuilder;
#[cfg(feature = "audio")]
use crate::audio_generation::AudioGenerationModel;
use crate::completion::{Completion, CompletionModel, CompletionModelDyn};
use crate::embeddings::{EmbeddingModel, EmbeddingsBuilder};
use crate::extractor::ExtractorBuilder;
#[cfg(feature = "image")]
use crate::image_generation::ImageGenerationModel;
use crate::transcription::{TranscriptionModel, TranscriptionModelDyn};
use crate::Embed;
use as_any::AsAny;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt::Debug;

use crate::embeddings::embedding::EmbeddingModelDyn;
pub use rig_derive::ProviderClient;

pub trait ProviderClient: AsCompletion + AsTranscription + AsEmbeddings + Debug {
    fn from_env() -> Self
    where
        Self: Sized;
}

pub trait CompletionClient: ProviderClient {
    type CompletionModel: CompletionModel;

    fn completion_model(&self, model: &str) -> Self::CompletionModel;

    fn agent(&self, model: &str) -> AgentBuilder<Self::CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Create an extractor builder with the given completion model.
    fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, Self::CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

pub trait CompletionClientDyn: ProviderClient {
    fn completion_model<'a>(&'a self, model: &'a str) -> Box<dyn CompletionModelDyn + 'a>;
}

impl<T: CompletionClient> CompletionClientDyn for T {
    fn completion_model<'a>(&'a self, model: &'a str) -> Box<dyn CompletionModelDyn + 'a> {
        Box::new(self.completion_model(model))
    }
}

impl<T: CompletionClientDyn> AsCompletion for T {
    fn as_completion(&self) -> Option<Box<&dyn CompletionClientDyn>> {
        Some(Box::new(self))
    }
}

pub trait TranscriptionClient: ProviderClient {
    type TranscriptionModel: TranscriptionModel;
    fn transcription_model(&self, model: &str) -> Self::TranscriptionModel;
}

pub trait TranscriptionClientDyn: ProviderClient {
    fn transcription_model<'a>(&'a self, model: &'a str) -> Box<dyn TranscriptionModelDyn + 'a>;
}

impl<T: TranscriptionClient> TranscriptionClientDyn for T {
    fn transcription_model<'a>(&'a self, model: &'a str) -> Box<dyn TranscriptionModelDyn + 'a> {
        Box::new(self.transcription_model(model))
    }
}

impl<T: TranscriptionClientDyn> AsTranscription for T {
    fn as_transcription(&self) -> Option<Box<&dyn TranscriptionClientDyn>> {
        Some(Box::new(self))
    }
}

pub trait EmbeddingsClient: ProviderClient {
    type EmbeddingModel: EmbeddingModel;
    fn embedding_model(&self, model: &str) -> Self::EmbeddingModel;
    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::EmbeddingModel;
    fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingsBuilder<Self::EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model))
    }
}

pub trait EmbeddingsClientDyn: ProviderClient {
    fn embedding_model<'a>(&'a self, model: &'a str) -> Box<dyn EmbeddingModelDyn + 'a>;
    fn embedding_model_with_ndims<'a>(
        &'a self,
        model: &'a str,
        ndims: usize,
    ) -> Box<dyn EmbeddingModelDyn + 'a>;
}

impl<T: EmbeddingsClient> EmbeddingsClientDyn for T {
    fn embedding_model<'a>(&'a self, model: &'a str) -> Box<dyn EmbeddingModelDyn + 'a> {
        Box::new(self.embedding_model(model))
    }

    fn embedding_model_with_ndims<'a>(
        &'a self,
        model: &'a str,
        ndims: usize,
    ) -> Box<dyn EmbeddingModelDyn + 'a> {
        Box::new(self.embedding_model_with_ndims(model, ndims))
    }
}

impl<T: EmbeddingsClientDyn> AsEmbeddings for T {
    fn as_embeddings(&self) -> Option<Box<&dyn EmbeddingsClientDyn>> {
        Some(Box::new(self))
    }
}

#[cfg(feature = "image")]
pub trait ImageGenerationClient: ProviderClient {
    type ImageGenerationModel: ImageGenerationModel;
    fn image_generation_model(&self, model: &str) -> Self::ImageGenerationModel;
}

#[cfg(feature = "audio")]
pub trait AudioGenerationClient: ProviderClient {
    type AudioGenerationModel: AudioGenerationModel;
    fn audio_generation_model(&self, model: &str) -> Self::AudioGenerationModel;
}

pub trait AsCompletion {
    fn as_completion(&self) -> Option<Box<&dyn CompletionClientDyn>> {
        None
    }
}

pub trait AsTranscription {
    fn as_transcription(&self) -> Option<Box<&dyn TranscriptionClientDyn>> {
        None
    }
}

pub trait AsEmbeddings {
    fn as_embeddings(&self) -> Option<Box<&dyn EmbeddingsClientDyn>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::audio_generation::AudioGenerationModel;
    use crate::client::{
        AsAny, AudioGenerationClient, CompletionClient, CompletionClientDyn, ProviderClient,
        TranscriptionClient,
    };
    use crate::completion::{Completion, CompletionModel, ToolDefinition};
    use crate::message::AssistantContent;
    use crate::providers::{anthropic, openai};
    use crate::tool::Tool;
    use crate::transcription::TranscriptionModel;
    use as_any::Downcast;
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use std::any::Any;

    macro_rules! generate_provider_tests {
        // entry point: split into functions and providers
        (
            $( $func:ident ),* ;
            $providers:tt
        ) => {
            $(
                generate_provider_tests!(@make_mod $func; $providers);
            )*
        };

        // internal: create a module per function
        (@make_mod $func:ident; [ $( ($name:ident, $args:expr) ),* $(,)? ]) => {
            mod $func {
                use super::*;

                $(
                    #[tokio::test]
                    #[ignore]
                    async fn $name() {
                        let client = <$name::Client as ProviderClient>::from_env();
                        let _ = super::$func(client, $args).await;
                    }
                )*
            }
        };
    }

    async fn test_completions<T: CompletionClient>(client: T, model: &str) {
        let model = client.completion_model(model);
        let resp = model
            .completion_request("Whats the capital of France?")
            .send()
            .await;

        assert!(
            resp.is_ok(),
            "Error occurred when prompting, {}",
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        match resp.choice.first() {
            AssistantContent::Text(text) => {
                assert!(text.text.to_lowercase().contains("paris"));
            }
            _ => {
                assert!(
                    false,
                    "First choice wasn't a Text message, {:?}",
                    resp.choice.first()
                );
            }
        }
    }

    async fn test_tools<T: CompletionClient>(client: T, model: &str) {
        let model = client.agent(model)
            .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.")
            .max_tokens(1024)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let request = model.completion("Calculate 2 - 5", vec![]).await;

        assert!(
            request.is_ok(),
            "Error occurred when building prompt, {}",
            request.err().unwrap()
        );

        let resp = request.unwrap().send().await;

        assert!(
            resp.is_ok(),
            "Error occurred when prompting, {}",
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        assert!(
            resp.choice.iter().any(|content| match content {
                AssistantContent::ToolCall(tc) => {
                    if tc.function.name != Subtract::NAME {
                        return false;
                    }

                    let arguments =
                        serde_json::from_value::<OperationArgs>((tc.function.arguments).clone())
                            .expect("Error parsing arguments");

                    arguments.x == 2.0 && arguments.y == 5.0
                }
                _ => false,
            }),
            "Model did not use the Subtract tool."
        )
    }

    generate_provider_tests!(
        test_completions, test_tools;
        [
            (anthropic, anthropic::CLAUDE_3_5_SONNET),
            (openai, openai::GPT_4)
        ]
    );

    async fn test_transcription<
        T: AudioGenerationClient<AudioGenerationModel = M>,
        M: AudioGenerationModel,
    >(
        client: T,
        args: (&str, &str),
    ) {
        let (model, voice) = args;
        let model = client.audio_generation_model(model);

        let request = model
            .audio_generation_request()
            .text("Hello world!")
            .voice(voice);

        let resp = request.send().await;

        assert!(
            resp.is_ok(),
            "Error occurred when sending request, {}",
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        assert!(resp.audio.len() > 0, "Returned audio was empty");
    }

    generate_provider_tests!(test_transcription; [(openai, (openai::TTS_1, "onyx"))]);

    #[test]
    pub fn test_polymorphism() {
        const COMPLETION: usize = 1;
        const TRANSCRIPTION: usize = 2;
        const EMBEDDINGS: usize = 4;

        let clients: Vec<(Box<dyn ProviderClient>, usize)> = vec![
            (
                Box::new(openai::Client::from_env()),
                COMPLETION + TRANSCRIPTION + EMBEDDINGS,
            ),
            (
                Box::new(anthropic::Client::from_env()), 
                COMPLETION
            ),
        ];

        for (client, features) in clients {
            let completion = client.as_completion();
            assert_eq!(
                completion.is_some(),
                features & COMPLETION > 0,
                "Error with completion"
            );

            let transcription = client.as_transcription();
            assert_eq!(
                transcription.is_some(),
                features & TRANSCRIPTION > 0,
                "Error with transcription"
            );

            let embeddings = client.as_embeddings();
            assert_eq!(
                embeddings.is_some(),
                features & EMBEDDINGS > 0,
                "Error with embeddings"
            );
        }
    }

    #[derive(Deserialize)]
    struct OperationArgs {
        x: f32,
        y: f32,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("Math error")]
    struct MathError;

    #[derive(Deserialize, Serialize)]
    struct Adder;
    impl Tool for Adder {
        const NAME: &'static str = "add";

        type Error = MathError;
        type Args = OperationArgs;
        type Output = f32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: "add".to_string(),
                description: "Add x and y together".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The first number to add"
                        },
                        "y": {
                            "type": "number",
                            "description": "The second number to add"
                        }
                    }
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> anyhow::Result<Self::Output, Self::Error> {
            println!("[tool-call] Adding {} and {}", args.x, args.y);
            let result = args.x + args.y;
            Ok(result)
        }
    }

    #[derive(Deserialize, Serialize)]
    struct Subtract;
    impl Tool for Subtract {
        const NAME: &'static str = "subtract";

        type Error = MathError;
        type Args = OperationArgs;
        type Output = f32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            serde_json::from_value(json!({
                "name": "subtract",
                "description": "Subtract y from x (i.e.: x - y)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The number to subtract from"
                        },
                        "y": {
                            "type": "number",
                            "description": "The number to subtract"
                        }
                    }
                }
            }))
            .expect("Tool Definition")
        }

        async fn call(&self, args: Self::Args) -> anyhow::Result<Self::Output, Self::Error> {
            println!("[tool-call] Subtracting {} from {}", args.y, args.x);
            let result = args.x - args.y;
            Ok(result)
        }
    }
}
