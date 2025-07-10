use std::fmt;
use std::path::Path;

use tera::{Context, Tera};

const CODE: &str = r#"
#[wasm_bindgen]
#[derive(Clone)]
pub struct {{provider}}Client(rig::providers::{{module_name}}::Client);

#[wasm_bindgen]
impl {{provider}}Client {
    #[wasm_bindgen(constructor)]
    pub fn new(api_key: &str) -> Self {
        Self(rig::providers::{{module_name}}::Client::new(api_key))
    }

    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self(rig::providers::{{module_name}}::Client::from_url(api_key, base_url))
    }

    pub fn completion_model(&self, model_name: &str) -> {{provider}}CompletionModel {
        {{provider}}CompletionModel::new(self, model_name)
    }

    pub fn agent(&self, model_name: &str) -> {{provider}}AgentBuilder {
        {{provider}}AgentBuilder::new(self, model_name)
    }

    pub fn embedding_model(&self, model_name: &str) -> {{provider}}EmbeddingModel {
        {{provider}}EmbeddingModel::new(self, model_name)
    }
}

"#;

#[derive(serde::Serialize)]
enum Provider {
    OpenAI,
    Anthropic,
    Azure,
    Groq,
    DeepSeek,
    Galadriel,
    Hyperbolic,
    Mira,
    Moonshot,
    Ollama,
    Perplexity,
    VoyageAI,
}

impl TryFrom<&str> for Provider {
    type Error = String;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let res = match value {
            "openai" => Self::OpenAI,
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
            Self::Perplexity => write!(f, "Perplexity"),
            Self::VoyageAI => write!(f, "VoyageAI"),
        }
    }
}

fn main() {
    let input_dir = Path::new("../rig-core/src/providers");
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

            if file_contents.contains("impl CompletionClient for Client") {
                let mut context = Context::new();
                context.insert("provider", &provider_name);
                context.insert("module_name", &filename);
                let file_contents = tera
                    .render_str(CODE, &context)
                    .expect("This shouldn't fail!");
                std::fs::write(out_dir.join(entry.file_name()), file_contents).unwrap();
            }
        }
    }
}
