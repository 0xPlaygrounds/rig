use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    agent::AgentBuilder, extractor::ExtractorBuilder,
    providers::inception::completion::CompletionModel,
};

const INCEPTION_API_BASE_URL: &str = "https://api.inceptionlabs.ai/v1";

#[derive(Clone)]
pub struct ClientBuilder<'a> {
    api_key: &'a str,
    base_url: &'a str,
}

impl<'a> ClientBuilder<'a> {
    pub fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            base_url: INCEPTION_API_BASE_URL,
        }
    }

    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn build(self) -> Client {
        Client::new(self.api_key, self.base_url)
    }
}

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    pub fn new(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        "Content-Type",
                        "application/json"
                            .parse()
                            .expect("Content-Type should parse"),
                    );
                    headers.insert(
                        "Authorization",
                        format!("Bearer {}", api_key)
                            .parse()
                            .expect("Authorization should parse"),
                    );
                    headers
                })
                .build()
                .expect("Inception reqwest client should build"),
        }
    }

    pub fn from_env() -> Self {
        let api_key = std::env::var("INCEPTION_API_KEY").expect("INCEPTION_API_KEY not set");
        ClientBuilder::new(&api_key).build()
    }

    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    pub fn agent(&self, model: &str) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }
}
