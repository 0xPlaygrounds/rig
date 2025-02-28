use crate::{client::ClientBuilder, completion::BedrockProvider};
use rig::providers::anthropic::client::ClientBuilder as AnthropicClientBuilder;
use rig::providers::anthropic::completion::CompletionModel;

pub struct AnthropicBedrockProviderBuilder<'a> {
    client_builder: ClientBuilder<'a>,
}

impl<'a> AnthropicBedrockProviderBuilder<'a> {
    pub fn new() -> Self {
        let client_builder = ClientBuilder::new()
            .additional_fields(vec![
                serde_json::json!({
                    "anthropic_version": "bedrock-2023-05-31"
                }),
                serde_json::json!({
                    "anthropic_beta": Vec::<String>::new()
                }),
            ])
            .deletable_fields(vec!["model".to_string()]);
        Self { client_builder }
    }

    pub fn add_beta_feature(mut self, beta_feature: String) -> Self {
        for field in &mut self.client_builder.additional_fields {
            if let serde_json::Value::Object(map) = field {
                if let Some(serde_json::Value::Array(beta_array)) = map.get_mut("anthropic_beta") {
                    beta_array.push(serde_json::json!(beta_feature));
                    return self;
                }
            }
        }
        self
    }

    pub fn add_additional_field(mut self, additional_field: serde_json::Value) -> Self {
        self.client_builder.add_additional_field(additional_field);
        self
    }

    pub fn deletable_fields(mut self, deletable_fields: Vec<String>) -> Self {
        let client_builder = self.client_builder.deletable_fields(deletable_fields);
        self.client_builder = client_builder;
        self
    }

    pub async fn build(self) -> crate::client::Client {
        self.client_builder.build().await
    }

    pub fn anthropic_completion_model() -> CompletionModel {
        let anthropic_client = AnthropicClientBuilder::new("").build();
        let completion_model = anthropic_client.completion_model(&String::default());
        completion_model
    }
}
