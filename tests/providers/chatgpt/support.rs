use assert_fs::TempDir;
use rig::AgentBuilder;
use rig::completion::{CompletionError, CompletionRequest, CompletionResponse};
use rig::http_runtime::HttpRuntime;
use rig::provider::ProviderConfig;
use rig::providers::chatgpt;
use rig::streaming::StreamingCompletionResponse;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

/// Connection details for a running ChatGPT cassette proxy.
///
/// Replaces the deleted `chatgpt::Client`: tests mint a plain
/// [`chatgpt::functions::Config`] (or an [`AgentBuilder`]) per model, pointed
/// at the cassette's base URL and carrying the recorded access token,
/// account id and default instructions.
pub(super) struct ChatGptCassette {
    access_token: String,
    account_id: String,
    base_url: String,
    default_instructions: String,
}

#[allow(dead_code)]
impl ChatGptCassette {
    /// Completion config for `model` aimed at the cassette proxy.
    pub(crate) fn config(&self, model: impl Into<String>) -> chatgpt::functions::Config {
        let mut cfg = chatgpt::functions::Config::new(model)
            .with_access_token(self.access_token.clone())
            .with_account_id(self.account_id.clone())
            .with_base_url(self.base_url.clone());
        cfg.default_instructions = Some(self.default_instructions.clone());
        cfg
    }

    /// An [`AgentBuilder`] for `model` aimed at the cassette proxy.
    pub(crate) fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(ProviderConfig::ChatGpt(self.config(model)))
    }

    /// A real-HTTP runtime — the cassette proxy is a live local server.
    pub(crate) fn http(&self) -> HttpRuntime {
        HttpRuntime::new()
    }

    /// Stand-in for the deleted `client.completion_model(model)`: a config
    /// plus runtime pair exposing `completion`/`stream`.
    pub(crate) fn completion_model(&self, model: impl Into<String>) -> ChatGptCassetteModel {
        ChatGptCassetteModel {
            cfg: self.config(model),
            rt: self.http(),
        }
    }
}

/// The deleted `ChatGPTCompletionModel`, re-expressed as config + runtime.
pub(super) struct ChatGptCassetteModel {
    cfg: chatgpt::functions::Config,
    rt: HttpRuntime,
}

#[allow(dead_code)]
impl ChatGptCassetteModel {
    pub(crate) fn with_strict_tools(mut self) -> Self {
        self.cfg = self.cfg.with_strict_tools();
        self
    }

    pub(crate) async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        chatgpt::functions::complete(&self.cfg, &self.rt, request).await
    }

    pub(crate) async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        chatgpt::functions::open_stream(&self.cfg, &self.rt, request).await
    }
}

async fn chatgpt_cassette_with_default_instructions(
    spec: impl Into<CassetteSpec>,
    default_instructions: impl Into<String>,
) -> (ProviderCassette, ChatGptCassette) {
    let cassette =
        ProviderCassette::start("chatgpt", spec, "https://chatgpt.com/backend-api/codex").await;
    let handle = ChatGptCassette {
        access_token: cassette.api_key("CHATGPT_ACCESS_TOKEN"),
        account_id: cassette.api_key("CHATGPT_ACCOUNT_ID"),
        base_url: cassette.base_url(),
        default_instructions: default_instructions.into(),
    };

    (cassette, handle)
}

async fn chatgpt_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, ChatGptCassette) {
    chatgpt_cassette_with_default_instructions(spec, "").await
}

async fn chatgpt_noninteractive_oauth_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, ChatGptCassette, TempDir) {
    let cassette =
        ProviderCassette::start("chatgpt", spec, "https://chatgpt.com/backend-api/codex").await;
    let temp = TempDir::new().expect("temp auth directory should be created");
    let auth_file = temp.path().join("auth.json");
    let record = serde_json::json!({
        "access_token": cassette.api_key("CHATGPT_ACCESS_TOKEN"),
        "refresh_token": serde_json::Value::Null,
        "id_token": serde_json::Value::Null,
        "expires_at": i64::MAX,
        "account_id": cassette.api_key("CHATGPT_ACCOUNT_ID"),
    });
    std::fs::write(
        &auth_file,
        serde_json::to_vec_pretty(&record).expect("auth record should serialize"),
    )
    .expect("auth record should be written");

    // The client's `oauth().allow_device_flow(false).auth_file(..)` path is
    // now a construction-time credential resolution through `Authenticator`.
    let authenticator = chatgpt::auth::Authenticator::new(
        chatgpt::auth::AuthSource::OAuth,
        Some(auth_file),
        chatgpt::auth::DeviceCodeHandler::default(),
        false,
    );
    let auth = authenticator
        .auth_context()
        .await
        .expect("cached OAuth auth should not require device flow");

    let handle = ChatGptCassette {
        access_token: auth.access_token,
        account_id: auth.account_id.unwrap_or_default(),
        base_url: cassette.base_url(),
        default_instructions: String::new(),
    };

    (cassette, handle, temp)
}

pub(super) async fn with_chatgpt_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(ChatGptCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, handle) = chatgpt_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(handle)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_chatgpt_cassette_default_instructions<F, Fut>(
    spec: impl Into<CassetteSpec>,
    default_instructions: impl Into<String>,
    test_body: F,
) where
    F: FnOnce(ChatGptCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, handle) =
        chatgpt_cassette_with_default_instructions(spec, default_instructions).await;
    let result = AssertUnwindSafe(test_body(handle)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_chatgpt_noninteractive_oauth_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(ChatGptCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, handle, _temp) = chatgpt_noninteractive_oauth_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(handle)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
