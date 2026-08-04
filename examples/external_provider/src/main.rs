//! Implements a typed out-of-tree completion provider, erases it into a
//! concrete runtime record, and uses ordinary high-level agent APIs.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::Result;
use rig::{
    AgentBuilder, OneOrMany,
    completion::{CompletionError, CompletionRequest, CompletionResponse, Usage},
    http_runtime::HttpRuntime,
    message::AssistantContent,
    provider::{
        ExternalCompletionProvider, ExternalCompletionProviderEntry, ExternalProviderConfig,
        ExternalProviderId, OwnedProviderDescriptor, Runtime,
    },
    streaming::{CompletionStream, RawStreamingChoice, StreamFinal},
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct LocalConfig {
    prefix: String,
}

struct LocalProvider {
    id: ExternalProviderId,
    calls: Arc<AtomicUsize>,
}

impl LocalProvider {
    fn new(id: ExternalProviderId) -> Self {
        Self {
            id,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl ExternalCompletionProvider for LocalProvider {
    type Config = LocalConfig;

    fn id(&self) -> ExternalProviderId {
        self.id.clone()
    }

    fn descriptor(&self) -> OwnedProviderDescriptor {
        OwnedProviderDescriptor::named("local-example")
            .with_tools(true)
            .with_response_format(true)
            .with_composes_native_output_with_tools(true)
    }

    async fn complete(
        &self,
        config: Self::Config,
        model: String,
        _runtime: HttpRuntime,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(CompletionResponse::new(
            OneOrMany::one(AssistantContent::text(format!(
                "{} unary response #{call}",
                config.prefix
            ))),
            Usage::new(),
            "local-example",
        )
        .with_model(model))
    }

    async fn open_stream(
        &self,
        config: Self::Config,
        model: String,
        _runtime: HttpRuntime,
        _request: CompletionRequest,
    ) -> Result<CompletionStream, CompletionError> {
        Ok(CompletionStream::from_stream(futures::stream::iter([
            Ok(RawStreamingChoice::Message(format!(
                "{} streamed response",
                config.prefix
            ))),
            Ok(RawStreamingChoice::FinalResponse(
                StreamFinal::new("local-example", Usage::new()).with_model(model),
            )),
        ])))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let id = ExternalProviderId::new("dev.rig.examples/local@1")?;
    let config = ExternalProviderConfig::new(
        id.clone(),
        "local-model",
        LocalConfig {
            prefix: "hello".to_owned(),
        },
    )?;

    // Only configuration serializes. A restored process explicitly rebuilds
    // and registers its executable provider implementation.
    let serialized = serde_json::to_string(&config)?;
    let restored: ExternalProviderConfig = serde_json::from_str(&serialized)?;
    let entry = ExternalCompletionProviderEntry::from_provider(LocalProvider::new(id));
    let runtime = Arc::new(Runtime::new().with_external_provider(entry)?);

    let agent = AgentBuilder::new(restored).runtime(runtime).build();
    println!("{}", agent.prompt("unary").await?);

    let response = agent.stream_run("streaming").into_final_response().await?;
    println!("{}", response.output);
    Ok(())
}
