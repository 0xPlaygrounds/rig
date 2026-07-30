//! This example shows how you can use OpenAI's Completions API.
//! By default, the OpenAI integration uses the Responses API. However, for the sake of backwards compatibility you may wish to use the Completions API.
//!
//! Choosing the API face is now a choice of `ProviderConfig` arm over plain
//! data: `ProviderConfig::OpenAi` speaks chat completions (this example),
//! while `ProviderConfig::OpenAiResponses` speaks the Responses API.

use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use rig::prelude::*;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::SdkTracerProvider;
use rig::providers::openai;
use tracing::Level;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_protocol(opentelemetry_otlp::Protocol::HttpBinary)
        .build()?;
    // Create a new OpenTelemetry trace pipeline that prints to stdout
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(Resource::builder().with_service_name("rig-demo").build())
        .build();
    let tracer = provider.tracer("readme_example");

    // Create a tracing layer with the configured tracer
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    let filter_layer = tracing_subscriber::filter::EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy();

    let fmt_layer = tracing_subscriber::fmt::layer().compact();

    // Use the tracing subscriber `Registry`, or any other subscriber
    // that impls `LookupSpan`
    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .with(otel_layer)
        .init();

    // The chat-completions face is `ProviderConfig::OpenAi`; the Responses
    // API face would be `ProviderConfig::OpenAiResponses`.
    let cfg = openai::functions::Config::from_env(openai::GPT_4O)?;
    let agent = AgentBuilder::new(ProviderConfig::OpenAi(cfg))
        .preamble("You are a helpful assistant")
        .build();

    let res = agent.prompt("Hello world!").await?;

    println!("GPT-4o: {res}");

    let _ = provider.shutdown();

    Ok(())
}
