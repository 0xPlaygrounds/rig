use rig_bevy::{LocalRuntime, TenantId};
use rig_core::{
    client::{CompletionClient, ProviderClient},
    completion::CompletionModel,
    providers::openai,
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = openai::Client::from_env()?;
    let model = client.completion_model(openai::GPT_5_2);
    let request = model
        .completion_request("Explain owned ECS effects in one sentence.")
        .build();
    let mut runtime = LocalRuntime::new(model, TenantId::new());
    let result = runtime.run(request, 1).await?;
    println!("{:?}", result.snapshot.output);
    Ok(())
}
