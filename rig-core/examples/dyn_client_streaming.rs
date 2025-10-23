use futures::StreamExt;
/// This example showcases using streaming with multiple clients by using a dynamic ClientBuilder.
/// In this example, we will use both OpenAI and Anthropic with streaming responses - so ensure you have your `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` set when using this example!
use rig::{client::builder::DynClientBuilder, providers::anthropic::CLAUDE_3_7_SONNET};

#[tokio::main]
async fn main() {
    let multi_client = DynClientBuilder::new();

    // Test streaming with OpenAI
    println!("=== Testing OpenAI Streaming ===");
    match test_openai_streaming(&multi_client).await {
        Ok(_) => println!("OpenAI streaming test completed successfully"),
        Err(e) => println!("OpenAI streaming test failed: {}", e),
    }

    // Test streaming with Anthropic
    println!("\n=== Testing Anthropic Streaming ===");
    match test_anthropic_streaming(&multi_client).await {
        Ok(_) => println!("Anthropic streaming test completed successfully"),
        Err(e) => println!("Anthropic streaming test failed: {}", e),
    }

    // Test streaming with ProviderModelId
    println!("\n=== Testing ProviderModelId Streaming ===");
    match test_provider_model_id_streaming(&multi_client).await {
        Ok(_) => println!("ProviderModelId streaming test completed successfully"),
        Err(e) => println!("ProviderModelId streaming test failed: {}", e),
    }
}

async fn test_openai_streaming(
    client: &DynClientBuilder,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Streaming prompt to OpenAI (gpt-4o): 'Tell me a short story about a robot learning to paint'"
    );

    let mut stream = client
        .stream_prompt(
            "openai",
            "gpt-4o",
            "Tell me a short story about a robot learning to paint",
        )
        .await?;

    print!("Response: ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(rig::streaming::StreamedAssistantContent::Text(text)) => {
                print!("{}", text.text);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(rig::streaming::StreamedAssistantContent::Reasoning(reasoning)) => {
                println!("\n[Reasoning: {}]", reasoning.reasoning.join(""));
            }
            Ok(rig::streaming::StreamedAssistantContent::ToolCall(tool_call)) => {
                println!("\n[Tool Call: {}]", tool_call.function.name);
            }
            Ok(rig::streaming::StreamedAssistantContent::Final(_)) => {
                println!("\n[Stream completed]");
                break;
            }
            Err(e) => {
                println!("\n[Error: {}]", e);
                break;
            }
            _ => {}
        }
    }
    println!();

    Ok(())
}

async fn test_anthropic_streaming(
    client: &DynClientBuilder,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Streaming prompt to Anthropic (Claude 3.7 Sonnet): 'Explain quantum computing in simple terms'"
    );

    let mut stream = client
        .stream_prompt(
            "anthropic",
            CLAUDE_3_7_SONNET,
            "Explain quantum computing in simple terms",
        )
        .await?;

    print!("Response: ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(rig::streaming::StreamedAssistantContent::Text(text)) => {
                print!("{}", text.text);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(rig::streaming::StreamedAssistantContent::Reasoning(reasoning)) => {
                println!("\n[Reasoning: {}]", reasoning.reasoning.join(""));
            }
            Ok(rig::streaming::StreamedAssistantContent::ToolCall(tool_call)) => {
                println!("\n[Tool Call: {}]", tool_call.function.name);
            }
            Ok(rig::streaming::StreamedAssistantContent::Final(_)) => {
                println!("\n[Stream completed]");
                break;
            }
            Err(e) => {
                println!("\n[Error: {}]", e);
                break;
            }
            _ => {}
        }
    }
    println!();

    Ok(())
}

async fn test_provider_model_id_streaming(
    client: &DynClientBuilder,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Streaming prompt using ProviderModelId: 'What are the benefits of renewable energy?'"
    );

    let provider_model = client.id("openai:gpt-4o")?;
    let mut stream = provider_model
        .stream_prompt("What are the benefits of renewable energy?")
        .await?;

    print!("Response: ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(rig::streaming::StreamedAssistantContent::Text(text)) => {
                print!("{}", text.text);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(rig::streaming::StreamedAssistantContent::Reasoning(reasoning)) => {
                println!("\n[Reasoning: {}]", reasoning.reasoning.join(""));
            }
            Ok(rig::streaming::StreamedAssistantContent::ToolCall(tool_call)) => {
                println!("\n[Tool Call: {}]", tool_call.function.name);
            }
            Ok(rig::streaming::StreamedAssistantContent::Final(_)) => {
                println!("\n[Stream completed]");
                break;
            }
            Err(e) => {
                println!("\n[Error: {}]", e);
                break;
            }
            _ => {}
        }
    }
    println!();

    Ok(())
}
