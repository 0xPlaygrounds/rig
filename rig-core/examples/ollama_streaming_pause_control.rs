use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::completion::GetTokenUsage;
use rig::prelude::*;
use rig::providers::ollama;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Use the direct model streaming to get access to pause control
    let model = ollama::Client::new().completion_model("gemma3:4b");
    let completion_request = model
        .completion_request("Explain backpropagation in neural networks.")
        .preamble("You are a helpful AI assistant. Provide concise explanations.".to_string())
        .temperature(0.7)
        .build();
    let mut stream = model.stream(completion_request).await?;

    let mut chunk_count = 0;

    // Process the stream with pause control
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(content) => {
                match content {
                    rig::streaming::StreamedAssistantContent::Text(text) => {
                        print!("{}", text.text);
                        std::io::Write::flush(&mut std::io::stdout())?;
                        chunk_count += 1;
                    }
                    rig::streaming::StreamedAssistantContent::ToolCall(tool_call) => {
                        println!("\n[Tool Call: {}]", tool_call.function.name);
                        chunk_count += 1;
                    }
                    rig::streaming::StreamedAssistantContent::Reasoning(reasoning) => {
                        println!("\n[Reasoning: {}]", reasoning.reasoning.join(""));
                        chunk_count += 1;
                    }
                    rig::streaming::StreamedAssistantContent::Final(response) => {
                        println!("\n\n[Stream completed]");
                        if let Some(usage) = response.token_usage() {
                            println!("Token usage: {:?}", usage);
                        }
                        break;
                    }
                }

                // Demonstrate pause control every 10 chunks
                if chunk_count % 50 == 0 && chunk_count > 0 {
                    println!("\n\n[Pausing stream for 2 seconds...]");
                    stream.pause();

                    // Wait for 2 seconds while paused
                    sleep(Duration::from_secs(2)).await;

                    println!("[Resuming stream...]");
                    stream.resume();
                }
            }
            Err(e) => {
                if e.to_string().contains("aborted") {
                    println!("\n[Stream cancelled]");
                    break;
                }
                eprintln!("Error: {}", e);
                break;
            }
        }
    }

    println!("\n\nStream processing completed!");
    println!("Total chunks processed: {}", chunk_count);

    Ok(())
}
