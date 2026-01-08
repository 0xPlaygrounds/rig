use anyhow::Result;
use futures::StreamExt;
use rig::OneOrMany;
use rig::completion::{CompletionModel, GetTokenUsage, ToolDefinition};
use rig::message::{AssistantContent, Message, ToolCall, ToolChoice};
use rig::prelude::*;
use rig::providers::gemini::{
    self,
    interactions_api::{AdditionalParameters, Tool},
};
use rig::streaming::StreamedAssistantContent;
use serde_json::json;
use std::io::Write;
use tracing_subscriber::EnvFilter;

fn extract_text(choice: &OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn first_tool_call(choice: &OneOrMany<AssistantContent>) -> Option<ToolCall> {
    choice.iter().find_map(|content| match content {
        AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
        _ => None,
    })
}

fn print_text(label: &str, choice: &OneOrMany<AssistantContent>) {
    let text = extract_text(choice);
    if text.is_empty() {
        println!("{label}: [non-text response]");
    } else {
        println!("{label}: {text}");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let client = gemini::Client::from_env().interactions_api();
    let model = client.completion_model("gemini-3-flash-preview");

    println!("== Basic interaction ==");
    let basic_params = AdditionalParameters {
        store: Some(true),
        ..Default::default()
    };
    let basic_request = model
        .completion_request("Give me two fun facts about hummingbirds.")
        .preamble("Be concise.".to_string())
        .additional_params(serde_json::to_value(&basic_params)?)
        .build();
    let basic_response = model.completion(basic_request).await?;
    print_text("Basic response", &basic_response.choice);

    let interaction_id = basic_response.raw_response.id.clone();
    if interaction_id.is_empty() {
        println!("No interaction id returned; skipping follow-up.");
    } else {
        println!("\n== Continue with previous_interaction_id ==");
        let follow_params = AdditionalParameters {
            previous_interaction_id: Some(interaction_id),
            ..Default::default()
        };
        let follow_request = model
            .completion_request("Now answer with a short analogy.")
            .additional_params(serde_json::to_value(&follow_params)?)
            .build();
        let follow_response = model.completion(follow_request).await?;
        print_text("Follow-up", &follow_response.choice);
    }

    println!("\n== Google Search tool ==");
    let search_params = AdditionalParameters {
        tools: Some(vec![Tool::GoogleSearch]),
        ..Default::default()
    };
    let search_request = model
        .completion_request("What is the most recent country the US stole the leader of?")
        .additional_params(serde_json::to_value(&search_params)?)
        .build();
    let search_response = model.completion(search_request).await?;
    print_text("Search response", &search_response.choice);

    if let Some(cited) = search_response.raw_response.text_with_inline_citations() {
        println!("Search response with citations: {cited}");
    }

    let exchanges = search_response.raw_response.google_search_exchanges();
    if exchanges.is_empty() {
        println!("No search tool outputs returned.");
    } else {
        for exchange in exchanges {
            let exchange_label = exchange
                .call_id
                .as_deref()
                .map(|id| format!(" ({id})"))
                .unwrap_or_default();
            println!("Search exchange{exchange_label}:");

            let queries = exchange.queries();
            if !queries.is_empty() {
                println!("Queries: {}", queries.join(", "));
            }

            let results = exchange.result_items();
            if results.is_empty() {
                println!("No search results returned.");
            } else {
                for (idx, result) in results.iter().enumerate() {
                    let title = result.title.as_deref().unwrap_or("Untitled");
                    let url = result.url.as_deref().unwrap_or("");
                    if url.is_empty() {
                        println!("[{}] {}", idx + 1, title);
                    } else {
                        println!("[{}] {} ({})", idx + 1, title, url);
                    }
                }
            }
        }
    }

    println!("\n== URL Context tool ==");
    let url_params = AdditionalParameters {
        tools: Some(vec![Tool::UrlContext]),
        ..Default::default()
    };
    let url1 = "https://www.rust-lang.org/";
    let url2 = "https://doc.rust-lang.org/book/";
    let url_prompt =
        format!("Compare the focus of the pages at {url1} and {url2}. Provide a concise summary.");
    let url_request = model
        .completion_request(url_prompt)
        .additional_params(serde_json::to_value(&url_params)?)
        .build();
    let url_response = model.completion(url_request).await?;
    print_text("URL context response", &url_response.choice);

    let url_exchanges = url_response.raw_response.url_context_exchanges();
    if url_exchanges.is_empty() {
        println!("No URL context tool outputs returned.");
    } else {
        for exchange in url_exchanges {
            let exchange_label = exchange
                .call_id
                .as_deref()
                .map(|id| format!(" ({id})"))
                .unwrap_or_default();
            println!("URL context exchange{exchange_label}:");

            let urls = exchange.urls();
            if !urls.is_empty() {
                println!("URLs: {}", urls.join(", "));
            }

            let results = exchange.result_items();
            if results.is_empty() {
                println!("No URL context results returned.");
            } else {
                for result in results {
                    let url = result.url.as_deref().unwrap_or("unknown");
                    let status = result.status.as_deref().unwrap_or("unknown");
                    println!("- {url} ({status})");
                }
            }
        }
    }

    println!("\n== Code execution tool ==");
    let code_params = AdditionalParameters {
        tools: Some(vec![Tool::CodeExecution]),
        ..Default::default()
    };
    let code_request = model
        .completion_request(
            "What is the sum of the first 50 prime numbers? Use code execution to compute it.",
        )
        .additional_params(serde_json::to_value(&code_params)?)
        .build();
    let code_response = model.completion(code_request).await?;
    print_text("Code execution response", &code_response.choice);

    let code_exchanges = code_response.raw_response.code_execution_exchanges();
    if code_exchanges.is_empty() {
        println!("No code execution tool outputs returned.");
    } else {
        for exchange in code_exchanges {
            let exchange_label = exchange
                .call_id
                .as_deref()
                .map(|id| format!(" ({id})"))
                .unwrap_or_default();
            println!("Code execution exchange{exchange_label}:");

            for snippet in exchange.code_snippets() {
                println!("Code:\n{snippet}");
            }

            for output in exchange.outputs() {
                println!("Output:\n{output}");
            }
        }
    }

    println!("\n== Tool call roundtrip ==");
    let add_tool = ToolDefinition {
        name: "add".to_string(),
        description: "Add two numbers together".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "x": { "type": "number" },
                "y": { "type": "number" }
            },
            "required": ["x", "y"]
        }),
    };
    let tool_params = AdditionalParameters {
        store: Some(true),
        ..Default::default()
    };
    let tool_request = model
        .completion_request("Use the add tool to sum 7 and 11.")
        .tool(add_tool)
        .tool_choice(ToolChoice::Required)
        .additional_params(serde_json::to_value(&tool_params)?)
        .build();
    let tool_response = model.completion(tool_request).await?;
    let tool_interaction_id = tool_response.raw_response.id.clone();

    if let Some(tool_call) = first_tool_call(&tool_response.choice) {
        println!(
            "Tool call: {}({})",
            tool_call.function.name, tool_call.function.arguments
        );
        let args = &tool_call.function.arguments;
        let x = args.get("x").and_then(|v| v.as_f64()).unwrap_or_default();
        let y = args.get("y").and_then(|v| v.as_f64()).unwrap_or_default();
        let result = json!({ "sum": x + y });

        let call_id = tool_call
            .call_id
            .clone()
            .unwrap_or_else(|| tool_call.id.clone());

        if tool_interaction_id.is_empty() {
            println!("No interaction id returned; skipping tool result.");
        } else {
            let tool_follow_params = AdditionalParameters {
                previous_interaction_id: Some(tool_interaction_id),
                ..Default::default()
            };
            let tool_follow_request = model
                .completion_request(Message::tool_result_with_call_id(
                    tool_call.function.name,
                    Some(call_id),
                    result.to_string(),
                ))
                .additional_params(serde_json::to_value(&tool_follow_params)?)
                .build();
            let tool_follow_response = model.completion(tool_follow_request).await?;
            print_text("Tool-assisted response", &tool_follow_response.choice);
        }
    } else {
        println!("No tool call returned; try a different prompt or model.");
    }

    println!("\n== Streaming ==");
    let stream_request = model
        .completion_request("Write a 3-line poem about rust and rivers.")
        .temperature(0.4)
        .build();
    let mut stream = model.stream(stream_request).await?;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(StreamedAssistantContent::Text(text)) => {
                print!("{}", text.text);
                std::io::stdout().flush()?;
            }
            Ok(StreamedAssistantContent::Final(res)) => {
                println!();
                if let Some(usage) = res.token_usage() {
                    println!("Token usage: {usage:?}");
                }
            }
            Ok(_) => {}
            Err(err) => {
                eprintln!("Error: {err}");
                break;
            }
        }
    }

    Ok(())
}
