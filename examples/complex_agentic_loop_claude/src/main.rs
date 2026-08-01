//! A complex agentic loop with Claude: an orchestrator agent that delegates to
//! specialized sub-agents, a knowledge-base tool, and the built-in think tool.
//!
//! Each sub-agent is exposed to the orchestrator as a [`PortableDynamicTool`]
//! whose callback closes over a clone of the agent (`Agent` is `Clone`) and
//! forwards the prompt to it — see [`agent_as_tool`].
use anyhow::Result;
use rig::OneOrMany;
use rig::agent::Agent;
use rig::http_runtime::HttpRuntime;
use rig::prelude::*;
use rig::providers::{anthropic, openai};
use rig::tool::{PortableDynamicTool, ToolExecutionError, ToolOutput};
use rig::vector_store::{SearchHit, VectorSearchRequest, VectorStoreError};
use rig::{
    Embed, embeddings::EmbeddingJob, message::Message, tool::builtin::ThinkTool,
    vector_store::in_memory_store::InMemoryVectorStore,
};
use serde::{Deserialize, Serialize};

/// A custom tool exposing the knowledge base to the agent. It embeds the
/// model's query, then runs a pre-embedded search against the store.
///
/// Embedding is plain config data plus a free function, so the tool holds an
/// `EmbeddingConfig` and an `HttpRuntime` rather than a model handle.
struct KnowledgeBaseTool {
    store: InMemoryVectorStore,
    embedding_config: openai::functions::EmbeddingConfig,
    rt: HttpRuntime,
}

#[derive(Deserialize, Serialize)]
struct KnowledgeBaseArgs {
    query: String,
}

impl rig::tool::PortableTool for KnowledgeBaseTool {
    const NAME: &'static str = "search_knowledge_base";
    type Args = KnowledgeBaseArgs;
    type Output = Vec<SearchHit>;
    type Error = VectorStoreError;

    fn description(&self) -> String {
        "Retrieves the most relevant documents from the sustainability knowledge base.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query string to search for relevant documents."
                }
            },
            "required": ["query"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let response =
            openai::functions::embed(&self.embedding_config, &self.rt, vec![args.query]).await?;
        let Some(query_embedding) = response.embeddings.into_iter().next() else {
            return Err(VectorStoreError::DatastoreError(
                "the embedding provider returned no embedding".into(),
            ));
        };
        let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 3);
        self.store.top_n(req).await
    }
}

/// Expose a sub-agent as a dynamic tool: the callback closes over a clone of
/// the agent and forwards the model-provided prompt to it.
fn agent_as_tool(agent: &Agent, name: &str, description: &str) -> PortableDynamicTool {
    let inner = agent.clone();
    PortableDynamicTool::new(
        name,
        description,
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The question or task for the sub-agent"
                }
            },
            "required": ["prompt"]
        }),
        move |args| {
            let inner = inner.clone();
            Box::pin(async move {
                let prompt = args
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let reply = inner
                    .prompt(prompt)
                    .await
                    .map_err(|e| ToolExecutionError::other(e.to_string()))?;
                Ok(ToolOutput::text(reply))
            })
        },
    )
}

// Define a knowledge base entry for our vector store
#[derive(Embed, Clone, Deserialize, Debug, Serialize, Eq, PartialEq, Default)]
struct KnowledgeEntry {
    id: String,
    title: String,
    #[embed]
    content: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Set up logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    let claude = anthropic::Client::from_env()?;

    // Embedding config for our vector store — OpenAI's embedding model here.
    let openai = openai::Client::from_env()?;
    let rt = openai.http();
    let embedding_config = openai.embedding_config(openai::TEXT_EMBEDDING_ADA_002);

    // Create a knowledge base with sample entries
    let knowledge_entries = vec![
        KnowledgeEntry {
            id: "kb1".to_string(),
            title: "Climate Change Effects".to_string(),
            content: "Climate change is causing rising sea levels, increased frequency of extreme weather events, \
                     and disruptions to ecosystems worldwide. The IPCC has projected that global temperatures \
                     could rise by 1.5°C to 4.5°C by 2100, depending on emission scenarios.".to_string(),
        },
        KnowledgeEntry {
            id: "kb2".to_string(),
            title: "Renewable Energy Technologies".to_string(),
            content: "Solar photovoltaic technology converts sunlight directly into electricity using semiconductor materials. \
                     Wind turbines convert kinetic energy from wind into mechanical power, which generators then convert to electricity. \
                     Hydroelectric power generates electricity by using flowing water to turn turbines connected to generators.".to_string(),
        },
        KnowledgeEntry {
            id: "kb3".to_string(),
            title: "Sustainable Agriculture Practices".to_string(),
            content: "Crop rotation improves soil health by alternating different crops in the same area across seasons. \
                     Agroforestry integrates trees with crop or livestock systems, enhancing biodiversity and resilience. \
                     Precision agriculture uses technology to optimize field-level management, reducing resource use while maximizing yields.".to_string(),
        },
        KnowledgeEntry {
            id: "kb4".to_string(),
            title: "Carbon Capture Methods".to_string(),
            content: "Direct air capture (DAC) extracts CO2 directly from the atmosphere using chemical processes. \
                     Bioenergy with carbon capture and storage (BECCS) combines biomass energy with geological CO2 storage. \
                     Enhanced weathering accelerates natural geological processes that remove CO2 from the atmosphere.".to_string(),
        },
    ];

    // Create embeddings for our knowledge base
    let embeddings = EmbeddingJob::new()
        .documents(knowledge_entries)
        .for_provider(&openai::functions::DESCRIPTOR)
        .run(|texts| openai::functions::embed(&embedding_config, &rt, texts))
        .await?;

    // Create vector store with the embeddings
    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |entry| entry.id.clone())?;

    // Expose the knowledge base as a custom tool
    let knowledge_base = KnowledgeBaseTool {
        store: vector_store,
        embedding_config,
        rt,
    };

    // Create specialized research agent that will be used as a tool
    let research_agent = claude
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble(
            "You are a specialized research agent focused on environmental science and sustainability.
            Your role is to provide detailed, accurate information about climate change, renewable energy,
            sustainable practices, and related topics. Always cite your sources when possible and
            maintain scientific accuracy in your responses."
        )
        .name("research_agent")
        .build();

    // Create a data analysis agent that will be used as a tool
    let analysis_agent = claude
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble(
            "You are a data analysis agent specialized in interpreting environmental and sustainability data.
            When given data or statistics, you analyze trends, identify patterns, and draw meaningful conclusions.
            You're skilled at explaining complex data in accessible terms while maintaining scientific accuracy.
            Always note limitations in the data and avoid overextending conclusions beyond what the evidence supports."
        )
        .name("data_analysis_agent")
        .build();

    // Create a recommendation agent that will be used as a tool
    let recommendation_agent = claude
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble(
            "You are a recommendation agent specialized in suggesting practical sustainability solutions.
            Based on research findings and analysis, you provide actionable recommendations for individuals,
            organizations, or policymakers. Your suggestions should be specific, feasible, and tailored to
            the context. Consider factors like cost, implementation difficulty, and potential impact when
            making recommendations."
        )
        .name("recommendation_agent")
        .build();

    // Create the main orchestrator agent that will use all the tools
    let orchestrator_agent = claude
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble(
            "You are an environmental sustainability advisor that helps users understand complex environmental issues
            and find practical solutions. You have access to several specialized tools:

            1. A knowledge base with information on climate change, renewable energy, sustainable agriculture, and carbon capture.
            2. A research agent that can provide detailed information on environmental science topics.
            3. A data analysis agent that can interpret environmental data and statistics.
            4. A recommendation agent that can suggest practical sustainability solutions.
            5. A think tool that allows you to reason through complex problems step by step.

            Your workflow:
            1. Use the knowledge base to retrieve relevant background information
            2. Use the research agent to gather detailed information on specific topics
            3. Use the data analysis agent to interpret any data or statistics
            4. Use the think tool to reason through the problem and plan your approach
            5. Use the recommendation agent to generate practical solutions

            Combine these tools effectively to provide comprehensive, accurate, and actionable advice on
            environmental sustainability issues."
        )
        .tool(ThinkTool)
        .tool(knowledge_base)
        .dynamic_tool(agent_as_tool(
            &research_agent,
            "research_agent",
            "Delegate detailed environmental science research questions to a specialized research agent.",
        ))
        .dynamic_tool(agent_as_tool(
            &analysis_agent,
            "data_analysis_agent",
            "Delegate interpretation of environmental data and statistics to a specialized analysis agent.",
        ))
        .dynamic_tool(agent_as_tool(
            &recommendation_agent,
            "recommendation_agent",
            "Delegate generation of practical sustainability recommendations to a specialized agent.",
        ))
        .name("orchestrator_agent")
        .build();

    println!("=== Complex Agentic Loop with Claude ===");
    println!("This example demonstrates a complex agentic loop using Claude with:");
    println!("- Multiple specialized agents used as tools");
    println!("- Vector store for knowledge retrieval");
    println!("- Think tool for complex reasoning");
    println!();

    // Example query that will exercise the complex agentic loop
    let query = "I'm a small business owner looking to reduce my company's carbon footprint. \
                We have 25 employees in a 5000 sq ft office space and a small fleet of 5 delivery vehicles. \
                What are the most cost-effective sustainability measures we could implement in the next 6-12 months? Try to stay concise.";

    println!("Query: {}", query);
    println!("\nProcessing...\n");

    // Send the query to the orchestrator agent with extended details to get chat history
    let empty_history: Vec<Message> = Vec::new();
    let response = orchestrator_agent
        .runner(query)
        .history(empty_history)
        .max_turns(15) // Allow multiple turns to demonstrate the complex loop
        .run()
        .await?;

    // Print the final response
    println!("\nFinal Response:\n{}", response.output);

    // Print the chat history to show the agentic loop
    println!("\nAgentic Loop Details:");
    if let Some(messages) = &response.messages {
        for (i, message) in messages.clone().into_iter().enumerate() {
            match message {
                Message::User { content } => println!(
                    "\nUser [{}]: {}",
                    i,
                    serde_json::to_string_pretty(&content)?
                ),
                Message::Assistant { content, .. } => println!(
                    "Assistant [{}]: {}",
                    i,
                    serde_json::to_string_pretty(&content)?
                ),
                _ => {
                    // Ignore other message types - the only other type of message that exists is system messages
                    // which can be ignored
                }
            }
        }
    }

    Ok(())
}
