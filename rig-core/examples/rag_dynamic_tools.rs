use anyhow::Result;
use rig::{
    completion::{Prompt, ToolDefinition},
    embeddings::EmbeddingsBuilder,
    providers::openai::Client,
    tool::{Tool, ToolEmbedding, ToolSet},
    vector_store::{in_memory_store::InMemoryVectorStore, VectorStore},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Deserialize, Serialize)]
struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "add",
            "description": "Add x and y together",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: String) -> Result<String> {
        let args: OperationArgs = serde_json::from_str(&args)?;
        let result = args.x + args.y;
        Ok(format!("{result}"))
    }
}

impl ToolEmbedding for Add {
    type Context = ();
    type State = ();

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self> {
        Ok(Add)
    }

    fn embedding_docs(&self) -> Vec<String> {
        vec!["Add x and y together".into()]
    }

    fn context(&self) -> Self::Context {}
}

#[derive(Deserialize, Serialize)]
struct Subtract;

impl Tool for Subtract {
    const NAME: &'static str = "subtract";

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "subtract",
            "description": "Subtract y from x (i.e.: x - y)",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The number to substract from"
                    },
                    "y": {
                        "type": "number",
                        "description": "The number to substract"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: String) -> Result<String> {
        let args: OperationArgs = serde_json::from_str(&args)?;
        let result = args.x - args.y;
        Ok(format!("{result}"))
    }
}

impl ToolEmbedding for Subtract {
    type Context = ();
    type State = ();

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self> {
        Ok(Subtract)
    }

    fn context(&self) -> Self::Context {}

    fn embedding_docs(&self) -> Vec<String> {
        vec!["Subtract y from x (i.e.: x - y)".into()]
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // required to enable CloudWatch error logging by the runtime
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        // disable printing the name of the module in every log line.
        .with_target(false)
        // this needs to be set to false, otherwise ANSI color codes will
        // show up in a confusing manner in CloudWatch logs.
        .with_ansi(false)
        // disabling time is handy because CloudWatch will add the ingestion time.
        .without_time()
        .init();

    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let embedding_model = openai_client.embedding_model("text-embedding-ada-002");

    // Create vector store, compute tool embeddings and load them in the store
    let mut vector_store = InMemoryVectorStore::default();

    let toolset = ToolSet::builder()
        .dynamic_tool(Add)
        .dynamic_tool(Subtract)
        .build();

    // for (toolname, _) in toolset.tools.iter() {
    //     tracing::info!("Toolset: {}", toolname);
    // }

    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .tools(&toolset)?
        .build()
        .await?;

    // for doc in embeddings.iter() {
    //     tracing::info!("Embeddings: {}", doc.id);
    // }

    vector_store.add_documents(embeddings).await?;

    // Create vector store index
    let index = vector_store.index(embedding_model);

    // Create RAG agent with a single context prompt and a dynamic tool source
    let calculator_rag = openai_client
        .tool_rag_agent("gpt-4")
        .preamble("You are a calculator here to help the user perform arithmetic operations.")
        // Add a dynamic tool source with a sample rate of 1 (i.e.: only
        // 1 additional tool will be added to prompts)
        .dynamic_tools(1, index, toolset)
        .build();

    // Prompt the agent and print the response
    let response = calculator_rag.prompt("Calculate 3 - 7", vec![]).await?;
    println!("{}", response);

    Ok(())
}
