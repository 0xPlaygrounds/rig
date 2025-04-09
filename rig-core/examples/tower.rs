use rig::{
    completion::CompletionRequestBuilder,
    middlewares::{
        completion::{CompletionLayer, CompletionService},
        tools::ToolLayer,
    },
    providers::openai::Client,
};
use tower::ServiceBuilder;

#[tokio::main]
async fn main() {
    let client = Client::from_env();
    let model = client.completion_model("gpt-4o");

    let comp_layer = CompletionLayer::builder(model).build();
    let tool_layer = ToolLayer::new(vec![Add]);
    let service = CompletionService::new(model);

    let service = ServiceBuilder::new()
        .layer(comp_layer)
        .layer(tool_layer)
        .service(service);

    let comp_request = CompletionRequestBuilder::new(model, "Please calculate 5+5 for me").build();

    let res = service.call(comp_request).await.unwrap();

    println!("{res:?}");
}

#[derive(Deserialize, Serialize)]
struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";

    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

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

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x + args.y;
        Ok(result)
    }
}
