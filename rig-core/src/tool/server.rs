use futures::{StreamExt, TryStreamExt, channel::oneshot::Canceled, stream};
use tokio::sync::mpsc::{Sender, error::SendError};

use crate::{
    completion::{CompletionError, ToolDefinition},
    tool::{Tool, ToolDyn, ToolError, ToolSet, ToolSetError},
    vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndexDyn},
};

pub struct ToolServer {
    /// A list of static tool names.
    /// These tools will always exist on the tool server for as long as they are not deleted.
    static_tool_names: Vec<String>,
    /// Dynamic tools. These tools will be dynamically fetched from a given vector store.
    dynamic_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    /// The toolset where tools are called (to be executed).
    toolset: ToolSet,
}

impl Default for ToolServer {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolServer {
    pub fn new() -> Self {
        Self {
            static_tool_names: Vec::new(),
            dynamic_tools: Vec::new(),
            toolset: ToolSet::default(),
        }
    }

    pub(crate) fn static_tool_names(mut self, names: Vec<String>) -> Self {
        self.static_tool_names = names;
        self
    }

    pub(crate) fn add_tools(mut self, tools: ToolSet) -> Self {
        self.toolset = tools;
        self
    }

    pub(crate) fn add_dynamic_tools(
        mut self,
        dyn_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    ) -> Self {
        self.dynamic_tools = dyn_tools;
        self
    }

    /// Add a static tool to the agent
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let toolname = tool.name();
        self.toolset.add_tool(tool);
        self.static_tool_names.push(toolname);
        self
    }

    // Add an MCP tool (from `rmcp`) to the agent
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    #[cfg(feature = "rmcp")]
    pub fn rmcp_tool(mut self, tool: rmcp::model::Tool, client: rmcp::service::ServerSink) -> Self {
        use crate::tool::rmcp::McpTool;
        let toolname = tool.name.clone();
        self.toolset
            .add_tool(McpTool::from_mcp_server(tool, client));
        self.static_tool_names.push(toolname.to_string());
        self
    }

    /// Add some dynamic tools to the agent. On each prompt, `sample` tools from the
    /// dynamic toolset will be inserted in the request.
    pub fn dynamic_tools(
        mut self,
        sample: usize,
        dynamic_tools: impl VectorStoreIndexDyn + 'static,
        toolset: ToolSet,
    ) -> Self {
        self.dynamic_tools.push((sample, Box::new(dynamic_tools)));
        self.toolset.add_tools(toolset);
        self
    }

    pub fn run(mut self) -> ToolServerHandle {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1000);

        #[cfg(not(target_family = "wasm"))]
        tokio::spawn(async move {
            while let Some(message) = rx.recv().await {
                self.handle_message(message).await;
            }
        });

        #[cfg(target_family = "wasm")]
        tokio::task::spawn_local(async move {
            while let Some(message) = rx.recv().await {
                self.handle_message(message).await;
            }
        });

        ToolServerHandle(tx)
    }

    pub async fn handle_message(&mut self, message: ToolServerRequest) {
        let ToolServerRequest {
            callback_channel,
            data,
        } = message;

        match data {
            ToolServerRequestMessageKind::AddTool(tool) => {
                self.static_tool_names.push(tool.name());
                self.toolset.add_tool_boxed(tool);
                callback_channel
                    .send(ToolServerResponse::ToolAdded)
                    .unwrap();
            }
            ToolServerRequestMessageKind::AppendToolset(tools) => {
                self.toolset.add_tools(tools);
                callback_channel
                    .send(ToolServerResponse::ToolAdded)
                    .unwrap();
            }
            ToolServerRequestMessageKind::RemoveTool { tool_name } => {
                self.static_tool_names.retain(|x| *x != tool_name);
                self.toolset.delete_tool(&tool_name);
                callback_channel
                    .send(ToolServerResponse::ToolDeleted)
                    .unwrap();
            }
            ToolServerRequestMessageKind::CallTool { name, args } => {
                match self.toolset.call(&name, args.clone()).await {
                    Ok(result) => {
                        let _ = callback_channel.send(ToolServerResponse::ToolExecuted { result });
                    }
                    Err(err) => {
                        let _ = callback_channel.send(ToolServerResponse::ToolError {
                            error: err.to_string(),
                        });
                    }
                }
            }
            ToolServerRequestMessageKind::GetToolDefs { prompt } => {
                let res = self.get_tool_definitions(prompt).await.unwrap();
                callback_channel
                    .send(ToolServerResponse::ToolDefinitions(res))
                    .unwrap();
            }
        }
    }

    pub async fn get_tool_definitions(
        &mut self,
        text: Option<String>,
    ) -> Result<Vec<ToolDefinition>, CompletionError> {
        let static_tool_names = self.static_tool_names.clone();
        let mut tools = if let Some(text) = text {
            stream::iter(self.dynamic_tools.iter())
                        .then(|(num_sample, index)| async {
                            let req = VectorSearchRequest::builder().query(text.clone()).samples(*num_sample as u64).build().expect("Creating VectorSearchRequest here shouldn't fail since the query and samples to return are always present");
                            Ok::<_, VectorStoreError>(
                                index
                                    .top_n_ids(req)
                                    .await?
                                    .into_iter()
                                    .map(|(_, id)| id)
                                    .collect::<Vec<String>>(),
                            )
                        })
                        .try_fold(vec![], |mut acc, docs| async {
                            for doc in docs {
                                if let Some(tool) = self.toolset.get(&doc) {
                                    acc.push(tool.definition(text.clone()).await)
                                } else {
                                    tracing::warn!("Tool implementation not found in toolset: {}", doc);
                                }
                            }
                            Ok(acc)
                        })
                        .await
                        .map_err(|e| CompletionError::RequestError(Box::new(e)))?
        } else {
            Vec::new()
        };

        for toolname in static_tool_names {
            if let Some(tool) = self.toolset.get(&toolname) {
                tools.push(tool.definition(String::new()).await)
            } else {
                tracing::warn!("Tool implementation not found in toolset: {}", toolname);
            }
        }

        Ok(tools)
    }
}

#[derive(Clone)]
pub struct ToolServerHandle(Sender<ToolServerRequest>);

impl ToolServerHandle {
    pub async fn add_tool(&self, tool: impl ToolDyn + 'static) -> Result<(), ToolServerError> {
        let tool = Box::new(tool);

        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::AddTool(tool),
            })
            .await?;

        let res = rx.await?;

        let ToolServerResponse::ToolAdded = res else {
            return Err(ToolServerError::InvalidMessage(res));
        };

        Ok(())
    }

    pub async fn append_toolset(&self, toolset: ToolSet) -> Result<(), ToolServerError> {
        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::AppendToolset(toolset),
            })
            .await?;

        let res = rx.await?;

        let ToolServerResponse::ToolAdded = res else {
            return Err(ToolServerError::InvalidMessage(res));
        };

        Ok(())
    }

    pub async fn remove_tool(&self, tool_name: &str) -> Result<(), ToolServerError> {
        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::RemoveTool {
                    tool_name: tool_name.to_string(),
                },
            })
            .await?;

        let res = rx.await?;

        let ToolServerResponse::ToolDeleted = res else {
            return Err(ToolServerError::InvalidMessage(res));
        };

        Ok(())
    }

    pub async fn call_tool(&self, tool_name: &str, args: &str) -> Result<String, ToolServerError> {
        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::CallTool {
                    name: tool_name.to_string(),
                    args: args.to_string(),
                },
            })
            .await?;

        let res = rx.await?;

        match res {
            ToolServerResponse::ToolExecuted { result, .. } => Ok(result),
            ToolServerResponse::ToolError { error } => Err(ToolServerError::ToolsetError(
                ToolSetError::ToolCallError(ToolError::ToolCallError(error.into())),
            )),
            invalid => Err(ToolServerError::InvalidMessage(invalid)),
        }
    }

    pub async fn get_tool_defs(
        &self,
        prompt: Option<String>,
    ) -> Result<Vec<ToolDefinition>, ToolServerError> {
        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::GetToolDefs { prompt },
            })
            .await?;

        let res = rx.await?;

        let ToolServerResponse::ToolDefinitions(tooldefs) = res else {
            return Err(ToolServerError::InvalidMessage(res));
        };

        Ok(tooldefs)
    }
}

pub struct ToolServerRequest {
    callback_channel: futures::channel::oneshot::Sender<ToolServerResponse>,
    data: ToolServerRequestMessageKind,
}

pub enum ToolServerRequestMessageKind {
    AddTool(Box<dyn ToolDyn>),
    AppendToolset(ToolSet),
    RemoveTool { tool_name: String },
    CallTool { name: String, args: String },
    GetToolDefs { prompt: Option<String> },
}

#[derive(PartialEq, Debug)]
pub enum ToolServerResponse {
    ToolAdded,
    ToolDeleted,
    ToolExecuted { result: String },
    ToolError { error: String },
    ToolDefinitions(Vec<ToolDefinition>),
}

#[derive(Debug, thiserror::Error)]
pub enum ToolServerError {
    #[error("Sending message was cancelled")]
    Canceled(#[from] Canceled),
    #[error("Toolset error: {0}")]
    ToolsetError(#[from] ToolSetError),
    #[error("Error while sending message: {0}")]
    SendError(#[from] SendError<ToolServerRequest>),
    #[error("An invalid message type was returned")]
    InvalidMessage(ToolServerResponse),
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    use crate::{
        completion::ToolDefinition,
        tool::{Tool, server::ToolServer},
    };

    #[derive(Deserialize)]
    struct OperationArgs {
        x: i32,
        y: i32,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("Math error")]
    struct MathError;

    #[derive(Deserialize, Serialize)]
    struct Adder;
    impl Tool for Adder {
        const NAME: &'static str = "add";
        type Error = MathError;
        type Args = OperationArgs;
        type Output = i32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: "add".to_string(),
                description: "Add x and y together".to_string(),
                parameters: json!({
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
                    },
                    "required": ["x", "y"],
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            println!("[tool-call] Adding {} and {}", args.x, args.y);
            let result = args.x + args.y;
            Ok(result)
        }
    }

    #[tokio::test]
    pub async fn test_toolserver() {
        let server = ToolServer::new();

        let handle = server.run();

        handle.add_tool(Adder).await.unwrap();
        let res = handle.get_tool_defs(None).await.unwrap();

        assert_eq!(res.len(), 1);

        let json_args_as_string =
            serde_json::to_string(&serde_json::json!({"x": 2, "y": 5})).unwrap();
        let res = handle.call_tool("add", &json_args_as_string).await.unwrap();
        assert_eq!(res, "7");

        handle.remove_tool("add").await.unwrap();
        let res = handle.get_tool_defs(None).await.unwrap();

        assert_eq!(res.len(), 0);
    }
}
