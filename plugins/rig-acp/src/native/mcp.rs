use std::collections::HashMap;
use std::process::Stdio;

use rig_harness::rig_core::{message::ToolName, tool::DynamicTool};
use rig_rmcp::McpConnection;
use rmcp::ServiceExt;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    #[error("invalid MCP declaration: {0}")]
    Declaration(&'static str),
    #[error("MCP setup failed: {0}")]
    Setup(#[from] Box<dyn std::error::Error + Send + Sync>),
}

pub(super) async fn connect(servers: &[Value]) -> Result<Vec<DynamicTool>, Error> {
    let mut tools = Vec::new();
    let mut names = std::collections::HashSet::new();
    for server in servers {
        let name = server
            .get("name")
            .and_then(Value::as_str)
            .ok_or(Error::Declaration("missing name"))?;
        let client = rmcp::model::ClientConfig::default();
        let connection = match server.get("type").and_then(Value::as_str) {
            Some("http") => {
                let url = server
                    .get("url")
                    .and_then(Value::as_str)
                    .ok_or(Error::Declaration("missing URL"))?;
                let mut headers = HashMap::new();
                for header in server
                    .get("headers")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let name = header
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or(Error::Declaration("missing header name"))?;
                    let value = header
                        .get("value")
                        .and_then(Value::as_str)
                        .ok_or(Error::Declaration("missing header value"))?;
                    headers.insert(
                        http::HeaderName::try_from(name).map_err(boxed)?,
                        http::HeaderValue::try_from(value).map_err(boxed)?,
                    );
                }
                let transport = StreamableHttpClientTransport::from_config(
                    StreamableHttpClientTransportConfig::with_uri(url).custom_headers(headers),
                );
                McpConnection::new(client.serve(transport).await.map_err(boxed)?)
            }
            None | Some("stdio") => {
                let executable = server
                    .get("command")
                    .and_then(Value::as_str)
                    .ok_or(Error::Declaration("missing command"))?;
                let mut command = tokio::process::Command::new(executable);
                command.stderr(Stdio::inherit()).kill_on_drop(true);
                for arg in server
                    .get("args")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    command.arg(arg.as_str().ok_or(Error::Declaration("invalid argument"))?);
                }
                for env in server
                    .get("env")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    command.env(
                        env.get("name")
                            .and_then(Value::as_str)
                            .ok_or(Error::Declaration("missing environment name"))?,
                        env.get("value")
                            .and_then(Value::as_str)
                            .ok_or(Error::Declaration("missing environment value"))?,
                    );
                }
                let transport = TokioChildProcess::new(command).map_err(boxed)?;
                McpConnection::new(client.serve(transport).await.map_err(boxed)?)
            }
            _ => return Err(Error::Declaration("unsupported transport")),
        };
        for tool in connection.tools().await.map_err(boxed)? {
            let tool_name = format!("{name}__{}", tool.name());
            if !names.insert(tool_name.clone()) {
                return Err(Error::Declaration("duplicate tool name"));
            }
            let name = ToolName::new(tool_name).map_err(boxed)?;
            tools.push(DynamicTool::try_from(tool.with_name(name)).map_err(boxed)?);
        }
    }
    Ok(tools)
}

fn boxed(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Setup(Box::new(error))
}
