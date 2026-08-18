//! Uses a local llama.cpp server (OpenAI-compatible chat completions API) with
//! a vision-capable model. The agent is given a `view_file` tool that returns
//! the image itself (not a description), so the model can actually "see" it.
//!
//! Requires a llama.cpp server running on `localhost:8080` with a vision model
//! loaded (for example `unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0`).

use std::fs;
use std::path::Path;

use anyhow::Result;
use base64::Engine;
use rig::agent::tool::{Tool, ToolContext, ToolOutput};
use rig::completion::message::{ImageMediaType, ToolResultContent};
use rig::prelude::*;
use rig::providers::openai;
use serde::Deserialize;

const MODEL: &str = "unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0";
// The image lives at the workspace root, two levels above this example's
// manifest directory.
const IMAGE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../img/ryzome-bg.png");

/// Arguments for the `view_file` tool.
#[derive(Debug, Deserialize)]
struct ViewFileArgs {
    file_name: String,
}

/// Errors returned by the `view_file` tool.
#[derive(Debug, thiserror::Error)]
enum ViewFileError {
    #[error("failed to read file: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported image format: {0}")]
    UnsupportedFormat(String),
}

/// A tool that returns the image at the given path so the model can see it.
///
/// The output is the image content itself — no textual description — which is
/// what lets a vision-capable model actually look at the file.
#[derive(Clone)]
struct ViewFile;

impl Tool for ViewFile {
    const NAME: &'static str = "view_file";

    type Args = ViewFileArgs;
    type Output = ToolOutput;
    type Error = ViewFileError;

    fn description(&self) -> String {
        "View an image file so the model can see its contents. Returns the image itself, not a description."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Path to the image file to view."
                }
            },
            "required": ["file_name"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let bytes = fs::read(&args.file_name)?;
        let media_type = media_type_for(&args.file_name)
            .ok_or_else(|| ViewFileError::UnsupportedFormat(args.file_name.clone()))?;
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);

        Ok(ToolOutput::one(ToolResultContent::image_base64(
            encoded,
            Some(media_type),
            None,
        )))
    }
}

/// Infer the image media type from a file's extension.
fn media_type_for(file_name: &str) -> Option<ImageMediaType> {
    let ext = Path::new(file_name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("png") => Some(ImageMediaType::PNG),
        Some("jpg") | Some("jpeg") => Some(ImageMediaType::JPEG),
        Some("gif") => Some(ImageMediaType::GIF),
        Some("webp") => Some(ImageMediaType::WEBP),
        Some("heic") => Some(ImageMediaType::HEIC),
        Some("heif") => Some(ImageMediaType::HEIF),
        Some("svg") => Some(ImageMediaType::SVG),
        _ => None,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // llama.cpp exposes the OpenAI-compatible chat completions API under /v1.
    let client = openai::CompletionsClient::builder()
        .api_key("not-needed")
        .base_url("http://localhost:8080/v1")
        .build()?;

    let agent = client
        .agent(MODEL)
        .preamble("You are a helpful vision assistant.")
        .tool(ViewFile)
        // One turn for the model to call `view_file`, one more to see the
        // returned image and answer.
        .default_max_turns(3)
        .build();

    let response = agent
        .prompt(format!(
            "Use the view_file tool to view the file '{IMAGE_PATH}', then describe in one word what the image contains."
        ))
        .await?;
    println!("{response}");

    Ok(())
}
