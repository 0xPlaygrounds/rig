//! The two tools the example offers: `write_file`, whose long `content`
//! argument makes streaming visible, and `add`, a small call the model can
//! make alongside it. Both record what ran, so the example can show that a
//! stopped run executed nothing.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

use rig::tool::{Tool, ToolContext};
use serde::Deserialize;
use serde_json::json;

/// What the tools executed, shared with the example's checks.
#[derive(Clone, Default)]
pub struct Executed(Arc<Mutex<Vec<String>>>);

impl Executed {
    fn record(&self, entry: String) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(entry);
    }

    /// Every executed call, in order.
    pub fn entries(&self) -> Vec<String> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("the path is empty")]
    EmptyPath,
}

#[derive(Deserialize)]
pub struct WriteFileArgs {
    path: String,
    content: String,
}

/// Writes a file into an in-memory workspace; nothing touches the disk.
#[derive(Clone, Default)]
pub struct WriteFile {
    pub files: Arc<Mutex<BTreeMap<String, String>>>,
    pub executed: Executed,
}

impl Tool for WriteFile {
    const NAME: &'static str = "write_file";
    type Error = ToolError;
    type Args = WriteFileArgs;
    type Output = String;

    fn description(&self) -> String {
        "Write text content to a file at a relative path, replacing it.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Relative file path" },
                "content": { "type": "string", "description": "The whole file content" }
            },
            "required": ["path", "content"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        if args.path.trim().is_empty() {
            return Err(ToolError::EmptyPath);
        }
        let bytes = args.content.len();
        self.executed
            .record(format!("write_file({}, {bytes} bytes)", args.path));
        self.files
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(args.path.clone(), args.content);
        Ok(format!("wrote {bytes} bytes to {}", args.path))
    }
}

#[derive(Deserialize)]
pub struct AddArgs {
    x: i64,
    y: i64,
}

/// Adds two integers.
#[derive(Clone, Default)]
pub struct Add {
    pub executed: Executed,
}

impl Tool for Add {
    const NAME: &'static str = "add";
    type Error = ToolError;
    type Args = AddArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Add the integers x and y.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": { "type": "integer" },
                "y": { "type": "integer" }
            },
            "required": ["x", "y"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.executed.record(format!("add({}, {})", args.x, args.y));
        Ok(args.x + args.y)
    }
}
