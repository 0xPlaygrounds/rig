//! Tool authoring, registration, and canonical structured execution.
//!
//! A typed [`Tool`] implements one [`Tool::call`] method. Rig erases it
//! internally, executes it through one structured path, and exposes a single
//! [`ToolResult`] view to hooks and runtime callers. [`ToolContext`] is the sole
//! path for typed inbound context and host-only result metadata.
//!
//! # Implementing a typed tool
//!
//! Ordinary serializable return values are converted to canonical model output
//! without first passing through a string.
//!
//! ```
//! use rig_agent::tool::{Tool, ToolContext};
//! use serde::{Deserialize, Serialize};
//! use std::convert::Infallible;
//!
//! #[derive(Deserialize)]
//! struct AddArgs {
//!     left: i64,
//!     right: i64,
//! }
//!
//! #[derive(Serialize)]
//! struct Sum {
//!     value: i64,
//! }
//!
//! #[derive(Clone, Debug, PartialEq)]
//! struct AuditRecord(i64);
//!
//! struct Add;
//!
//! impl Tool for Add {
//!     const NAME: &'static str = "add";
//!     type Args = AddArgs;
//!     type Output = Sum;
//!     type Error = Infallible;
//!
//!     fn description(&self) -> String {
//!         "Add two integers".into()
//!     }
//!
//!     fn parameters(&self) -> serde_json::Value {
//!         serde_json::json!({
//!             "type": "object",
//!             "properties": {
//!                 "left": { "type": "integer" },
//!                 "right": { "type": "integer" }
//!             },
//!             "required": ["left", "right"]
//!         })
//!     }
//!
//!     async fn call(
//!         &self,
//!         context: &mut ToolContext,
//!         args: Self::Args,
//!     ) -> Result<Self::Output, Self::Error> {
//!         let value = args.left + args.right;
//!         context.insert_result(AuditRecord(value));
//!         Ok(Sum { value })
//!     }
//! }
//! ```
//!
//! Return [`ToolOutput`] for explicit JSON or multimodal presentation. A
//! [`ToolResultContent`](rig_core::message::ToolResultContent) or a `Vec` of
//! content blocks can also be used directly as a typed tool output without
//! being mistaken for ordinary JSON.
//!
//! ```
//! use rig_core::{
//!     message::{ImageMediaType, ToolResultContent},
//!     tool::ToolOutput,
//! };
//!
//! let output = ToolOutput::one(ToolResultContent::image_base64(
//!     "iVBORw0KGgo=",
//!     Some(ImageMediaType::PNG),
//!     None,
//! ));
//! assert!(matches!(
//!     output.as_content().first(),
//!     Some(ToolResultContent::Image(_))
//! ));
//! ```
//!
//! Explicit [`ToolExecutionError`] constructors keep their detailed message
//! model-visible so validation failures can tell the model how to recover. The
//! default [`Tool::map_error`] conversion preserves an arbitrary source error
//! for operators but exposes only safe kind-level feedback. Override
//! [`Tool::map_error`] or use [`ToolExecutionError::with_model_output`] when a
//! domain error has deliberate structured or actionable model feedback.
//!
//! # Migration from the parallel tool APIs
//!
//! | Removed concept | Canonical replacement |
//! | --- | --- |
//! | Multiple typed `call*` methods | One [`Tool::call`] method |
//! | Public dynamic dispatch traits | [`DynamicTool`] |
//! | Parallel error and failure types | [`ToolExecutionError`] and [`crate::tool::ToolErrorKind`] |
//! | Author-facing outcome enums | Ordinary `Result<T, Self::Error>` normalized at dispatch |
//! | Separate call/result extension maps | [`ToolContext`] |
//! | Parallel string/structured dispatch | [`ToolSet::execute`] and [`server::ToolServerHandle::execute`] |
//!
//! Model-visible output remains typed throughout dispatch. Rendering to text is
//! a terminal provider or telemetry concern; Rig does not reconstruct rich

pub mod builtin;
pub mod server;

pub use rig_core::tool::{
    DynamicTool, ErasedEmbeddingTool, ErasedTool, IntoToolOutput, PortableDynamicTool,
    RegisteredTool, Tool, ToolCatalog, ToolDispatch, ToolEmbedding, ToolErrorKind,
    ToolExecutionError, ToolOutput, ToolResult, ToolSet, dispatch_tool, tool_definition,
};
pub use rig_core::tool::{MissingToolContext, ToolContext};

#[cfg(test)]
mod toolset_clone_tests {
    use std::sync::Arc;

    use super::{RegisteredTool, ToolSet};
    use crate::test_utils::{MockAddTool, MockSubtractTool};

    fn erased_ptr(set: &ToolSet, name: &str) -> *const () {
        match &set.get(name).expect("registered").clone() {
            RegisteredTool::Static(tool) => Arc::as_ptr(tool).cast(),
            RegisteredTool::Embedding(tool) => Arc::as_ptr(tool).cast(),
        }
    }

    /// A clone shares the tool implementations (pointer-equal `Arc`s) and is
    /// independent for subsequent registration changes.
    #[test]
    fn clone_shares_implementations_and_diverges_on_mutation() {
        let mut original = ToolSet::default();
        original.add_tool(MockAddTool);

        let mut clone = original.clone();
        assert_eq!(erased_ptr(&original, "add"), erased_ptr(&clone, "add"));
        assert_eq!(original.tool_definitions(), clone.tool_definitions());

        clone.add_tool(MockSubtractTool);
        assert!(clone.contains("subtract"));
        assert!(!original.contains("subtract"));

        original.delete_tool("add");
        assert!(clone.contains("add"));
    }
}

#[cfg(test)]
mod migrated_tests {
    use crate::test_utils::{
        MockExampleTool, MockImageOutputTool, MockObjectOutputTool, MockStringOutputTool,
        MockToolError, mock_math_toolset,
    };
    use portable_fixtures::{
        PortableEmbeddingFixture, portable_dynamic_fixture, portable_fixture_output,
    };
    use rig_core::embeddings::tool::ToolSchema;
    use rig_core::message::{DocumentSourceKind, ToolResultContent};
    use serde_json::json;

    use super::*;

    /// Portable-tool fixtures relocated from the removed `rig-runtime-conformance`
    /// crate; used only by these migrated tests.
    mod portable_fixtures {
        use rig_core::{
            message::{ImageMediaType, ToolResultContent},
            tool::{
                PortableDynamicTool, PortableTool, PortableToolEmbedding, ToolExecutionError,
                ToolOutput,
            },
        };
        use serde::{Deserialize, Serialize};

        const PORTABLE_FIXTURE_IMAGE: &str = "cG9ydGFibGUtZml4dHVyZQ==";

        #[derive(Clone, Debug, Deserialize, Serialize)]
        pub struct PortableEmbeddingArgs {
            pub value: String,
            #[serde(default)]
            pub fail: bool,
        }

        #[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
        pub struct PortableEmbeddingContext {
            pub prefix: String,
        }

        #[derive(Debug, thiserror::Error)]
        #[error("portable fixture failure")]
        pub struct PortableFixtureError;

        pub fn portable_fixture_output(label: impl Into<String>) -> ToolOutput {
            let mut content = vec![ToolResultContent::json(
                serde_json::json!({"label": label.into()}),
            )];
            content.push(ToolResultContent::image_base64(
                PORTABLE_FIXTURE_IMAGE,
                Some(ImageMediaType::PNG),
                None,
            ));
            ToolOutput::content(content).expect("fixture content is non-empty")
        }

        pub fn portable_dynamic_fixture() -> PortableDynamicTool {
            PortableDynamicTool::new(
                "portable_runtime_name",
                "portable dynamic definition",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "fail": {"type": "boolean"}
                    },
                    "required": ["value"]
                }),
                |arguments| {
                    Box::pin(async move {
                        if arguments
                            .get("fail")
                            .and_then(serde_json::Value::as_bool)
                            .unwrap_or_default()
                        {
                            Err(ToolExecutionError::provider("portable dynamic failure")
                                .with_code("portable_dynamic_fixture")
                                .with_model_output(portable_fixture_output(
                                    "portable dynamic failure",
                                )))
                        } else {
                            Ok(portable_fixture_output(format!(
                                "dynamic:{}",
                                arguments
                                    .get("value")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or_default()
                            )))
                        }
                    })
                },
            )
        }

        #[derive(Clone)]
        pub struct PortableEmbeddingFixture {
            context: PortableEmbeddingContext,
        }

        impl PortableEmbeddingFixture {
            pub fn new(prefix: impl Into<String>) -> Self {
                Self {
                    context: PortableEmbeddingContext {
                        prefix: prefix.into(),
                    },
                }
            }
        }

        impl PortableTool for PortableEmbeddingFixture {
            const NAME: &'static str = "portable_embedding_fixture";
            type Args = PortableEmbeddingArgs;
            type Output = ToolOutput;
            type Error = PortableFixtureError;

            fn description(&self) -> String {
                format!("{} portable embedding fixture", self.context.prefix)
            }

            fn parameters(&self) -> serde_json::Value {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "fail": {"type": "boolean"}
                    },
                    "required": ["value"]
                })
            }

            fn map_error(&self, error: Self::Error) -> ToolExecutionError {
                ToolExecutionError::provider(error.to_string())
                    .with_code("portable_fixture")
                    .with_model_output(portable_fixture_output("portable failure"))
                    .with_source(error)
            }

            async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
                if arguments.fail {
                    Err(PortableFixtureError)
                } else {
                    Ok(portable_fixture_output(format!(
                        "{}:{}",
                        self.context.prefix, arguments.value
                    )))
                }
            }
        }

        impl PortableToolEmbedding for PortableEmbeddingFixture {
            type InitError = std::convert::Infallible;
            type Context = PortableEmbeddingContext;
            type State = ();

            fn embedding_docs(&self) -> Vec<String> {
                vec![format!(
                    "{} portable embedding document",
                    self.context.prefix
                )]
            }

            fn context(&self) -> Self::Context {
                self.context.clone()
            }

            fn init(_state: Self::State, context: Self::Context) -> Result<Self, Self::InitError> {
                Ok(Self { context })
            }
        }
    }

    fn get_test_toolset() -> ToolSet {
        mock_math_toolset()
    }

    #[test]
    fn test_get_tool_definitions() {
        let toolset = get_test_toolset();
        let tools = toolset.tool_definitions();
        assert_eq!(tools.len(), 2);
        assert_eq!(
            tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            vec!["add", "subtract"],
            "provider definitions must use registered tool names in order"
        );
        assert!(tools.iter().all(|tool| !tool.description.is_empty()));
        assert!(tools.iter().all(|tool| tool.parameters.is_object()));
    }

    #[test]
    fn test_tool_deletion() {
        let mut toolset = get_test_toolset();
        assert_eq!(toolset.len(), 2);
        toolset.delete_tool("add");
        assert!(!toolset.contains("add"));
        assert_eq!(toolset.len(), 1);
        assert_eq!(
            toolset.names().map(str::to_owned).collect::<Vec<_>>(),
            vec!["subtract".to_string()]
        );
    }

    #[test]
    fn deleting_a_middle_tool_preserves_order_of_survivors() {
        // Guards the `shift_remove` (not `swap_remove`) choice in `delete_tool`.
        // `swap_remove` would move the last tool into the deleted slot, so this
        // only catches a regression with 3+ tools and a non-last deletion: here
        // a `swap_remove("beta")` would yield [alpha, delta, gamma].
        let mut toolset = ToolSet::default();
        for name in ["alpha", "beta", "gamma", "delta"] {
            toolset.add_dynamic_tool(named_tool(name, "test tool"));
        }

        toolset.delete_tool("beta");

        assert_eq!(
            toolset.names().map(str::to_owned).collect::<Vec<_>>(),
            vec![
                "alpha".to_string(),
                "gamma".to_string(),
                "delta".to_string()
            ],
            "survivors must keep their registration order after a middle deletion"
        );
    }

    /// A runtime-defined tool used by ordering and duplicate-registration tests.
    fn named_tool(name: &str, description: &str) -> DynamicTool {
        let output = format!("called {description}");
        DynamicTool::new(
            name,
            description,
            json!({ "type": "object", "properties": {} }),
            move |_context, _args| {
                let output = output.clone();
                Box::pin(async move { Ok(ToolOutput::text(output)) })
            },
        )
    }

    #[test]
    fn tool_definition_uses_flattened_dyn_metadata() {
        let tool = named_tool("alpha", "runtime description");
        let definition = tool.definition();

        assert_eq!(definition.name, "alpha");
        assert_eq!(definition.description, "runtime description");
        assert_eq!(definition.parameters["type"], "object");
    }

    #[tokio::test]
    async fn tool_definitions_follow_registration_order() {
        // Enough names that any non-order-preserving storage would almost
        // surely surface a regression: its iteration order would differ from
        // insertion order.
        let names: Vec<String> = (0..32).map(|i| format!("tool_{i:02}")).collect();
        let mut toolset = ToolSet::default();
        for name in &names {
            toolset.add_dynamic_tool(named_tool(name, "test tool"));
        }

        let defs = toolset.tool_definitions();
        let def_names: Vec<String> = defs.into_iter().map(|def| def.name).collect();
        assert_eq!(def_names, names);

        let docs = toolset.documents();
        let doc_ids: Vec<String> = docs.into_iter().map(|doc| doc.id).collect();
        assert_eq!(doc_ids, names);
    }

    #[tokio::test]
    async fn typed_tool_name_is_definition_source_of_truth() {
        struct NamedTool;

        impl Tool for NamedTool {
            const NAME: &'static str = "canonical";
            type Error = rig::tool::ToolExecutionError;
            type Args = serde_json::Value;
            type Output = String;

            fn description(&self) -> String {
                "uses the canonical typed name".to_string()
            }
            fn parameters(&self) -> serde_json::Value {
                json!({ "type": "object", "properties": {} })
            }
            async fn call(
                &self,
                _context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                Ok("ok".to_string())
            }
        }

        let mut toolset = ToolSet::default();
        toolset.add_tool(NamedTool);

        let defs = toolset.tool_definitions();
        assert_eq!(defs[0].name, NamedTool::NAME);

        let docs = toolset.documents();
        assert_eq!(docs[0].id, NamedTool::NAME);
        assert!(docs[0].text.contains(NamedTool::NAME));
    }

    #[test]
    fn retrieved_tool_schemas_use_canonical_name() {
        #[derive(Debug, thiserror::Error)]
        #[error("init error")]
        struct InitError;

        struct RetrievedTool;

        impl Tool for RetrievedTool {
            const NAME: &'static str = "retrieved";
            type Error = rig::tool::ToolExecutionError;
            type Args = serde_json::Value;
            type Output = String;

            fn description(&self) -> String {
                "dynamic tool".to_string()
            }

            fn parameters(&self) -> serde_json::Value {
                json!({ "type": "object", "properties": {} })
            }

            async fn call(
                &self,
                _context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                Ok("ok".to_string())
            }
        }

        impl ToolEmbedding for RetrievedTool {
            type InitError = InitError;
            type Context = ();
            type State = ();

            fn embedding_docs(&self) -> Vec<String> {
                vec!["dynamic tool docs".to_string()]
            }

            fn context(&self) -> Self::Context {}

            fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
                Ok(Self)
            }
        }

        let mut toolset = ToolSet::default();
        toolset.add_retrieved_tool(RetrievedTool);

        let schemas = toolset.schemas().unwrap();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name, RetrievedTool::NAME);
        assert_eq!(schemas[0].embedding_docs, vec!["dynamic tool docs"]);
    }

    #[tokio::test]
    async fn portable_embedding_tool_uses_classic_retrieval_without_schema_drift() {
        let tool = PortableEmbeddingFixture::new("shared");
        let portable_schema = ToolSchema::try_from(&tool).unwrap();
        let mut toolset = ToolSet::default();
        toolset.add_retrieved_tool(tool);

        let schemas = toolset.schemas().unwrap();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name, portable_schema.name);
        assert_eq!(schemas[0].context, portable_schema.context);
        assert_eq!(schemas[0].embedding_docs, portable_schema.embedding_docs);

        let handle = server::ToolServer::new()
            .retrieved_tools(
                1,
                crate::test_utils::MockToolIndex::new([portable_schema.name.as_str()]),
                toolset,
            )
            .run();
        let definitions = handle
            .tool_defs(Some("find the shared portable tool".to_string()))
            .await
            .unwrap();

        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, portable_schema.name);
        assert_eq!(
            definitions[0].description,
            "shared portable embedding fixture"
        );
        assert_eq!(
            definitions[0].parameters,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "fail": {"type": "boolean"}
                },
                "required": ["value"]
            })
        );

        let success = handle
            .execute(
                &definitions[0].name,
                r#"{"value":"ok"}"#,
                &mut ToolContext::new(),
            )
            .await;
        assert!(success.is_success());
        assert_eq!(success.output(), &portable_fixture_output("shared:ok"));

        let failure = handle
            .execute(
                &definitions[0].name,
                r#"{"value":"ignored","fail":true}"#,
                &mut ToolContext::new(),
            )
            .await;
        let error = failure
            .error()
            .expect("portable failure should be retained");
        assert_eq!(error.kind(), ToolErrorKind::Provider);
        assert_eq!(error.code(), Some("portable_fixture"));
        assert_eq!(
            error.model_output(),
            &portable_fixture_output("portable failure")
        );
        assert_eq!(failure.output(), error.model_output());
    }

    #[tokio::test]
    async fn portable_dynamic_tool_executes_in_classic_registry_without_callback_rewrite() {
        let portable = portable_dynamic_fixture();
        let mut toolset = ToolSet::default();
        toolset.add_dynamic_tool(named_tool("before", "before"));
        let registered_name = toolset.add_portable_dynamic_tool(portable);
        toolset.add_dynamic_tool(named_tool("after", "after"));

        assert_eq!(registered_name, "portable_runtime_name");
        assert_eq!(
            toolset
                .tool_definitions()
                .iter()
                .map(|definition| definition.name.as_str())
                .collect::<Vec<_>>(),
            ["before", "portable_runtime_name", "after"]
        );

        let result = toolset
            .execute(
                "portable_runtime_name",
                r#"{"value":"ok"}"#,
                &mut ToolContext::new(),
            )
            .await;
        assert!(result.is_success());
        assert_eq!(result.output(), &portable_fixture_output("dynamic:ok"));

        let failure = toolset
            .execute(
                "portable_runtime_name",
                r#"{"value":"ignored","fail":true}"#,
                &mut ToolContext::new(),
            )
            .await;
        assert!(failure.is_error());
        let error = failure
            .error()
            .expect("portable failure should be retained");
        assert_eq!(error.kind(), ToolErrorKind::Provider);
        assert_eq!(error.code(), Some("portable_dynamic_fixture"));
        assert_eq!(
            error.model_output(),
            &portable_fixture_output("portable dynamic failure")
        );
        assert_eq!(failure.output(), error.model_output());
    }

    #[tokio::test]
    async fn duplicate_registration_replaces_in_place() {
        let mut toolset = ToolSet::default();
        toolset.add_dynamic_tool(named_tool("alpha", "first alpha"));
        toolset.add_dynamic_tool(named_tool("beta", "beta"));
        toolset.add_dynamic_tool(named_tool("alpha", "second alpha"));

        let defs = toolset.tool_definitions();
        assert_eq!(
            defs.iter().map(|def| def.name.as_str()).collect::<Vec<_>>(),
            vec!["alpha", "beta"],
            "the duplicate should be deduped and keep its original position"
        );
        assert_eq!(
            defs[0].description, "second alpha",
            "the last registration should win"
        );

        let output = toolset
            .execute("alpha", "{}", &mut ToolContext::new())
            .await
            .output()
            .render();
        assert_eq!(output, "called second alpha");
    }

    #[tokio::test]
    async fn add_tools_merges_in_order_and_replaces_existing() {
        let mut base = ToolSet::default();
        base.add_dynamic_tool(named_tool("alpha", "base alpha"));
        base.add_dynamic_tool(named_tool("beta", "base beta"));

        let mut incoming = ToolSet::default();
        incoming.add_dynamic_tool(named_tool("gamma", "incoming gamma"));
        incoming.add_dynamic_tool(named_tool("alpha", "incoming alpha"));

        base.add_tools(incoming);

        let defs = base.tool_definitions();
        assert_eq!(
            defs.iter().map(|def| def.name.as_str()).collect::<Vec<_>>(),
            vec!["alpha", "beta", "gamma"],
            "merged tools should follow registration order with replaced names keeping position"
        );
        assert_eq!(defs[0].description, "incoming alpha");
    }

    #[tokio::test]
    async fn string_tool_outputs_are_preserved_verbatim() {
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockStringOutputTool);

        let output = toolset
            .execute("string_output", "{}", &mut ToolContext::new())
            .await;

        assert_eq!(output.output(), &ToolOutput::text("Hello\nWorld"));
    }

    #[tokio::test]
    async fn json_shaped_string_output_stays_literal_text_through_dispatch() {
        struct JsonShapedStringTool;

        impl Tool for JsonShapedStringTool {
            const NAME: &'static str = "json_shaped_string";
            type Error = rig::tool::ToolExecutionError;
            type Args = serde_json::Value;
            type Output = String;

            fn description(&self) -> String {
                "Returns text that happens to look like a rich-content envelope".into()
            }

            fn parameters(&self) -> serde_json::Value {
                json!({"type": "object"})
            }

            async fn call(
                &self,
                _context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                Ok(r#"{"type":"image","data":"literal"}"#.to_string())
            }
        }

        let mut toolset = ToolSet::default();
        toolset.add_tool(JsonShapedStringTool);

        let result = toolset
            .execute(JsonShapedStringTool::NAME, "{}", &mut ToolContext::new())
            .await;

        assert_eq!(
            result.output(),
            &ToolOutput::text(r#"{"type":"image","data":"literal"}"#)
        );
    }

    #[tokio::test]
    async fn explicit_image_tool_outputs_remain_structured() {
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockImageOutputTool);

        let result = toolset
            .execute("image_output", "{}", &mut ToolContext::new())
            .await;
        let content = result.output().clone().into_content();

        assert_eq!(content.len(), 1);
        match content.first() {
            Some(ToolResultContent::Image(image)) => {
                assert!(matches!(image.data, DocumentSourceKind::Base64(_)));
                assert_eq!(
                    image.media_type,
                    Some(rig_core::message::ImageMediaType::PNG)
                );
            }
            other => panic!("expected image tool result content, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn object_tool_outputs_still_serialize_as_json() {
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockObjectOutputTool);

        let result = toolset
            .execute("object_output", "{}", &mut ToolContext::new())
            .await;

        assert_eq!(
            result.output(),
            &ToolOutput::json(json!({
                "status": "ok",
                "count": 42
            }))
        );
    }

    #[tokio::test]
    async fn null_args_are_preserved_for_unit_args() {
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockExampleTool);

        let output = toolset
            .execute("example_tool", "null", &mut ToolContext::new())
            .await;

        assert_eq!(output.output(), &ToolOutput::text("Example answer"));
    }

    // Struct-typed args with all-optional fields — serde rejects `null` for these
    // even though the fields are optional. The normalization in crate-private erased dispatch
    // falls back from `null` to `{}` so callers can omit the
    // wrapping `Option<Args>` workaround.
    #[tokio::test]
    async fn null_args_are_normalized_to_empty_object() {
        #[derive(serde::Deserialize, serde::Serialize)]
        struct NoRequiredArgs {
            label: Option<String>,
        }

        struct NoArgTool;

        impl Tool for NoArgTool {
            const NAME: &'static str = "no_arg_tool";
            type Error = MockToolError;
            type Args = NoRequiredArgs;
            type Output = String;

            fn description(&self) -> String {
                "Tool with no required arguments".to_string()
            }

            fn parameters(&self) -> serde_json::Value {
                json!({"type": "object", "properties": {}})
            }

            async fn call(
                &self,
                _context: &mut ToolContext,
                args: Self::Args,
            ) -> Result<Self::Output, Self::Error> {
                Ok(args.label.unwrap_or_else(|| "default".to_string()))
            }
        }

        let mut toolset = ToolSet::default();
        toolset.add_tool(NoArgTool);

        // `null` is what LLMs send when no arguments are provided; without the
        // normalization this would return an `InvalidArgs` execution error.
        let output = toolset
            .execute("no_arg_tool", "null", &mut ToolContext::new())
            .await;

        assert_eq!(output.output(), &ToolOutput::text("default"));
    }
}
