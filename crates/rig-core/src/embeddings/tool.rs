//! The module defines the [ToolSchema] struct: a tool's discovery record.

use crate::Embed;
use serde::Serialize;

use super::embed::EmbedError;

/// A tool's discovery record — the owned data used to find a tool by
/// similarity search before advertising it for a turn.
///
/// This is plain data, not a projection of a trait. Build one per tool you want
/// discoverable, embed the batch, and store the results keyed by [`Self::name`];
/// a retrieval hook then searches the store and narrows the advertised tool set
/// with `RequestPatch::active_tools`. Tools themselves remain registered as
/// executable records — discovery selects among them, it does not construct
/// them.
#[derive(Clone, Serialize, Default, Eq, PartialEq, Debug)]
pub struct ToolSchema {
    /// The tool's name, matching the executable record it selects.
    pub name: String,
    /// Free-form data carried alongside the tool, untouched by Rig.
    ///
    /// Nothing in the runtime reads this. It is a slot for hosts that keep
    /// their own registry keyed by tool name and want the entry to travel with
    /// the discovery record. Defaults to [`serde_json::Value::Null`].
    pub context: serde_json::Value,
    /// The documents embedded to make this tool discoverable.
    pub embedding_docs: Vec<String>,
}

impl Embed for ToolSchema {
    fn embed(&self, embedder: &mut super::embed::TextEmbedder) -> Result<(), EmbedError> {
        for doc in &self.embedding_docs {
            embedder.embed(doc.clone());
        }
        Ok(())
    }
}

impl ToolSchema {
    /// A discovery record for `name`, made discoverable by `embedding_docs`.
    ///
    /// Pass the tool's own `NAME` constant rather than a string literal:
    /// `name` is what retrieval returns and what `RequestPatch::active_tools`
    /// matches against, so a name that does not match a registered tool yields
    /// a record that can be retrieved but never activated — with no error.
    ///
    /// ```rust
    /// # use rig_core::{embeddings::ToolSchema, tool::PortableTool};
    /// # fn build<T: PortableTool>() -> ToolSchema {
    /// ToolSchema::new(T::NAME, vec!["…".to_string()])
    /// # }
    /// ```
    ///
    /// # Example
    /// ```rust
    /// use rig_core::embeddings::ToolSchema;
    ///
    /// let tool = ToolSchema::new("nothing", vec!["Do nothing.".to_string()]);
    ///
    /// assert_eq!(tool.name, "nothing".to_string());
    /// assert_eq!(tool.embedding_docs, vec!["Do nothing.".to_string()]);
    /// assert!(tool.context.is_null());
    /// ```
    pub fn new(name: impl Into<String>, embedding_docs: Vec<String>) -> Self {
        Self {
            name: name.into(),
            context: serde_json::Value::Null,
            embedding_docs,
        }
    }

    /// Attach host-defined [`context`](Self::context) data.
    pub fn with_context(mut self, context: serde_json::Value) -> Self {
        self.context = context;
        self
    }

    /// Attach host-defined [`context`](Self::context) by serializing `context`.
    ///
    /// ```rust
    /// use rig_core::embeddings::ToolSchema;
    ///
    /// let tool = ToolSchema::new("add", vec!["Add x and y.".to_string()])
    ///     .try_with_context(&("v1", 2))
    ///     .unwrap();
    ///
    /// assert_eq!(tool.context, serde_json::json!(["v1", 2]));
    /// ```
    pub fn try_with_context<T>(mut self, context: &T) -> Result<Self, EmbedError>
    where
        T: Serialize + ?Sized,
    {
        self.context = serde_json::to_value(context).map_err(EmbedError::new)?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::ToolSchema;
    use crate::{
        Embed,
        embeddings::embed::TextEmbedder,
        tool::{PortableTool, ToolExecutionError},
    };

    struct NamedTool;

    impl PortableTool for NamedTool {
        const NAME: &'static str = "static_name";

        type Args = ();
        type Output = ();
        type Error = ToolExecutionError;

        fn description(&self) -> String {
            "A statically named tool".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({})
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok(())
        }
    }

    #[test]
    fn discovery_name_tracks_the_tool_name_constant() {
        // Retrieval returns this name and `active_tools` matches on it, so a
        // record built from the tool's own constant can always be activated.
        // Building from a literal is what silently breaks discovery.
        let schema = ToolSchema::new(NamedTool::NAME, vec!["named tool".to_string()]);

        assert_eq!(schema.name, NamedTool::NAME);
    }

    #[test]
    fn new_defaults_context_to_null() {
        let schema = ToolSchema::new("add", vec!["Add x and y.".to_string()]);

        assert_eq!(schema.name, "add");
        assert_eq!(schema.embedding_docs, vec!["Add x and y.".to_string()]);
        assert_eq!(schema.context, serde_json::Value::Null);
    }

    #[test]
    fn context_is_carried_verbatim() {
        // Rig never interprets `context`; it round-trips exactly as given.
        let attached = serde_json::json!({ "registry_key": 7 });
        let schema = ToolSchema::new("add", vec!["doc".to_string()]).with_context(attached.clone());
        assert_eq!(schema.context, attached);

        let serialized = ToolSchema::new("add", vec!["doc".to_string()])
            .try_with_context(&attached)
            .expect("value serializes");
        assert_eq!(serialized.context, attached);
    }

    #[test]
    fn every_embedding_doc_is_embedded_in_order() {
        // Discovery quality depends on all docs reaching the embedder.
        let schema = ToolSchema::new(
            "add",
            vec!["Add x and y.".to_string(), "Sum two numbers.".to_string()],
        );

        let mut embedder = TextEmbedder::default();
        schema.embed(&mut embedder).expect("embedding succeeds");

        assert_eq!(
            embedder.texts,
            vec!["Add x and y.".to_string(), "Sum two numbers.".to_string()]
        );
    }
}
