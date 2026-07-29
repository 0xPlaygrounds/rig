extern crate proc_macro;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

mod embed;
mod resolve;
mod tool;
mod tool_router;

//References:
//<https://doc.rust-lang.org/book/ch19-06-macros.html#how-to-write-a-custom-derive-macro>
//<https://doc.rust-lang.org/reference/procedural-macros.html>
/// A macro that allows you to implement the `rig::embedding::Embed` trait by deriving it.
/// Usage can be found below:
///
/// ```text
/// use rig::Embed;
/// use rig_derive::Embed;
///
/// #[derive(Embed)]
/// struct Foo {
///     id: String,
///     #[embed] // this helper shows which field to embed
///     description: String
///}
/// ```
#[proc_macro_derive(Embed, attributes(embed))]
pub fn derive_embedding_trait(item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as DeriveInput);

    embed::expand_derive_embedding(&mut input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// A procedural macro that transforms a function into a portable
/// `rig_core::tool::PortableTool`, or into the classic contextual
/// `rig::tool::Tool` when the function accepts classic runtime context.
///
/// # Examples
///
/// Basic usage:
/// ```text
/// use rig_derive::rig_tool;
///
/// #[rig_tool]
/// fn add(a: i32, b: i32) -> Result<i32, rig::tool::ToolExecutionError> {
///     Ok(a + b)
/// }
/// ```
///
/// With description:
/// ```text
/// use rig_derive::rig_tool;
///
/// #[rig_tool(description = "Perform basic arithmetic operations")]
/// fn calculator(x: i32, y: i32, operation: String) -> Result<i32, rig::tool::ToolExecutionError> {
///     match operation.as_str() {
///         "add" => Ok(x + y),
///         "subtract" => Ok(x - y),
///         "multiply" => Ok(x * y),
///         "divide" => Ok(x / y),
///         _ => Err(rig::tool::ToolExecutionError::other("Unknown operation")),
///     }
/// }
/// ```
///
/// With a custom tool name:
/// ```text
/// use rig_derive::rig_tool;
///
/// // Explicit names must be string literals that start with an ASCII letter
/// // or `_`, may contain ASCII letters, digits, `_`, or `-`, and be at most
/// // 64 characters long.
/// #[rig_tool(name = "search-docs", description = "Search the documentation")]
/// fn search_docs_impl(query: String) -> Result<String, rig::tool::ToolExecutionError> {
///     Ok(format!("Searching docs for {query}"))
/// }
/// ```
///
/// With parameter descriptions:
/// ```text
/// use rig_derive::rig_tool;
///
/// #[rig_tool(
///     description = "A tool that performs string operations",
///     params(
///         text = "The input text to process",
///         operation = "The operation to perform (uppercase, lowercase, reverse)"
///     )
/// )]
/// fn string_processor(text: String, operation: String) -> Result<String, rig::tool::ToolExecutionError> {
///     match operation.as_str() {
///         "uppercase" => Ok(text.to_uppercase()),
///         "lowercase" => Ok(text.to_lowercase()),
///         "reverse" => Ok(text.chars().rev().collect()),
///         _ => Err(rig::tool::ToolExecutionError::other("Unknown operation")),
///     }
/// }
/// ```
///
/// # Required parameters
///
/// Required-ness is derived from the parameter types: every non-`Option`
/// parameter is required, and `Option<T>` parameters are optional (absent
/// fields deserialize to `None`). An explicit `required(...)` list overrides
/// this. A parameter *omitted* from an explicit list is deserialized with
/// `#[serde(default)]`, so its type must be `Option<T>` or implement
/// `Default` — the advertised schema and the deserializer always agree.
/// Listing an `Option<T>` parameter is a compile error (schemars and serde
/// would both silently ignore the directive). Names in `params(...)` and
/// `required(...)` must match actual parameters.
///
/// ```text
/// use rig_derive::rig_tool;
///
/// // `b` is advertised as optional and defaults to `0` when omitted.
/// #[rig_tool(required(a))]
/// fn add(a: i64, b: i64) -> Result<i64, rig::tool::ToolExecutionError> {
///     Ok(a + b)
/// }
/// ```
///
/// # Execution context
///
/// ```text
/// use rig::tool::ToolContext;
/// use rig_derive::rig_tool;
///
/// #[rig_tool]
/// fn current_user(
///     // The marker is required for imported names and type aliases. A fully
///     // qualified `&mut rig::tool::ToolContext` — including under a renamed
///     // dependency — is recognized directly.
///     #[rig(context)] context: &mut ToolContext,
///     greeting: String,
/// ) -> Result<String, rig::tool::ToolExecutionError> {
///     let user = context
///         .get::<String>()
///         .map(String::as_str)
///         .unwrap_or("guest");
///     Ok(format!("{greeting}, {user}!"))
/// }
/// ```
#[proc_macro_attribute]
pub fn rig_tool(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as tool::args::MacroArgs);
    let input_fn = parse_macro_input!(input as syn::ItemFn);

    tool::expand::expand_rig_tool(args, input_fn)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derive an inherent tool router on a struct whose named fields each
/// implement the classic contextual `rig::tool::Tool` trait.
///
/// ```text
/// use rig::tool_router::ToolRouter;
///
/// #[derive(ToolRouter)]
/// struct MyTools {
///     add: Add,       // each field implements `Tool`
///     search: Search,
/// }
/// ```
///
/// Generates, in field-declaration order:
///
/// - `fn catalog(&self) -> rig::agent::prepare::ToolCatalog` — provider-facing
///   definitions for every field, all executable;
/// - `async fn dispatch(&self, call: &rig::message::ToolCall) -> ToolResult` —
///   routes one model call to the matching field by `Tool::NAME` with the
///   classic argument parsing, error normalization, and not-found shape;
/// - `async fn dispatch_all(&self, calls: &[PendingToolCall], concurrency: usize)
///   -> Vec<UserContent>` — batch execution honoring preresolved results,
///   bounded concurrency, and call-order result shaping, infallible like the
///   classic loop (tool errors become model-visible results).
#[proc_macro_derive(ToolRouter)]
pub fn derive_tool_router(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);

    tool_router::expand_tool_router(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
