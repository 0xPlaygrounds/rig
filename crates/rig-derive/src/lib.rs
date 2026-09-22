extern crate proc_macro;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

mod context;
mod embed;
mod resolve;
mod tool;

/// Derives `rig_core::embeddings::Embed` using fields marked with `#[embed]`.
///
/// ```
/// use rig_derive::Embed;
///
/// #[derive(Embed)]
/// struct Document {
///     #[embed]
///     description: String,
/// }
/// ```
#[proc_macro_derive(Embed, attributes(embed))]
pub fn derive_embedding_trait(item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as DeriveInput);

    embed::expand_derive_embedding(&mut input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derives `rig_core::tool::ContextValue` with the type name as its default key.
/// `#[context(key = "...")]` overrides the key. The type must support serde
/// serialization and deserialization.
///
/// ```
/// use rig_derive::ContextValue;
///
/// #[derive(serde::Serialize, serde::Deserialize, ContextValue)]
/// #[context(key = "session.id")]
/// struct SessionId(String);
/// ```
#[proc_macro_derive(ContextValue, attributes(context))]
pub fn derive_context_value(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    context::expand_derive_context_value(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Generates a tool type, parameter struct, and uppercase static from a function
/// returning `Result<T, E>`. Functions with a mutable context implement
/// `rig_core::tool::Tool`; others implement `rig_core::tool::PortableTool`.
/// Imported context names need `#[rig(context)]`; qualified Rig paths are
/// recognized directly. The context is excluded from model arguments.
///
/// Accepts `name`, `description`, `params(name = "description")`, and
/// `required(name, ...)`. Explicit names must start with an ASCII letter or `_`,
/// contain only ASCII letters, digits, `_`, or `-`, and be at most 64 bytes.
/// Description attributes override doc comments. Unknown or duplicate options
/// and parameter names are rejected.
///
/// Non-`Option` arguments are required by default. An explicit required list
/// makes omitted arguments default through serde, requiring `Default`.
/// Listing an `Option` argument as required is rejected.
///
/// ```
/// use rig_derive::rig_tool;
///
/// #[rig_tool(description = "Add integers", params(a = "First operand"), required(a))]
/// fn add(a: i32, b: i32) -> Result<i32, rig_core::tool::ToolExecutionError> {
///     Ok(a + b)
/// }
/// ```
#[proc_macro_attribute]
pub fn rig_tool(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as tool::args::MacroArgs);
    let input_fn = parse_macro_input!(input as syn::ItemFn);

    tool::expand::expand_rig_tool(&args, &input_fn)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
