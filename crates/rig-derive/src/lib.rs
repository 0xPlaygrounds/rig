extern crate proc_macro;

use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use std::{collections::HashMap, ops::Deref};
use syn::{
    Attribute, DeriveInput, Expr, ExprLit, Ident, Lit, Meta, PathArguments, ReturnType, Token,
    Type,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

mod basic;
mod custom;
mod embed;

pub(crate) const EMBED: &str = "embed";

pub(crate) fn rig_core_path() -> proc_macro2::TokenStream {
    match proc_macro_crate::crate_name("rig-core") {
        Ok(proc_macro_crate::FoundCrate::Itself) => quote!(crate),
        Ok(proc_macro_crate::FoundCrate::Name(name)) => {
            let ident = format_ident!("{name}");
            quote!(::#ident)
        }
        Err(_) => match proc_macro_crate::crate_name("rig") {
            Ok(proc_macro_crate::FoundCrate::Itself) => quote!(crate),
            Ok(proc_macro_crate::FoundCrate::Name(name)) => {
                let ident = format_ident!("{name}");
                quote!(::#ident)
            }
            Err(_) => match proc_macro_crate::crate_name("rig-agent") {
                Ok(proc_macro_crate::FoundCrate::Itself) => quote!(crate::core),
                Ok(proc_macro_crate::FoundCrate::Name(name)) => {
                    let ident = format_ident!("{name}");
                    quote!(::#ident::core)
                }
                Err(_) => quote!(::rig_core),
            },
        },
    }
}

fn rig_agent_path() -> proc_macro2::TokenStream {
    match proc_macro_crate::crate_name("rig-agent") {
        Ok(proc_macro_crate::FoundCrate::Itself) => quote!(crate),
        Ok(proc_macro_crate::FoundCrate::Name(name)) => {
            let ident = format_ident!("{name}");
            quote!(::#ident)
        }
        Err(_) => match proc_macro_crate::crate_name("rig") {
            Ok(proc_macro_crate::FoundCrate::Itself) => quote!(crate),
            Ok(proc_macro_crate::FoundCrate::Name(name)) => {
                let ident = format_ident!("{name}");
                quote!(::#ident)
            }
            Err(_) => quote!(::rig_agent),
        },
    }
}

fn rig_agent_tool_path() -> proc_macro2::TokenStream {
    match proc_macro_crate::crate_name("rig-agent") {
        Ok(proc_macro_crate::FoundCrate::Itself) => quote!(crate::tool),
        Ok(proc_macro_crate::FoundCrate::Name(name)) => {
            let ident = format_ident!("{name}");
            quote!(::#ident::tool)
        }
        Err(_) => match proc_macro_crate::crate_name("rig") {
            // Through the facade, contextual tools implement `rig::tool::Tool`
            // (the classic contextual trait under the default `agent` feature),
            // matching the documented facade path.
            Ok(proc_macro_crate::FoundCrate::Itself) => quote!(crate::tool),
            Ok(proc_macro_crate::FoundCrate::Name(name)) => {
                let ident = format_ident!("{name}");
                quote!(::#ident::tool)
            }
            Err(_) => quote!(::rig_agent::tool),
        },
    }
}

fn rig_portable_path() -> proc_macro2::TokenStream {
    match proc_macro_crate::crate_name("rig-core") {
        Ok(proc_macro_crate::FoundCrate::Itself) => quote!(crate),
        Ok(proc_macro_crate::FoundCrate::Name(name)) => {
            let ident = format_ident!("{name}");
            quote!(::#ident)
        }
        Err(_) => match proc_macro_crate::crate_name("rig") {
            Ok(proc_macro_crate::FoundCrate::Itself) => quote!(crate::core),
            Ok(proc_macro_crate::FoundCrate::Name(name)) => {
                let ident = format_ident!("{name}");
                quote!(::#ident::core)
            }
            Err(_) => match proc_macro_crate::crate_name("rig-agent") {
                Ok(proc_macro_crate::FoundCrate::Itself) => quote!(crate::core),
                Ok(proc_macro_crate::FoundCrate::Name(name)) => {
                    let ident = format_ident!("{name}");
                    quote!(::#ident::core)
                }
                Err(_) => quote!(::rig_core),
            },
        },
    }
}

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

struct MacroArgs {
    name: Option<String>,
    description: Option<String>,
    param_descriptions: HashMap<String, String>,
    required: Option<Vec<String>>,
}

fn parse_string_literal(expr: &Expr, field_name: &str) -> syn::Result<String> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(lit_str),
            ..
        }) => Ok(lit_str.value()),
        _ => Err(syn::Error::new_spanned(
            expr,
            format!("`{field_name}` must be a string literal"),
        )),
    }
}

fn validate_explicit_tool_name(name: &str, expr: &Expr) -> syn::Result<()> {
    if name.is_empty() || name.len() > 64 {
        return Err(syn::Error::new_spanned(
            expr,
            "`name` must be between 1 and 64 characters long",
        ));
    }

    let mut chars = name.chars();
    let Some(first_char) = chars.next() else {
        return Err(syn::Error::new_spanned(
            expr,
            "`name` must be between 1 and 64 characters long",
        ));
    };

    if !first_char.is_ascii_alphabetic() && first_char != '_' {
        return Err(syn::Error::new_spanned(
            expr,
            "`name` must start with an ASCII letter or underscore",
        ));
    }

    if chars.any(|ch| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-') {
        return Err(syn::Error::new_spanned(
            expr,
            "`name` may only contain ASCII letters, digits, underscores, or hyphens",
        ));
    }

    Ok(())
}

impl Parse for MacroArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut description = None;
        let mut param_descriptions = HashMap::new();
        let mut required = None;

        // If the input is empty, return default values
        if input.is_empty() {
            return Ok(MacroArgs {
                name,
                description,
                param_descriptions,
                required,
            });
        }

        let meta_list: Punctuated<Meta, Token![,]> = Punctuated::parse_terminated(input)?;

        for meta in meta_list {
            match meta {
                Meta::NameValue(nv) => {
                    let ident = nv.path.get_ident().ok_or_else(|| {
                        syn::Error::new_spanned(
                            &nv.path,
                            "unsupported top-level #[rig_tool] argument",
                        )
                    })?;

                    match ident.to_string().as_str() {
                        "name" => {
                            let parsed_name = parse_string_literal(&nv.value, "name")?;
                            validate_explicit_tool_name(&parsed_name, &nv.value)?;
                            name = Some(parsed_name);
                        }
                        "description" => {
                            description = Some(parse_string_literal(&nv.value, "description")?);
                        }
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &nv.path,
                                format!("unsupported top-level #[rig_tool] argument `{}`", ident),
                            ));
                        }
                    }
                }
                Meta::List(list) => {
                    let ident = list.path.get_ident().ok_or_else(|| {
                        syn::Error::new_spanned(
                            &list.path,
                            "unsupported top-level #[rig_tool] argument",
                        )
                    })?;

                    match ident.to_string().as_str() {
                        "params" => {
                            let nested: Punctuated<Meta, Token![,]> =
                                list.parse_args_with(Punctuated::parse_terminated)?;

                            for meta in nested {
                                if let Meta::NameValue(nv) = meta
                                    && let Expr::Lit(ExprLit {
                                        lit: Lit::Str(lit_str),
                                        ..
                                    }) = nv.value
                                {
                                    let Some(param_ident) = nv.path.get_ident() else {
                                        return Err(syn::Error::new_spanned(
                                            &nv.path,
                                            "parameter descriptions must use identifier keys",
                                        ));
                                    };
                                    let param_name = param_ident.to_string();
                                    param_descriptions.insert(param_name, lit_str.value());
                                }
                            }
                        }
                        "required" => {
                            let required_variables: Punctuated<Ident, Token![,]> =
                                list.parse_args_with(Punctuated::parse_terminated)?;

                            required = Some(
                                required_variables
                                    .into_iter()
                                    .map(|x| x.to_string())
                                    .collect(),
                            );
                        }
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &list.path,
                                format!("unsupported top-level #[rig_tool] argument `{}`", ident),
                            ));
                        }
                    }
                }
                Meta::Path(path) => {
                    let message = if let Some(ident) = path.get_ident() {
                        format!("unsupported top-level #[rig_tool] argument `{ident}`")
                    } else {
                        "unsupported top-level #[rig_tool] argument".to_string()
                    };

                    return Err(syn::Error::new_spanned(path, message));
                }
            }
        }

        Ok(MacroArgs {
            name,
            description,
            param_descriptions,
            required,
        })
    }
}

/// Extract doc comment text from `#[doc = "..."]` attributes.
fn extract_doc_comment(attrs: &[Attribute]) -> Option<String> {
    let lines: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if !attr.path().is_ident("doc") {
                return None;
            }
            if let Meta::NameValue(nv) = &attr.meta
                && let Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) = &nv.value
            {
                return Some(s.value());
            }
            None
        })
        .collect();

    if lines.is_empty() {
        return None;
    }

    Some(
        lines
            .iter()
            .map(|l| l.strip_prefix(' ').unwrap_or(l))
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string(),
    )
}

/// Check if a type is `Option<T>`.
fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
    {
        return segment.ident == "Option";
    }
    false
}

/// Returns whether `ty` uses an unambiguous path to Rig's tool execution context.
///
/// Procedural macros cannot resolve imported type names. Matching only the last
/// `ToolContext` path segment would therefore steal unrelated application types
/// with the same name. Imported aliases use the explicit `#[rig(context)]`
/// parameter marker instead.
fn is_tool_context_type(ty: &Type) -> bool {
    let ty = match ty {
        Type::Group(group) => &*group.elem,
        Type::Paren(paren) => &*paren.elem,
        ty => ty,
    };

    let Type::Path(type_path) = ty else {
        return false;
    };
    let segments = type_path
        .path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>();

    matches!(
        segments.as_slice(),
        [root, tool, context]
            if matches!(root.as_str(), "rig" | "rig_agent")
                && tool == "tool"
                && context == "ToolContext"
    ) || matches!(
        segments.as_slice(),
        [root, agent, tool, context]
            if root == "rig"
                && agent == "agent"
                && tool == "tool"
                && context == "ToolContext"
    )
}

/// Whether a function parameter explicitly marks itself as Rig's runtime
/// context. The marker is removed from the emitted function.
fn has_tool_context_marker(attrs: &[Attribute]) -> syn::Result<bool> {
    let mut marked = false;
    for attr in attrs.iter().filter(|attr| attr.path().is_ident("rig")) {
        if marked {
            return Err(syn::Error::new_spanned(
                attr,
                "duplicate `#[rig(context)]` parameter marker",
            ));
        }

        let Meta::List(list) = &attr.meta else {
            return Err(syn::Error::new_spanned(
                attr,
                "expected `#[rig(context)]` on the runtime context parameter",
            ));
        };
        let marker: Ident = list.parse_args().map_err(|_| {
            syn::Error::new_spanned(
                attr,
                "expected `#[rig(context)]` on the runtime context parameter",
            )
        })?;
        if marker != "context" {
            return Err(syn::Error::new_spanned(
                marker,
                "the only supported parameter marker is `#[rig(context)]`",
            ));
        }
        marked = true;
    }
    Ok(marked)
}

/// Classify a function parameter as the distinguished execution context.
///
/// An owned or shared `ToolContext` is almost certainly an authoring mistake:
/// tools need the exact mutable context supplied by the runtime so result
/// metadata and mutations remain visible to the caller.
fn is_tool_context_parameter(ty: &Type, explicitly_marked: bool) -> syn::Result<bool> {
    let ty = match ty {
        Type::Group(group) => &*group.elem,
        Type::Paren(paren) => &*paren.elem,
        ty => ty,
    };

    if let Type::Reference(reference) = ty
        && (explicitly_marked || is_tool_context_type(&reference.elem))
    {
        if reference.mutability.is_none() {
            return Err(syn::Error::new_spanned(
                ty,
                "a `ToolContext` parameter must have type `&mut ToolContext`",
            ));
        }

        return Ok(true);
    }

    if explicitly_marked || is_tool_context_type(ty) {
        return Err(syn::Error::new_spanned(
            ty,
            "a `ToolContext` parameter must have type `&mut ToolContext`",
        ));
    }

    Ok(false)
}

fn result_type_tokens(
    return_type: &ReturnType,
) -> syn::Result<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
    let ReturnType::Type(_, ty) = return_type else {
        return Err(syn::Error::new_spanned(
            return_type,
            "function must have a return type of Result<T, E>",
        ));
    };

    let Type::Path(type_path) = ty.deref() else {
        return Err(syn::Error::new_spanned(
            ty,
            "return type must be Result<T, E>",
        ));
    };

    let Some(last_segment) = type_path.path.segments.last() else {
        return Err(syn::Error::new_spanned(
            &type_path.path,
            "return type must be Result<T, E>",
        ));
    };

    if last_segment.ident != "Result" {
        return Err(syn::Error::new_spanned(
            &last_segment.ident,
            "return type must be Result<T, E>",
        ));
    }

    let PathArguments::AngleBracketed(args) = &last_segment.arguments else {
        return Err(syn::Error::new_spanned(
            &last_segment.arguments,
            "expected angle-bracketed type parameters for Result<T, E>",
        ));
    };

    let mut generic_args = args.args.iter();
    let Some(output) = generic_args.next() else {
        return Err(syn::Error::new_spanned(
            &args.args,
            "expected Result<T, E> with exactly two type parameters",
        ));
    };
    let Some(error) = generic_args.next() else {
        return Err(syn::Error::new_spanned(
            &args.args,
            "expected Result<T, E> with exactly two type parameters",
        ));
    };

    if generic_args.next().is_some() {
        return Err(syn::Error::new_spanned(
            &args.args,
            "expected Result<T, E> with exactly two type parameters",
        ));
    }

    Ok((quote!(#output), quote!(#error)))
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
/// With execution context:
/// ```text
/// use rig::tool::ToolContext;
/// use rig_derive::rig_tool;
///
/// #[rig_tool]
/// fn current_user(
///     // The marker is required for imported names and type aliases. A fully
///     // qualified `&mut rig::tool::ToolContext` is also recognized directly.
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
    let args = parse_macro_input!(args as MacroArgs);
    let input_fn = parse_macro_input!(input as syn::ItemFn);

    // Extract function details
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let tool_name = args.name.clone().unwrap_or_else(|| fn_name_str.clone());
    let vis = &input_fn.vis;
    let is_async = input_fn.sig.asyncness.is_some();

    // Build a cleaned copy of the function with macro-only parameter attributes
    // stripped. Neither parameter doc comments nor our context marker belongs in
    // the emitted Rust function.
    let cleaned_fn = {
        let mut f = input_fn.clone();
        for arg in f.sig.inputs.iter_mut() {
            if let syn::FnArg::Typed(pat_type) = arg {
                pat_type
                    .attrs
                    .retain(|a| !a.path().is_ident("doc") && !a.path().is_ident("rig"));
            }
        }
        f
    };

    // Extract return type and get Output and Error types from Result<T, E>
    let return_type = &input_fn.sig.output;
    let (output_type, error_type) = match result_type_tokens(return_type) {
        Ok(types) => types,
        Err(error) => return error.into_compile_error().into(),
    };

    // Generate PascalCase struct name from the function name
    let struct_name = format_ident!("{}", { fn_name_str.to_case(Case::Pascal) });

    // Tool description: explicit attribute > doc comment > default
    let fn_doc = extract_doc_comment(&input_fn.attrs);
    let tool_description = match args.description {
        Some(desc) => quote! { #desc.to_string() },
        None => match fn_doc {
            Some(doc) => quote! { #doc.to_string() },
            None => quote! { format!("Function to {}", #tool_name) },
        },
    };

    // Build model-facing fields independently from function-call arguments so
    // the host-only ToolContext never enters the generated JSON schema.
    let mut param_names = Vec::new();
    let mut field_tokens = Vec::new();
    let mut call_arguments = Vec::new();
    let mut context_param_name = None;

    for arg in input_fn.sig.inputs.iter() {
        let syn::FnArg::Typed(pat_type) = arg else {
            return syn::Error::new_spanned(arg, "tools cannot have a receiver parameter")
                .into_compile_error()
                .into();
        };

        let explicitly_marked = match has_tool_context_marker(&pat_type.attrs) {
            Ok(marked) => marked,
            Err(error) => return error.into_compile_error().into(),
        };
        let is_context = match is_tool_context_parameter(&pat_type.ty, explicitly_marked) {
            Ok(is_context) => is_context,
            Err(error) => return error.into_compile_error().into(),
        };

        if is_context {
            if context_param_name.is_some() {
                return syn::Error::new_spanned(
                    pat_type,
                    "a tool function may have at most one `&mut ToolContext` parameter",
                )
                .into_compile_error()
                .into();
            }

            if let syn::Pat::Ident(param_ident) = &*pat_type.pat {
                context_param_name = Some(param_ident.ident.to_string());
            } else {
                context_param_name = Some(String::new());
            }
            call_arguments.push(quote! { _context });
            continue;
        }

        let syn::Pat::Ident(param_ident) = &*pat_type.pat else {
            return syn::Error::new_spanned(
                &pat_type.pat,
                "tool parameters must use identifier patterns",
            )
            .into_compile_error()
            .into();
        };

        let param_name = &param_ident.ident;
        let param_name_str = param_name.to_string();
        let ty = &pat_type.ty;

        // Determine the description for this field:
        // explicit params() > parameter doc comment > default
        let field_doc_attr = if let Some(explicit) = args.param_descriptions.get(&param_name_str) {
            // Explicit override via params() — use #[schemars(description = "...")]
            quote! { #[schemars(description = #explicit)] }
        } else if let Some(doc) = extract_doc_comment(&pat_type.attrs) {
            // Doc comment on the parameter — propagate as #[doc = "..."]
            quote! { #[doc = #doc] }
        } else {
            // Default fallback
            let default_desc = format!("Parameter {param_name_str}");
            quote! { #[schemars(description = #default_desc)] }
        };

        // Auto-add #[serde(default)] for Option<T> fields
        let serde_default = if is_option_type(ty) {
            quote! { #[serde(default)] }
        } else {
            quote! {}
        };

        field_tokens.push(quote! {
            #field_doc_attr
            #serde_default
            #vis #param_name: #ty
        });

        param_names.push(param_name);
        call_arguments.push(quote! { args.#param_name });
    }

    if let Some(context_param_name) = context_param_name.as_deref() {
        let context_is_described = args.param_descriptions.contains_key(context_param_name);
        let context_is_required = args
            .required
            .as_ref()
            .is_some_and(|required| required.iter().any(|name| name == context_param_name));

        if context_is_described || context_is_required {
            return syn::Error::new_spanned(
                &input_fn.sig,
                "`ToolContext` is host-only and cannot be listed in `params(...)` or `required(...)`",
            )
            .into_compile_error()
            .into();
        }
    }

    // Default required to all parameters only when required(...) was omitted.
    let required_args: Vec<String> = args
        .required
        .unwrap_or_else(|| param_names.iter().map(|n| n.to_string()).collect());

    let params_struct_name = format_ident!("{}Parameters", struct_name);
    let static_name = format_ident!("{}", fn_name_str.to_uppercase());

    let has_context = context_param_name.is_some();
    let tool_owner = if has_context {
        rig_agent_path()
    } else {
        rig_portable_path()
    };
    let tool_module = if has_context {
        rig_agent_tool_path()
    } else {
        quote!(#tool_owner::tool)
    };
    // Contextual tools implement the classic `Tool` trait; context-free tools
    // implement the portable `PortableTool` contract owned by `rig-core`.
    let tool_trait = if has_context {
        quote!(#tool_module::Tool)
    } else {
        quote!(#tool_module::PortableTool)
    };

    // Generate the call implementation based on whether the function is async
    let call_impl = if has_context && is_async {
        quote! {
            async fn call(
                &self,
                _context: &mut #tool_module::ToolContext,
                args: Self::Args,
            ) -> Result<Self::Output, Self::Error> {
                #fn_name(#(#call_arguments),*).await
            }
        }
    } else if has_context {
        quote! {
            async fn call(
                &self,
                _context: &mut #tool_module::ToolContext,
                args: Self::Args,
            ) -> Result<Self::Output, Self::Error> {
                #fn_name(#(#call_arguments),*)
            }
        }
    } else if is_async {
        quote! {
            async fn call(
                &self,
                args: Self::Args,
            ) -> Result<Self::Output, Self::Error> {
                #fn_name(#(#call_arguments),*).await
            }
        }
    } else {
        quote! {
            async fn call(
                &self,
                args: Self::Args,
            ) -> Result<Self::Output, Self::Error> {
                #fn_name(#(#call_arguments),*)
            }
        }
    };

    // `schemars` is a portable re-export owned by `rig-core`; resolve it through
    // the portable path so contextual tools (whose `tool_owner` is the runtime
    // crate, which no longer re-exports core at its root) still find it.
    let schemars_owner = rig_portable_path();
    let schemars_crate = format!("{}::schemars", schemars_owner.to_string().replace(' ', ""));
    let expanded = quote! {
        #[derive(serde::Deserialize, #schemars_owner::schemars::JsonSchema)]
        #[schemars(crate = #schemars_crate)]
        #vis struct #params_struct_name {
            #(#field_tokens,)*
        }

        #cleaned_fn

        #[derive(Default)]
        #vis struct #struct_name;

        impl #tool_trait for #struct_name {
            const NAME: &'static str = #tool_name;

            type Args = #params_struct_name;
            type Output = #output_type;
            type Error = #error_type;

            fn description(&self) -> String {
                #tool_description.to_string()
            }

            fn parameters(&self) -> serde_json::Value {
                let mut schema = serde_json::to_value(
                    #schemars_owner::schemars::schema_for!(#params_struct_name)
                ).expect("tool parameter schema is always serializable");
                schema["required"] = serde_json::json!([#(#required_args),*]);
                schema
            }

            #call_impl
        }

        #vis static #static_name: #struct_name = #struct_name;
    };

    TokenStream::from(expanded)
}
