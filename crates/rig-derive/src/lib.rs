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
mod client;
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
            Err(_) => quote!(::rig_core),
        },
    }
}

#[proc_macro_derive(ProviderClient, attributes(client))]
pub fn derive_provider_client(input: TokenStream) -> TokenStream {
    client::provider_client(input)
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

/// A procedural macro that transforms a function into a `rig::tool::Tool` that can be used with a `rig::agent::Agent`.
///
/// # Examples
///
/// Basic usage:
/// ```text
/// use rig_derive::rig_tool;
///
/// #[rig_tool]
/// fn add(a: i32, b: i32) -> Result<i32, rig::tool::ToolError> {
///     Ok(a + b)
/// }
/// ```
///
/// With description:
/// ```text
/// use rig_derive::rig_tool;
///
/// #[rig_tool(description = "Perform basic arithmetic operations")]
/// fn calculator(x: i32, y: i32, operation: String) -> Result<i32, rig::tool::ToolError> {
///     match operation.as_str() {
///         "add" => Ok(x + y),
///         "subtract" => Ok(x - y),
///         "multiply" => Ok(x * y),
///         "divide" => Ok(x / y),
///         _ => Err(rig::tool::ToolError::ToolCallError("Unknown operation".into())),
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
/// fn search_docs_impl(query: String) -> Result<String, rig::tool::ToolError> {
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
/// fn string_processor(text: String, operation: String) -> Result<String, rig::tool::ToolError> {
///     match operation.as_str() {
///         "uppercase" => Ok(text.to_uppercase()),
///         "lowercase" => Ok(text.to_lowercase()),
///         "reverse" => Ok(text.chars().rev().collect()),
///         _ => Err(rig::tool::ToolError::ToolCallError("Unknown operation".into())),
///     }
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

    // Build a cleaned copy of the function with doc attrs stripped from parameters,
    // since `#[doc]` on function parameters is not allowed by the compiler.
    let cleaned_fn = {
        let mut f = input_fn.clone();
        for arg in f.sig.inputs.iter_mut() {
            if let syn::FnArg::Typed(pat_type) = arg {
                pat_type.attrs.retain(|a| !a.path().is_ident("doc"));
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
            None => quote! { format!("Function to {}", Self::NAME) },
        },
    };

    // Extract parameter names, doc comments, and build struct field tokens
    let mut param_names = Vec::new();
    let mut field_tokens = Vec::new();

    for arg in input_fn.sig.inputs.iter() {
        if let syn::FnArg::Typed(pat_type) = arg
            && let syn::Pat::Ident(param_ident) = &*pat_type.pat
        {
            let param_name = &param_ident.ident;
            let param_name_str = param_name.to_string();
            let ty = &pat_type.ty;

            // Determine the description for this field:
            // explicit params() > parameter doc comment > default
            let field_doc_attr =
                if let Some(explicit) = args.param_descriptions.get(&param_name_str) {
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
        }
    }

    // Default required to all parameters only when required(...) was omitted.
    let required_args: Vec<String> = args
        .required
        .unwrap_or_else(|| param_names.iter().map(|n| n.to_string()).collect());

    let params_struct_name = format_ident!("{}Parameters", struct_name);
    let static_name = format_ident!("{}", fn_name_str.to_uppercase());

    // Generate the call implementation based on whether the function is async
    let call_impl = if is_async {
        quote! {
            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                #fn_name(#(args.#param_names,)*).await
            }
        }
    } else {
        quote! {
            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                #fn_name(#(args.#param_names,)*)
            }
        }
    };

    let rig_core = rig_core_path();
    let schemars_crate = format!("{}::schemars", rig_core.to_string().replace(' ', ""));
    let expanded = quote! {
        #[derive(serde::Deserialize, #rig_core::schemars::JsonSchema)]
        #[schemars(crate = #schemars_crate)]
        #vis struct #params_struct_name {
            #(#field_tokens,)*
        }

        #cleaned_fn

        #[derive(Default)]
        #vis struct #struct_name;

        impl #rig_core::tool::Tool for #struct_name {
            const NAME: &'static str = #tool_name;

            type Args = #params_struct_name;
            type Output = #output_type;
            type Error = #error_type;

            fn name(&self) -> String {
                #tool_name.to_string()
            }

            async fn definition(&self, _prompt: String) -> #rig_core::completion::ToolDefinition {
                let mut schema = serde_json::to_value(
                    #rig_core::schemars::schema_for!(#params_struct_name)
                ).expect("schema serialization");
                schema["required"] = serde_json::json!([#(#required_args),*]);

                #rig_core::completion::ToolDefinition {
                    name: #tool_name.to_string(),
                    description: #tool_description.to_string(),
                    parameters: schema,
                }
            }

            #call_impl
        }

        #vis static #static_name: #struct_name = #struct_name;
    };

    TokenStream::from(expanded)
}
