extern crate proc_macro;

use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use std::{collections::HashMap, ops::Deref};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    DeriveInput, Expr, ExprLit, Lit, Meta, PathArguments, ReturnType, Token, Type,
};

mod basic;
mod custom;
mod embed;

pub(crate) const EMBED: &str = "embed";

/// References:
/// <https://doc.rust-lang.org/book/ch19-06-macros.html#how-to-write-a-custom-derive-macro>
/// <https://doc.rust-lang.org/reference/procedural-macros.html>
#[proc_macro_derive(Embed, attributes(embed))]
pub fn derive_embedding_trait(item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as DeriveInput);

    embed::expand_derive_embedding(&mut input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

struct MacroArgs {
    description: Option<String>,
    param_descriptions: HashMap<String, String>,
}

impl Parse for MacroArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut description = None;
        let mut param_descriptions = HashMap::new();

        // If the input is empty, return default values
        if input.is_empty() {
            return Ok(MacroArgs {
                description,
                param_descriptions,
            });
        }

        let meta_list: Punctuated<Meta, Token![,]> = Punctuated::parse_terminated(input)?;

        for meta in meta_list {
            match meta {
                Meta::NameValue(nv) => {
                    let ident = nv.path.get_ident().unwrap().to_string();
                    if let Expr::Lit(ExprLit {
                        lit: Lit::Str(lit_str),
                        ..
                    }) = nv.value
                    {
                        if ident.as_str() == "description" {
                            description = Some(lit_str.value());
                        }
                    }
                }
                Meta::List(list) if list.path.is_ident("params") => {
                    let nested: Punctuated<Meta, Token![,]> =
                        list.parse_args_with(Punctuated::parse_terminated)?;

                    for meta in nested {
                        if let Meta::NameValue(nv) = meta {
                            if let Expr::Lit(ExprLit {
                                lit: Lit::Str(lit_str),
                                ..
                            }) = nv.value
                            {
                                let param_name = nv.path.get_ident().unwrap().to_string();
                                param_descriptions.insert(param_name, lit_str.value());
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(MacroArgs {
            description,
            param_descriptions,
        })
    }
}

fn get_json_type(ty: &Type) -> proc_macro2::TokenStream {
    match ty {
        Type::Path(type_path) => {
            let segment = &type_path.path.segments[0];
            let type_name = segment.ident.to_string();

            // Handle Vec types
            if type_name == "Vec" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let syn::GenericArgument::Type(inner_type) = &args.args[0] {
                        let inner_json_type = get_json_type(inner_type);
                        return quote! {
                            "type": "array",
                            "items": { #inner_json_type }
                        };
                    }
                }
                return quote! { "type": "array" };
            }

            // Handle primitive types
            match type_name.as_str() {
                "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "f32" | "f64" => {
                    quote! { "type": "number" }
                }
                "String" | "str" => {
                    quote! { "type": "string" }
                }
                "bool" => {
                    quote! { "type": "boolean" }
                }
                // Handle other types as objects
                _ => {
                    quote! { "type": "object" }
                }
            }
        }
        _ => {
            quote! { "type": "object" }
        }
    }
}

/// A procedural macro that transforms a function into a `rig::tool::Tool` that can be used with a `rig::agent::Agent`.
///
/// # Examples
///
/// Basic usage:
/// ```rust
/// use rig_derive::rig_tool;
///
/// #[rig_tool]
/// fn add(a: i32, b: i32) -> Result<i32, rig::tool::ToolError> {
///     Ok(a + b)
/// }
/// ```
///
/// With description:
/// ```rust
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
/// With parameter descriptions:
/// ```rust
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
    let is_async = input_fn.sig.asyncness.is_some();

    // Extract return type and get Output and Error types from Result<T, E>
    let return_type = &input_fn.sig.output;
    let output_type = match return_type {
        ReturnType::Type(_, ty) => {
            if let Type::Path(type_path) = ty.deref() {
                if let Some(last_segment) = type_path.path.segments.last() {
                    if last_segment.ident == "Result" {
                        if let PathArguments::AngleBracketed(args) = &last_segment.arguments {
                            if args.args.len() == 2 {
                                let output = args.args.first().unwrap();
                                let error = args.args.last().unwrap();

                                // Convert the error type to a string for comparison
                                let error_str = quote!(#error).to_string().replace(" ", "");
                                if !error_str.contains("rig::tool::ToolError") {
                                    panic!("Expected rig::tool::ToolError as second type parameter but found {}", error_str);
                                }

                                quote!(#output)
                            } else {
                                panic!("Expected Result with two type parameters");
                            }
                        } else {
                            panic!("Expected angle bracketed type parameters for Result");
                        }
                    } else {
                        panic!("Return type must be a Result");
                    }
                } else {
                    panic!("Invalid return type");
                }
            } else {
                panic!("Invalid return type");
            }
        }
        _ => panic!("Function must have a return type"),
    };

    // Generate PascalCase struct name from the function name
    let struct_name = format_ident!("{}", { fn_name_str.to_case(Case::Pascal) });

    // Use provided description or generate a default one
    let tool_description = match args.description {
        Some(desc) => quote! { #desc.to_string() },
        None => quote! { format!("Function to {}", Self::NAME) },
    };

    // Extract parameter names, types, and descriptions
    let mut param_names = Vec::new();
    let mut param_types = Vec::new();
    let mut param_descriptions = Vec::new();
    let mut json_types = Vec::new();

    for arg in input_fn.sig.inputs.iter() {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(param_ident) = &*pat_type.pat {
                let param_name = &param_ident.ident;
                let param_name_str = param_name.to_string();
                let ty = &pat_type.ty;
                let default_parameter_description = format!("Parameter {}", param_name_str);
                let description = args
                    .param_descriptions
                    .get(&param_name_str)
                    .map(|s| s.to_owned())
                    .unwrap_or(default_parameter_description);

                param_names.push(param_name);
                param_types.push(ty);
                param_descriptions.push(description);
                json_types.push(get_json_type(ty));
            }
        }
    }

    let params_struct_name = format_ident!("{}Parameters", struct_name);
    let static_name = format_ident!("{}", fn_name_str.to_uppercase());

    // Generate the call implementation based on whether the function is async
    let call_impl = if is_async {
        quote! {
            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                #fn_name(#(args.#param_names,)*).await.map_err(|e| rig::tool::ToolError::ToolCallError(e.into()))
            }
        }
    } else {
        quote! {
            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                #fn_name(#(args.#param_names,)*).map_err(|e| rig::tool::ToolError::ToolCallError(e.into()))
            }
        }
    };

    let expanded = quote! {
        #[derive(serde::Deserialize)]
        struct #params_struct_name {
            #(#param_names: #param_types,)*
        }

        #input_fn

        #[derive(Default)]
        pub(crate) struct #struct_name;

        impl rig::tool::Tool for #struct_name {
            const NAME: &'static str = #fn_name_str;

            type Args = #params_struct_name;
            type Output = #output_type;
            type Error = rig::tool::ToolError;

            fn name(&self) -> String {
                #fn_name_str.to_string()
            }

            async fn definition(&self, _prompt: String) -> rig::completion::ToolDefinition {
                let parameters = serde_json::json!({
                    "type": "object",
                    "properties": {
                        #(
                            stringify!(#param_names): {
                                #json_types,
                                "description": #param_descriptions
                            }
                        ),*
                    }
                });

                rig::completion::ToolDefinition {
                    name: #fn_name_str.to_string(),
                    description: #tool_description.to_string(),
                    parameters,
                }
            }

            #call_impl
        }

        pub static #static_name: #struct_name = #struct_name;
    };

    TokenStream::from(expanded)
}
