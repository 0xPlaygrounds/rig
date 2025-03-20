use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use std::{collections::HashMap, ops::Deref};
use syn::{
    parse::Parse, parse::ParseStream, parse_macro_input, punctuated::Punctuated, Expr, ExprLit,
    Lit, Meta, PathArguments, ReturnType, Token, Type,
};

struct MacroArgs {
    description: Option<String>,
    param_descriptions: HashMap<String, String>,
}

impl Parse for MacroArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut description = None;
        let mut param_descriptions = HashMap::new();

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

#[proc_macro_attribute]
pub fn rig_tool(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as MacroArgs);
    let input_fn = parse_macro_input!(input as syn::ItemFn);

    // Extract function details
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();

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

    // Use provided name or function name as default
    let tool_description = args.description.unwrap_or_default();

    // Extract parameter names, types, and descriptions
    let mut param_defs = Vec::new();
    let mut param_names = Vec::new();

    for arg in input_fn.sig.inputs.iter() {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(param_ident) = &*pat_type.pat {
                let param_name = &param_ident.ident;
                let param_name_str = param_name.to_string();
                let ty = &pat_type.ty;
                let description = args
                    .param_descriptions
                    .get(&param_name_str)
                    .map(|s| s.as_str())
                    .unwrap_or("");

                param_names.push(param_name);
                param_defs.push(quote! {
                    #[schemars(description = #description)]
                    #param_name: #ty
                });
            }
        }
    }

    // Generate the implementation
    let params_struct_name = format_ident!("{}Parameters", struct_name);
    let expanded = quote! {
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        struct #params_struct_name {
            #(#param_defs,)*
        }

        #input_fn

        #[derive(Default)]
        pub(crate) struct #struct_name;

        impl rig::tool::Tool for #struct_name {
            const NAME: &'static str = #fn_name_str;

            type Args = serde_json::Value;
            type Output = #output_type;
            type Error = rig::tool::ToolError;

            fn name(&self) -> String {
                #fn_name_str.to_string()
            }

            async fn definition(&self, _prompt: String) -> rig::completion::ToolDefinition {
                let schema = schemars::schema_for!(#params_struct_name);
                let parameters = serde_json::to_value(schema).unwrap();

                rig::completion::ToolDefinition {
                    name: #fn_name_str.to_string(),
                    description: #tool_description.to_string(),
                    parameters,
                }
            }

            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                // Extract parameters and call the function
                let params: #params_struct_name = serde_json::from_value(args).map_err(|e| rig::tool::ToolError::JsonError(e.into()))?;
                let result = #fn_name(#(params.#param_names,)*).await.map_err(|e| rig::tool::ToolError::ToolCallError(e.into()))?;

                Ok(result)
            }
        }
    };

    TokenStream::from(expanded)
}
