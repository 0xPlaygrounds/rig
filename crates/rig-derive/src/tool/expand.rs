//! Code generation for `#[rig_tool]`.

use convert_case::{Case, Casing};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::ops::Deref;
use syn::{Attribute, Expr, ExprLit, Lit, Meta, PathArguments, ReturnType, Type};

use crate::resolve::{CrateRefs, crate_attr_string};
use crate::tool::args::MacroArgs;
use crate::tool::classify::{has_tool_context_marker, is_tool_context_parameter};

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

/// Check if a type is `Option<T>`. Matches by the final path segment, the
/// conventional proc-macro approximation: a non-`std` type named `Option` is
/// misdetected, but such a type would break `Deserialize` expectations anyway.
fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
    {
        return segment.ident == "Option";
    }
    false
}

fn result_type_tokens(return_type: &ReturnType) -> syn::Result<(TokenStream, TokenStream)> {
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
    let (Some(output), Some(error)) = (generic_args.next(), generic_args.next()) else {
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

/// A function parameter the model supplies (everything except the context).
struct ModelParam<'a> {
    ident: &'a syn::Ident,
    ty: &'a Type,
    attrs: &'a [Attribute],
    optional: bool,
}

pub(crate) fn expand_rig_tool(args: MacroArgs, input_fn: syn::ItemFn) -> syn::Result<TokenStream> {
    let refs = CrateRefs::resolve();
    let core = &refs.core;

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

    let (output_type, error_type) = result_type_tokens(&input_fn.sig.output)?;

    // Generate PascalCase struct name from the function name
    let struct_name = format_ident!("{}", fn_name_str.to_case(Case::Pascal));

    // Tool description: explicit attribute > doc comment > default
    let fn_doc = extract_doc_comment(&input_fn.attrs);
    let tool_description = match &args.description {
        Some(desc) => quote! { #desc.to_string() },
        None => match fn_doc {
            Some(doc) => quote! { #doc.to_string() },
            None => quote! { format!("Function to {}", #tool_name) },
        },
    };

    // Every parameter is model-facing: the host-only `ToolContext` was removed,
    // and a leftover one is rejected below rather than entering the schema.
    let mut model_params = Vec::new();
    let mut call_arguments = Vec::new();

    for arg in input_fn.sig.inputs.iter() {
        let syn::FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "tools cannot have a receiver parameter",
            ));
        };

        let explicitly_marked = has_tool_context_marker(&pat_type.attrs)?;
        if is_tool_context_parameter(&pat_type.ty, explicitly_marked, &refs) {
            return Err(syn::Error::new_spanned(
                pat_type,
                "ToolContext was removed; close over your state (or use \
                 `PortableDynamicTool::new`) instead of a `&mut ToolContext` parameter",
            ));
        }

        let syn::Pat::Ident(param_ident) = &*pat_type.pat else {
            return Err(syn::Error::new_spanned(
                &pat_type.pat,
                "tool parameters must use identifier patterns",
            ));
        };

        let ident = &param_ident.ident;
        call_arguments.push(quote! { args.#ident });
        model_params.push(ModelParam {
            ident,
            ty: &pat_type.ty,
            attrs: &pat_type.attrs,
            optional: is_option_type(&pat_type.ty),
        });
    }

    // Validate `params(...)` and `required(...)` names against the actual
    // parameter list so a typo cannot silently alter the advertised schema.
    let model_names: Vec<String> = model_params
        .iter()
        .map(|param| param.ident.to_string())
        .collect();
    let validate_name = |ident: &syn::Ident| -> syn::Result<()> {
        let name = ident.to_string();
        if !model_names.contains(&name) {
            return Err(syn::Error::new_spanned(
                ident,
                format!("`{name}` does not match any parameter of `{fn_name_str}`"),
            ));
        }
        Ok(())
    };

    for (ident, _) in &args.param_descriptions {
        validate_name(ident)?;
    }
    if let Some(required) = &args.required {
        for ident in required {
            validate_name(ident)?;
            // schemars excludes `Option` fields from `required` regardless of
            // attributes, and serde deserializes a missing `Option` to `None`,
            // so listing one here would be silently ignored on both sides.
            // Reject it instead of dropping the author's directive.
            if model_params
                .iter()
                .any(|param| param.optional && param.ident == ident)
            {
                return Err(syn::Error::new_spanned(
                    ident,
                    "an `Option` parameter cannot be listed in `required(...)`; \
                     drop the `Option` or omit it",
                ));
            }
        }
    }

    // Required-ness has one source of truth: the parameter types, unless
    // `required(...)` explicitly overrides it. Either way the deserializer and
    // the advertised schema agree, because optional fields get
    // `#[serde(default)]` and schemars derives `required` from exactly that.
    let explicit_required: Option<Vec<String>> = args
        .required
        .as_ref()
        .map(|list| list.iter().map(|ident| ident.to_string()).collect());

    let field_tokens: Vec<TokenStream> = model_params
        .iter()
        .map(|param| {
            let ident = param.ident;
            let ty = param.ty;
            let name = ident.to_string();

            // Field description: explicit params() > parameter doc comment > default
            let field_doc_attr = if let Some(explicit) = args.description_for(&name) {
                quote! { #[schemars(description = #explicit)] }
            } else if let Some(doc) = extract_doc_comment(param.attrs) {
                quote! { #[doc = #doc] }
            } else {
                let default_desc = format!("Parameter {name}");
                quote! { #[schemars(description = #default_desc)] }
            };

            let is_required = match &explicit_required {
                None => !param.optional,
                Some(required) => required.contains(&name),
            };
            // A parameter the schema advertises as optional must also be
            // optional for the deserializer. Absent `Option` fields become
            // `None`; any other type falls back to its `Default` — a missing
            // `Default` impl is a compile error here rather than a runtime
            // deserialization failure when the model omits the field.
            let serde_default = (!is_required).then(|| quote! { #[serde(default)] });

            quote! {
                #field_doc_attr
                #serde_default
                #vis #ident: #ty
            }
        })
        .collect();

    let params_struct_name = format_ident!("{}Parameters", struct_name);
    let static_name = format_ident!("{}", fn_name_str.to_uppercase());

    // Every generated tool implements the portable `PortableTool` contract
    // owned by `rig-core`.
    let tool_trait = quote!(#core::tool::PortableTool);

    let await_suffix = is_async.then(|| quote!(.await));
    let call_impl = quote! {
        async fn call(
            &self,
            args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            #fn_name(#(#call_arguments),*) #await_suffix
        }
    };

    // Tools additionally get a record constructor: the tool as a
    // `PortableDynamicTool` value (name/description/parameters plus a callback
    // wrapping the annotated function). The callback converts the concrete
    // output type through its `IntoToolOutput` implementation — statically
    // dispatched, no runtime type inspection — and normalizes the concrete
    // error exactly like `PortableTool::map_error`'s default.
    let portable_constructor = {
        quote! {
            impl #struct_name {
                /// Build this tool as a runtime-authored portable record.
                ///
                /// The returned `PortableDynamicTool` carries the same name,
                /// description, and parameter schema as the trait
                /// implementation and executes the annotated function.
                #vis fn portable(self) -> #core::tool::PortableDynamicTool {
                    let description = #core::tool::PortableTool::description(&self);
                    let parameters = #core::tool::PortableTool::parameters(&self);
                    #core::tool::PortableDynamicTool::new(
                        #tool_name,
                        description,
                        parameters,
                        move |arguments| {
                            async move {
                                // Mirror the classic argument parser's `null`
                                // fallback for tools without required fields.
                                let arguments = if arguments.is_null() {
                                    #core::serde_json::Value::Object(#core::serde_json::Map::new())
                                } else {
                                    arguments
                                };
                                let args: #params_struct_name =
                                    #core::serde_json::from_value(arguments).map_err(|error| {
                                        #core::tool::ToolExecutionError::invalid_args(
                                            ::std::format!(
                                                "failed to parse tool arguments: {error}"
                                            ),
                                        )
                                        .with_source(error)
                                    })?;
                                match #fn_name(#(#call_arguments),*) #await_suffix {
                                    Ok(output) => {
                                        #core::tool::IntoToolOutput::into_tool_output(output)
                                    }
                                    Err(error) => Err(
                                        #core::tool::ToolExecutionError::from_error(error),
                                    ),
                                }
                            }
                        },
                    )
                }
            }
        }
    };

    // `serde`, `serde_json`, and `schemars` are portable re-exports owned by
    // `rig-core`; resolving them through the core namespace keeps generated
    // code independent of the calling crate's direct dependencies.
    let serde_crate = crate_attr_string(core, "serde");
    let schemars_crate = crate_attr_string(core, "schemars");

    Ok(quote! {
        #[derive(#core::serde::Deserialize, #core::schemars::JsonSchema)]
        #[serde(crate = #serde_crate)]
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
                #tool_description
            }

            fn parameters(&self) -> #core::serde_json::Value {
                static SCHEMA: ::std::sync::LazyLock<#core::serde_json::Value> =
                    ::std::sync::LazyLock::new(|| {
                        let mut schema =
                            #core::schemars::schema_for!(#params_struct_name).to_value();
                        // Providers expect an explicit `required` array even
                        // when no parameter is required.
                        if let Some(object) = schema.as_object_mut() {
                            object
                                .entry("required")
                                .or_insert_with(|| #core::serde_json::Value::Array(Vec::new()));
                        }
                        schema
                    });
                ::std::clone::Clone::clone(&*SCHEMA)
            }

            #call_impl
        }

        #portable_constructor

        #vis static #static_name: #struct_name = #struct_name;
    })
}
