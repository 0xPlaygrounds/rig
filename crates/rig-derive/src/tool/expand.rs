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

    // Classify parameters: model-facing fields are built independently from
    // function-call arguments so the host-only ToolContext never enters the
    // generated JSON schema.
    let mut model_params = Vec::new();
    let mut call_arguments = Vec::new();
    let mut context: Option<(&syn::PatType, syn::Ident)> = None;

    for arg in input_fn.sig.inputs.iter() {
        let syn::FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "tools cannot have a receiver parameter",
            ));
        };

        let explicitly_marked = has_tool_context_marker(&pat_type.attrs)?;
        let is_context = is_tool_context_parameter(&pat_type.ty, explicitly_marked, &refs)?;

        if is_context {
            if context.is_some() {
                return Err(syn::Error::new_spanned(
                    pat_type,
                    "a tool function may have at most one `&mut ToolContext` parameter",
                ));
            }
            let syn::Pat::Ident(param_ident) = &*pat_type.pat else {
                return Err(syn::Error::new_spanned(
                    &pat_type.pat,
                    "the `ToolContext` parameter must be a plain identifier",
                ));
            };
            context = Some((pat_type, param_ident.ident.clone()));
            call_arguments.push(quote! { _context });
            continue;
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

    let has_context = context.is_some();

    // Contextual tools live in the classic runtime crate; give a targeted
    // error when it is not reachable instead of emitting an unresolved path.
    let agent_root = match (&context, &refs.agent) {
        (Some(_), Some(agent)) => Some(agent.clone()),
        (Some((pat_type, _)), None) => {
            return Err(syn::Error::new_spanned(
                pat_type,
                "contextual tools (`&mut ToolContext`) require a dependency on `rig` or \
                 `rig-agent`; portable tools only need `rig-core`",
            ));
        }
        (None, _) => None,
    };

    // Validate `params(...)` and `required(...)` names against the actual
    // parameter list so a typo cannot silently alter the advertised schema.
    let model_names: Vec<String> = model_params
        .iter()
        .map(|param| param.ident.to_string())
        .collect();
    let context_name = context.as_ref().map(|(_, ident)| ident.to_string());

    let validate_name = |ident: &syn::Ident| -> syn::Result<()> {
        let name = ident.to_string();
        if Some(&name) == context_name.as_ref() {
            return Err(syn::Error::new_spanned(
                ident,
                "`ToolContext` is host-only and cannot be listed in `params(...)` or `required(...)`",
            ));
        }
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

    let tool_module = match &agent_root {
        Some(agent) => quote!(#agent::tool),
        None => quote!(#core::tool),
    };
    // Contextual tools implement the classic `Tool` trait; context-free tools
    // implement the portable `PortableTool` contract owned by `rig-core`.
    let tool_trait = if has_context {
        quote!(#tool_module::Tool)
    } else {
        quote!(#tool_module::PortableTool)
    };

    let context_arg = has_context.then(|| quote! { _context: &mut #tool_module::ToolContext, });
    let await_suffix = is_async.then(|| quote!(.await));
    let call_impl = quote! {
        async fn call(
            &self,
            #context_arg
            args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            #fn_name(#(#call_arguments),*) #await_suffix
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

        #vis static #static_name: #struct_name = #struct_name;
    })
}
