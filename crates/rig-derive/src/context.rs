//! `#[derive(ContextValue)]`: the slot key a `ToolContext` value lives under.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, Expr, ExprLit, Lit, Meta};

use crate::resolve::CrateRefs;

/// Expand `#[derive(ContextValue)]`. The key defaults to the type's name;
/// `#[context(key = "…")]` overrides it. Generic types are rejected: a key
/// names one value shape, and a generic type is a family of them.
pub(crate) fn expand_derive_context_value(input: &DeriveInput) -> syn::Result<TokenStream> {
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "`ContextValue` cannot be derived for a generic type: a slot key names one value shape; implement the trait by hand for each instantiation",
        ));
    }
    let ident = &input.ident;
    let mut key = ident.to_string();
    for attr in input
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("context"))
    {
        let Meta::List(list) = &attr.meta else {
            return Err(syn::Error::new_spanned(
                attr,
                "expected `#[context(key = \"…\")]`",
            ));
        };
        let value: Meta = list.parse_args()?;
        match value {
            Meta::NameValue(name_value) if name_value.path.is_ident("key") => {
                match name_value.value {
                    Expr::Lit(ExprLit {
                        lit: Lit::Str(text),
                        ..
                    }) => {
                        let text = text.value();
                        if text.is_empty() {
                            return Err(syn::Error::new_spanned(
                                attr,
                                "a `ContextValue` key must not be empty",
                            ));
                        }
                        key = text;
                    }
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "`key` takes a string literal",
                        ));
                    }
                }
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "expected `#[context(key = \"…\")]`",
                ));
            }
        }
    }
    let core = CrateRefs::resolve().core;
    Ok(quote! {
        impl #core::tool::ContextValue for #ident {
            const KEY: &'static str = #key;
        }
    })
}
