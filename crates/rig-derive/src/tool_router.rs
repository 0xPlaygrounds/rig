//! Expansion of `#[derive(ToolRouter)]`.
//!
//! Generates a monomorphic inherent impl on a struct whose named fields each
//! implement the classic contextual `Tool` trait: a `catalog()` of
//! provider-facing definitions in field order, a `dispatch()` that routes one
//! model tool call to the matching field, and a `dispatch_all()` batch driver.
//! All runtime behavior (argument parsing, error normalization, result
//! shaping, concurrency) lives in `rig_agent::tool::router_support`; the
//! expansion only wires fields to it.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, spanned::Spanned};

use crate::resolve::CrateRefs;

pub(crate) fn expand_tool_router(input: &DeriveInput) -> syn::Result<TokenStream> {
    let refs = CrateRefs::resolve();
    let core = &refs.core;
    let Some(agent) = &refs.agent else {
        return Err(syn::Error::new(
            input.ident.span(),
            "#[derive(ToolRouter)] requires the classic runtime: add `rig` or `rig-agent` \
             as a dependency of this crate",
        ));
    };

    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new(
            input.ident.span(),
            "#[derive(ToolRouter)] only supports structs",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new(
            input.ident.span(),
            "#[derive(ToolRouter)] requires named fields, each implementing the `Tool` trait",
        ));
    };

    let mut definitions = Vec::new();
    let mut dispatch_arms = Vec::new();
    for field in &fields.named {
        let Some(ident) = &field.ident else {
            return Err(syn::Error::new(
                field.span(),
                "#[derive(ToolRouter)] requires named fields",
            ));
        };
        let ty = &field.ty;
        definitions.push(quote! {
            #agent::tool::portable_tool_definition(&self.#ident)
        });
        // First field with a matching `NAME` wins, mirroring registration
        // order precedence.
        dispatch_arms.push(quote! {
            if name == <#ty as #agent::tool::PortableTool>::NAME {
                return #agent::tool::router_support::execute_typed(
                    &self.#ident,
                    &call.function.arguments,
                )
                .await;
            }
        });
    }

    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics #ident #ty_generics #where_clause {
            /// The provider-facing tool catalog for this router: definitions
            /// in field-declaration order, with every name executable.
            pub fn catalog(&self) -> #agent::agent::prepare::ToolCatalog {
                #agent::agent::prepare::ToolCatalog::new(::std::vec![
                    #(#definitions),*
                ])
            }

            /// Execute one model tool call against the matching field's tool.
            ///
            /// Arguments are parsed, the output normalized, and typed errors
            /// mapped exactly as the classic registry dispatch does; an
            /// unknown tool name yields the classic not-found failure. Errors
            /// are returned as failed `ToolResult`s, never panics.
            pub async fn dispatch(
                &self,
                call: &#core::message::ToolCall,
            ) -> #core::tool::ToolResult {
                let name = call.function.name.as_str();
                #(#dispatch_arms)*
                #agent::tool::router_support::not_found(name)
            }

            /// Execute a batch of pending tool calls and return their shaped
            /// tool-result contents in call order.
            ///
            /// Preresolved results are returned verbatim without executing;
            /// the rest run with at most `concurrency` (minimum 1) in flight.
            /// Infallible: like the classic loop, every tool error becomes
            /// model-visible tool-result content.
            pub async fn dispatch_all(
                &self,
                calls: &[#agent::agent::run::PendingToolCall],
                concurrency: usize,
            ) -> ::std::vec::Vec<#core::message::UserContent> {
                #agent::tool::router_support::dispatch_pending(calls, concurrency, |call| {
                    self.dispatch(call)
                })
                .await
            }
        }
    })
}
