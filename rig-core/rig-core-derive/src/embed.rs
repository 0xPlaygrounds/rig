use proc_macro2::TokenStream;
use quote::quote;
use syn::DataStruct;

use crate::{
    basic::{add_struct_bounds, basic_embed_fields},
    custom::custom_embed_fields,
};

pub(crate) fn expand_derive_embedding(input: &mut syn::DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let data = &input.data;
    let generics = &mut input.generics;

    let target_stream = match data {
        syn::Data::Struct(data_struct) => {
            let (basic_targets, basic_target_size) = data_struct.basic(generics);
            let (custom_targets, custom_target_size) = data_struct.custom()?;

            // If there are no fields tagged with `#[embed]` or `#[embed(embed_with = "...")]`, return an empty TokenStream.
            // ie. do not implement `Embed` trait for the struct.
            if basic_target_size + custom_target_size == 0 {
                return Err(syn::Error::new_spanned(
                    name,
                    "Add at least one field tagged with #[embed] or #[embed(embed_with = \"...\")].",
                ));
            }

            quote! {
                #basic_targets;
                #custom_targets;
            }
        }
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "Embed derive macro should only be used on structs",
            ))
        }
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let gen = quote! {
        // Note: `Embed` trait is imported with the macro.

        impl #impl_generics Embed for #name #ty_generics #where_clause {
            fn embed(&self, embedder: &mut rig::embeddings::embed::TextEmbedder) -> Result<(), rig::embeddings::embed::EmbedError> {
                #target_stream;

                Ok(())
            }
        }
    };

    Ok(gen)
}

trait StructParser {
    // Handles fields tagged with `#[embed]`
    fn basic(&self, generics: &mut syn::Generics) -> (TokenStream, usize);

    // Handles fields tagged with `#[embed(embed_with = "...")]`
    fn custom(&self) -> syn::Result<(TokenStream, usize)>;
}

impl StructParser for DataStruct {
    fn basic(&self, generics: &mut syn::Generics) -> (TokenStream, usize) {
        let embed_targets = basic_embed_fields(self)
            // Iterate over every field tagged with `#[embed]`
            .map(|field| {
                add_struct_bounds(generics, &field.ty);

                let field_name = &field.ident;

                quote! {
                    self.#field_name
                }
            })
            .collect::<Vec<_>>();

        (
            quote! {
                #(#embed_targets.embed(embedder)?;)*
            },
            embed_targets.len(),
        )
    }

    fn custom(&self) -> syn::Result<(TokenStream, usize)> {
        let embed_targets = custom_embed_fields(self)?
            // Iterate over every field tagged with `#[embed(embed_with = "...")]`
            .into_iter()
            .map(|(field, custom_func_path)| {
                let field_name = &field.ident;

                quote! {
                    #custom_func_path(embedder, self.#field_name.clone())?;
                }
            })
            .collect::<Vec<_>>();

        Ok((
            quote! {
                #(#embed_targets)*
            },
            embed_targets.len(),
        ))
    }
}
