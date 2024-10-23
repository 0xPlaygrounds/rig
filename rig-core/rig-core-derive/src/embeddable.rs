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

            // If there are no fields tagged with #[embed] or #[embed(embed_with = "...")], return an empty TokenStream.
            // ie. do not implement `ExtractEmbeddingFields` trait for the struct.
            if basic_target_size + custom_target_size == 0 {
                return Err(syn::Error::new_spanned(
                    name,
                    "Add at least one field tagged with #[embed] or #[embed(embed_with = \"...\")].",
                ));
            }

            quote! {
                let mut embed_targets = #basic_targets;
                embed_targets.extend(#custom_targets)
            }
        }
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "ExtractEmbeddingFields derive macro should only be used on structs",
            ))
        }
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let gen = quote! {
        // Note: `ExtractEmbeddingFields` trait is imported with the macro.

        impl #impl_generics ExtractEmbeddingFields for #name #ty_generics #where_clause {
            type Error = rig::embeddings::embeddable::ExtractEmbeddingFieldsError;

            fn extract_embedding_fields(&self) -> Result<rig::OneOrMany<String>, Self::Error> {
                #target_stream;

                rig::OneOrMany::merge(
                    embed_targets.into_iter()
                        .collect::<Result<Vec<_>, _>>()?
                ).map_err(rig::embeddings::embeddable::ExtractEmbeddingFieldsError::new)
            }
        }
    };

    Ok(gen)
}

trait StructParser {
    // Handles fields tagged with #[embed]
    fn basic(&self, generics: &mut syn::Generics) -> (TokenStream, usize);

    // Handles fields tagged with #[embed(embed_with = "...")]
    fn custom(&self) -> syn::Result<(TokenStream, usize)>;
}

impl StructParser for DataStruct {
    fn basic(&self, generics: &mut syn::Generics) -> (TokenStream, usize) {
        let embed_targets = basic_embed_fields(self)
            // Iterate over every field tagged with #[embed]
            .map(|field| {
                add_struct_bounds(generics, &field.ty);

                let field_name = &field.ident;

                quote! {
                    self.#field_name
                }
            })
            .collect::<Vec<_>>();

        if !embed_targets.is_empty() {
            (
                quote! {
                    vec![#(#embed_targets.extract_embedding_fields()),*]
                },
                embed_targets.len(),
            )
        } else {
            (
                quote! {
                    vec![]
                },
                0,
            )
        }
    }

    fn custom(&self) -> syn::Result<(TokenStream, usize)> {
        let embed_targets = custom_embed_fields(self)?
            // Iterate over every field tagged with #[embed(embed_with = "...")]
            .into_iter()
            .map(|(field, custom_func_path)| {
                let field_name = &field.ident;

                quote! {
                    #custom_func_path(self.#field_name.clone())
                }
            })
            .collect::<Vec<_>>();

        Ok(if !embed_targets.is_empty() {
            (
                quote! {
                    vec![#(#embed_targets),*]
                },
                embed_targets.len(),
            )
        } else {
            (
                quote! {
                    vec![]
                },
                0,
            )
        })
    }
}
