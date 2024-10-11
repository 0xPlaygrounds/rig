use proc_macro2::TokenStream;
use quote::quote;
use syn::{parse_str, DataStruct};

use crate::{
    basic::{add_struct_bounds, basic_embed_fields},
    custom::custom_embed_fields,
};
const VEC_TYPE: &str = "Vec";
const MANY_EMBEDDING: &str = "ManyEmbedding";
const SINGLE_EMBEDDING: &str = "SingleEmbedding";

pub fn expand_derive_embedding(input: &mut syn::DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;

    // Handles fields tagged with #[embed]
    let embed_targets = match &input.data {
        syn::Data::Struct(data_struct) => basic_embed_fields(data_struct)
            .map(|field| {
                add_struct_bounds(&mut input.generics, &field.ty);

                let field_name = field.ident;

                quote! {
                    self.#field_name
                }
            })
            .collect::<Vec<_>>(),
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "Embeddable derive macro should only be used on structs",
            ))
        }
    };

    let embed_targets_quote = if !embed_targets.is_empty() {
        quote! {
            vec![#(#embed_targets.embeddable()),*]
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        }
    } else {
        quote! {
            vec![]
        }
    };

    // Handles fields tagged with #[embed(embed_with = "...")]
    let custom_embed_targets = match &input.data {
        syn::Data::Struct(data_struct) => custom_embed_fields(data_struct)?
            .map(|(field, custom_func_path)| {
                let field_name = field.ident;

                quote! {
                    #custom_func_path(self.#field_name.clone())
                }
            })
            .collect::<Vec<_>>(),
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "Embeddable derive macro should only be used on structs",
            ))
        }
    };

    let custom_embed_targets_quote = if !custom_embed_targets.is_empty() {
        quote! {
            vec![#(#custom_embed_targets),*]
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        }
    } else {
        quote! {
            vec![]
        }
    };

    // If there are no fields tagged with #[embed] or #[embed(embed_with = "...")], return an empty TokenStream.
    // ie. do not implement Embeddable trait for the struct.
    if embed_targets.is_empty() && custom_embed_targets.is_empty() {
        return Ok(TokenStream::new());
    }

    // Determine whether the Embeddable::Kind should be SinleEmbedding or ManyEmbedding.
    let embed_kind = match &input.data {
        syn::Data::Struct(data_struct) => embed_kind(data_struct)?,
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "Embeddable derive macro should only be used on structs",
            ))
        }
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let gen = quote! {
        // Note: Embeddable trait is imported with the macro.

        impl #impl_generics Embeddable for #name #ty_generics #where_clause {
            type Kind = rig::embeddings::embeddable::#embed_kind;
            type Error = rig::embeddings::embeddable::EmbeddableError;

            fn embeddable(&self) -> Result<Vec<String>, Self::Error> {
                let mut embed_targets = #embed_targets_quote;

                let custom_embed_targets = #custom_embed_targets_quote;

                embed_targets.extend(custom_embed_targets);

                Ok(embed_targets)
            }
        }
    };
    eprintln!("Generated code:\n{}", gen);

    Ok(gen)
}

/// If the total number of fields tagged with #[embed] or #[embed(embed_with = "...")] is 1,
/// returns the kind of embedding that field should be.
/// If the total number of fields tagged with #[embed] or #[embed(embed_with = "...")] is greater than 1,
/// return ManyEmbedding.
fn embed_kind(data_struct: &DataStruct) -> syn::Result<syn::Expr> {
    fn embed_kind(field: &syn::Field) -> syn::Result<syn::Expr> {
        match &field.ty {
            syn::Type::Path(path) => {
                if path.path.segments.first().unwrap().ident == VEC_TYPE {
                    parse_str(MANY_EMBEDDING)
                } else {
                    parse_str(SINGLE_EMBEDDING)
                }
            }
            _ => parse_str(SINGLE_EMBEDDING),
        }
    }
    let fields = basic_embed_fields(data_struct)
        .chain(custom_embed_fields(data_struct)?.map(|(f, _)| f))
        .collect::<Vec<_>>();

    if fields.len() == 1 {
        fields.iter().map(embed_kind).next().unwrap()
    } else {
        parse_str(MANY_EMBEDDING)
    }
}
