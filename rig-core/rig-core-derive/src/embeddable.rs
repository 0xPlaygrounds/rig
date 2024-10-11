use proc_macro2::TokenStream;
use quote::quote;
use syn::{parse_quote, parse_str, Attribute, DataStruct, Meta};

use crate::{custom::CustomAttributeParser, EMBED};
const VEC_TYPE: &str = "Vec";
const MANY_EMBEDDING: &str = "ManyEmbedding";
const SINGLE_EMBEDDING: &str = "SingleEmbedding";

pub fn expand_derive_embedding(input: &mut syn::DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;

    let (embed_targets, custom_embed_targets, embed_kind) = match &input.data {
        syn::Data::Struct(data_struct) => {
            // Handles fields tagged with #[embed]
            let embed_targets = data_struct
                .basic_embed_fields()
                .map(|field| {
                    add_struct_bounds(&mut input.generics, &field.ty);

                    let field_name = field.ident;

                    quote! {
                        self.#field_name
                    }
                })
                .collect::<Vec<_>>();

            // Handles fields tagged with #[embed(embed_with = "...")]
            let custom_embed_targets = data_struct
                .custom_embed_fields()?
                .map(|(field, _)| {
                    let field_name = field.ident;

                    quote! {
                        self.#field_name
                    }
                })
                .collect::<Vec<_>>();

            (
                embed_targets,
                custom_embed_targets,
                data_struct.embed_kind()?,
            )
        }
        _ => panic!("Embeddable trait can only be derived for structs"),
    };

    // If there are no fields tagged with #[embed] or #[embed(embed_with = "...")], return an empty TokenStream.
    // ie. do not implement Embeddable trait for the struct.
    if embed_targets.is_empty() && custom_embed_targets.is_empty() {
        return Ok(TokenStream::new());
    }

    // Import the paths to the custom functions.
    let custom_func_paths = match &input.data {
        syn::Data::Struct(data_struct) => data_struct
            .custom_embed_fields()?
            .map(|(_, custom_func_path)| {
                quote! {
                    use #custom_func_path::embeddable;
                }
            })
            .collect::<Vec<_>>(),
        _ => vec![],
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let gen = quote! {
        // Note: we do NOT import the Embeddable trait here because if there are multiple structs in the same file
        // that derive Embed, there will be import conflicts.

        #(#custom_func_paths);*

        impl #impl_generics Embeddable for #name #ty_generics #where_clause {
            type Kind = #embed_kind;
            type Error = EmbeddingGenerationError;

            fn embeddable(&self) -> Result<Vec<String>, Self::Error> {
                vec![#(#embed_targets.clone()),*].embeddable()

                // let custom_embed_targets = vec![#( embeddable( #embed_targets ); ),*]
                //     .iter()
                //     .collect::<Result<Vec<_>, _>>()?
                //     .into_iter()
                //     .flatten();

                // Ok(embed_targets.chain(custom_embed_targets).collect())
            }
        }
    };
    eprintln!("Generated code:\n{}", gen);

    Ok(gen)
}

// Adds bounds to where clause that force all fields tagged with #[embed] to implement the Embeddable trait.
fn add_struct_bounds(generics: &mut syn::Generics, field_type: &syn::Type) {
    let where_clause = generics.make_where_clause();

    where_clause.predicates.push(parse_quote! {
        #field_type: Embeddable
    });
}

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

trait StructParser {
    /// Finds and returns fields with simple #[embed] attribute tags only.
    fn basic_embed_fields(&self) -> impl Iterator<Item = syn::Field>;
    /// Finds and returns fields with #[embed(embed_with = "...")] attribute tags only.
    /// Also returns the attribute in question.
    fn custom_embed_fields(&self)
        -> syn::Result<impl Iterator<Item = (syn::Field, syn::ExprPath)>>;

    /// If the total number of fields tagged with #[embed] or #[embed(embed_with = "...")] is 1,
    /// returns the kind of embedding that field should be.
    /// If the total number of fields tagged with #[embed] or #[embed(embed_with = "...")] is greater than 1,
    /// return ManyEmbedding.
    fn embed_kind(&self) -> syn::Result<syn::Expr> {
        let fields = self
            .basic_embed_fields()
            .chain(self.custom_embed_fields()?.map(|(f, _)| f))
            .collect::<Vec<_>>();

        if fields.len() == 1 {
            fields.iter().map(embed_kind).next().unwrap()
        } else {
            parse_str("ManyEmbedding")
        }
    }
}

impl StructParser for DataStruct {
    fn basic_embed_fields(&self) -> impl Iterator<Item = syn::Field> {
        self.fields.clone().into_iter().filter(|field| {
            field
                .attrs
                .clone()
                .into_iter()
                .any(|attribute| match attribute {
                    Attribute {
                        meta: Meta::Path(path),
                        ..
                    } => path.is_ident(EMBED),
                    _ => false,
                })
        })
    }

    fn custom_embed_fields(
        &self,
    ) -> syn::Result<impl Iterator<Item = (syn::Field, syn::ExprPath)>> {
        Ok(self
            .fields
            .clone()
            .into_iter()
            .map(|field| {
                field
                    .attrs
                    .clone()
                    .into_iter()
                    .map(|attribute| {
                        if attribute.is_custom()? {
                            Ok::<_, syn::Error>(Some((field.clone(), attribute.expand_tag()?)))
                        } else {
                            Ok(None)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .flatten())
    }
}
