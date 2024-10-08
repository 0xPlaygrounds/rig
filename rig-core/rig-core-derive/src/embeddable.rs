use proc_macro::TokenStream;
use quote::quote;
use syn::{
    meta::ParseNestedMeta, parse_quote, parse_str, punctuated::Punctuated, spanned::Spanned,
    Attribute, DataStruct, ExprPath, Meta, Token,
};

const EMBED: &str = "embed";
const EMBED_WITH: &str = "embed_with";

pub fn expand_derive_embedding(input: &mut syn::DeriveInput) -> TokenStream {
    let name = &input.ident;

    let (func_calls, embed_kind) =
        match &input.data {
            syn::Data::Struct(data_struct) => {
                // Handles fields tagged with #[embed]
                let mut function_calls = data_struct
                    .basic_embed_fields()
                    .map(|field| {
                        add_struct_bounds(&mut input.generics, &field.ty);

                        let field_name = field.ident;

                        quote! {
                            self.#field_name.embeddable()
                        }
                    })
                    .collect::<Vec<_>>();

                // Handles fields tagged with #[embed(embed_with = "...")]
                function_calls.extend(data_struct.custom_embed_fields().unwrap().map(
                    |(field, _)| {
                        let field_name = field.ident;

                        quote! {
                            embeddable(&self.#field_name)
                        }
                    },
                ));

                (function_calls, data_struct.embed_kind().unwrap())
            }
            _ => panic!("Embeddable can only be derived for structs"),
        };

    // Import the paths to the custom functions.
    let custom_func_paths = match &input.data {
        syn::Data::Struct(data_struct) => data_struct
            .custom_embed_fields()
            .unwrap()
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
        use rig::embeddings::Embeddable;
        use rig::embeddings::#embed_kind;

        #(#custom_func_paths);*

        impl #impl_generics Embeddable for #name #ty_generics #where_clause {
            type Kind = #embed_kind;

            fn embeddable(&self) -> Vec<String> {
                vec![
                    #(#func_calls),*
                ].into_iter().flatten().collect()
            }
        }
    };
    eprintln!("Generated code:\n{}", gen);

    gen.into()
}

// Adds bounds to where clause that force all fields tagged with #[embed] to implement the Embeddable trait.
fn add_struct_bounds(generics: &mut syn::Generics, field_type: &syn::Type) {
    let where_clause = generics.make_where_clause();

    where_clause.predicates.push(parse_quote! {
        #field_type: Embeddable
    });
}

fn embed_kind(field: &syn::Field) -> Result<syn::Expr, syn::Error> {
    match &field.ty {
        syn::Type::Array(_) => parse_str("ManyEmbedding"),
        _ => parse_str("SingleEmbedding"),
    }
}

trait AttributeParser {
    /// Finds and returns fields with simple #[embed] attribute tags only.
    fn basic_embed_fields(&self) -> impl Iterator<Item = syn::Field>;
    /// Finds and returns fields with #[embed(embed_with = "...")] attribute tags only.
    /// Also returns the attribute in question.
    fn custom_embed_fields(
        &self,
    ) -> Result<impl Iterator<Item = (syn::Field, syn::ExprPath)>, syn::Error>;

    /// If the total number of fields tagged with #[embed] or #[embed(embed_with = "...")] is 1,
    /// returns the kind of embedding that field should be.
    /// If the total number of fields tagged with #[embed] or #[embed(embed_with = "...")] is greater than 1,
    /// return ManyEmbedding.
    fn embed_kind(&self) -> Result<syn::Expr, syn::Error> {
        let fields = self
            .basic_embed_fields()
            .chain(self.custom_embed_fields().unwrap().map(|(f, _)| f))
            .collect::<Vec<_>>();

        if fields.len() == 1 {
            fields.iter().map(embed_kind).next().unwrap()
        } else {
            parse_str("ManyEmbedding")
        }
    }
}

impl AttributeParser for DataStruct {
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
    ) -> Result<impl Iterator<Item = (syn::Field, syn::ExprPath)>, syn::Error> {
        // Determine if field is tagged with #[embed(embed_with = "...")] attribute.
        fn is_custom_embed(attribute: &syn::Attribute) -> Result<bool, syn::Error> {
            let is_custom_embed = match attribute.meta {
                Meta::List(_) => attribute
                    .parse_args_with(Punctuated::<Meta, Token![=]>::parse_terminated)?
                    .into_iter()
                    .any(|meta| meta.path().is_ident(EMBED_WITH)),
                _ => false,
            };

            Ok(attribute.path().is_ident(EMBED) && is_custom_embed)
        }

        // Get the "..." part of the #[embed(embed_with = "...")] attribute.
        // Ex: If attribute is tagged with #[embed(embed_with = "my_embed")], returns "my_embed".
        fn expand_tag(attribute: &syn::Attribute) -> Result<syn::ExprPath, syn::Error> {
            let mut custom_func_path = None;

            attribute.parse_nested_meta(|meta| {
                custom_func_path = Some(meta.function_path()?);
                Ok(())
            })?;

            match custom_func_path {
                Some(path) => Ok(path),
                None => Err(syn::Error::new(
                    attribute.span(),
                    format!(
                        "expected {} attribute to have format: `#[embed(embed_with = \"...\")]`",
                        EMBED_WITH
                    ),
                )),
            }
        }

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
                        if is_custom_embed(&attribute)? {
                            Ok::<_, syn::Error>(Some((field.clone(), expand_tag(&attribute)?)))
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

trait CustomFunction {
    fn function_path(&self) -> Result<ExprPath, syn::Error>;
}

impl CustomFunction for ParseNestedMeta<'_> {
    fn function_path(&self) -> Result<ExprPath, syn::Error> {
        // #[embed(embed_with = "...")]
        let expr = self.value().unwrap().parse::<syn::Expr>().unwrap();
        let mut value = &expr;
        while let syn::Expr::Group(e) = value {
            value = &e.expr;
        }
        let string = if let syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(lit_str),
            ..
        }) = value
        {
            let suffix = lit_str.suffix();
            if !suffix.is_empty() {
                return Err(syn::Error::new(
                    lit_str.span(),
                    format!("unexpected suffix `{}` on string literal", suffix),
                ));
            }
            lit_str.clone()
        } else {
            return Err(syn::Error::new(
                value.span(),
                format!(
                    "expected {} attribute to be a string: `{} = \"...\"`",
                    EMBED_WITH, EMBED_WITH
                ),
            ));
        };

        string.parse()
    }
}
