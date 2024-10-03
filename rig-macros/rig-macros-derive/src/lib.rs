extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Attribute, DeriveInput, Meta};

// https://doc.rust-lang.org/book/ch19-06-macros.html#how-to-write-a-custom-derive-macro
// https://doc.rust-lang.org/reference/procedural-macros.html

#[proc_macro_derive(Embedding, attributes(embed))]
pub fn derive_embed_trait(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);

    impl_embeddable_macro(&input)
}

fn impl_embeddable_macro(input: &syn::DeriveInput) -> TokenStream {
    let name = &input.ident;

    let embeddings = match &input.data {
        syn::Data::Struct(data_struct) => {
            data_struct.fields.clone().into_iter().filter(|field| {
                    field
                        .attrs
                        .clone()
                        .into_iter()
                        .any(|attribute| match attribute {
                            Attribute {
                                meta: Meta::Path(path),
                                ..
                            } => match path.get_ident() {
                                Some(attribute_name) => attribute_name == "embed",
                                None => false
                            }
                            _ => false,
                        })
            }).map(|field| {
                let field_name = field.ident.expect("");

                quote! {
                    self.#field_name.embeddable()
                }

            }).collect::<Vec<_>>()
        }
        _ => vec![]
    };

    let gen = quote! {
        impl Embeddable for #name {
            type Kind = String;

            fn embeddable(&self) -> Vec<String> {
                vec![
                    #(#embeddings),*
                ].into_iter().flatten().collect()

            }
        }
    };
    gen.into()
}
