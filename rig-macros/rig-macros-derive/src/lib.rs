extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{Attribute, Meta};

// https://doc.rust-lang.org/book/ch19-06-macros.html#how-to-write-a-custom-derive-macro
// https://doc.rust-lang.org/reference/procedural-macros.html

#[proc_macro_derive(Embedding, attributes(embed))]
pub fn derive_embed_trait(item: TokenStream) -> TokenStream {
    let ast = syn::parse(item).unwrap();

    impl_embeddable_macro(&ast)
}

fn impl_embeddable_macro(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;

    match &ast.data {
        syn::Data::Struct(data_struct) => {
            let field_to_embed = data_struct.fields.clone().into_iter().find(|field| {
                field
                    .attrs
                    .clone()
                    .into_iter()
                    .find(|attribute| match attribute {
                        Attribute {
                            meta: Meta::Path(path),
                            ..
                        } => match path.get_ident() {
                            Some(attribute_name) => attribute_name == "embed",
                            None => false,
                        },
                        _ => return false,
                    })
                    .is_some()
            });
        }
        _ => {}
    };

    let gen = quote! {
        impl Embeddable for #name {
            type Kind = String;

            fn embeddable(&self) {
                println!("{}", stringify!(#name));
            }
        }
    };
    gen.into()
}
