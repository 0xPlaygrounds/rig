use syn::{parse_quote, Attribute, DataStruct, Meta};

use crate::EMBED;

/// Finds and returns fields with simple `#[embed]` attribute tags only.
pub(crate) fn basic_embed_fields(data_struct: &DataStruct) -> impl Iterator<Item = &syn::Field> {
    data_struct.fields.iter().filter(|field| {
        field.attrs.iter().any(|attribute| match attribute {
            Attribute {
                meta: Meta::Path(path),
                ..
            } => path.is_ident(EMBED),
            _ => false,
        })
    })
}

/// Adds bounds to where clause that force all fields tagged with `#[embed]` to implement the `Embed` trait.
pub(crate) fn add_struct_bounds(generics: &mut syn::Generics, field_type: &syn::Type) {
    let where_clause = generics.make_where_clause();

    where_clause.predicates.push(parse_quote! {
        #field_type: Embed
    });
}
