use rig_derive::Embed;

// Of two `#[embed(embed_with = "...")]` attributes on one field, only the
// first used to be honored; the duplicate must error.
#[derive(Embed)]
struct Doc {
    #[embed(embed_with = "first_embed")]
    #[embed(embed_with = "second_embed")]
    value: String,
}

fn main() {}
