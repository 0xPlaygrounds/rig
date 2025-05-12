use rig::Embed;

#[derive(rig_derive::Embed, Debug)]
pub struct Greetings {
    #[embed]
    pub message: String,
}
