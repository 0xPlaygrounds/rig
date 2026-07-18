pub mod cli_chatbot;

#[cfg(feature = "discord-bot")]
#[cfg_attr(docsrs, doc(cfg(feature = "discord-bot")))]
pub mod discord_bot;
