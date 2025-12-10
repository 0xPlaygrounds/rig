//! Integration for deploying your Rig agents (and more) as Discord bots.
//! This feature is not WASM-compatible (and as such, is incompatible with the `worker` feature).
use crate::OneOrMany;
use crate::agent::Agent;
use crate::completion::{AssistantContent, CompletionModel, request::Chat};
use crate::message::{Message as RigMessage, UserContent};
use serenity::all::{
    Command, CommandInteraction, Context, CreateCommand, CreateThread, EventHandler,
    GatewayIntents, Interaction, Message, Ready, async_trait,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// Bot state containing the agent and conversation histories
struct BotState<M: CompletionModel> {
    agent: Agent<M>,
    conversations: Arc<RwLock<HashMap<u64, Vec<RigMessage>>>>,
}

impl<M: CompletionModel> BotState<M> {
    fn new(agent: Agent<M>) -> Self {
        Self {
            agent,
            conversations: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

// Event handler for the Discord bot
struct Handler<M: CompletionModel> {
    state: Arc<BotState<M>>,
}

#[async_trait]
impl<M> EventHandler for Handler<M>
where
    M: CompletionModel + Send + Sync + 'static,
{
    async fn ready(&self, ctx: Context, ready: Ready) {
        println!("{} is connected!", ready.user.name);

        let register_cmd =
            CreateCommand::new("new").description("Start a new chat session with the bot");

        // Register slash command globally
        let command = Command::create_global_command(&ctx.http, register_cmd).await;

        match command {
            Ok(cmd) => println!("Registered global command: {}", cmd.name),
            Err(e) => eprintln!("Failed to register command: {}", e),
        }
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        if let Interaction::Command(command) = interaction {
            self.handle_command(&ctx, &command).await;
        }
    }

    async fn message(&self, ctx: Context, msg: Message) {
        // Ignore bot's own messages
        if msg.author.bot {
            return;
        }

        // Only respond to messages in threads created by the bot
        let conversations = self.state.conversations.read().await;
        if conversations.contains_key(&msg.channel_id.get()) {
            drop(conversations);
            self.handle_thread_message(&ctx, &msg).await;
        }
    }
}

impl<M> Handler<M>
where
    M: CompletionModel + Send + Sync + 'static,
{
    async fn handle_command(&self, ctx: &Context, command: &CommandInteraction) {
        if command.data.name.as_str() == "new" {
            // Defer the response to prevent timeout
            if let Err(e) = command.defer(&ctx.http).await {
                eprintln!("Failed to defer command: {}", e);
                return;
            }

            // Create a new thread
            let thread_name = format!("AI Conversation - {}", command.user.name);

            let thread = match command
                .channel_id
                .create_thread(
                    &ctx.http,
                    CreateThread::new(thread_name)
                        .kind(serenity::all::ChannelType::PublicThread)
                        .auto_archive_duration(serenity::all::AutoArchiveDuration::OneDay),
                )
                .await
            {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Failed to create thread: {}", e);
                    let _ = command
                        .edit_response(
                            &ctx.http,
                            serenity::all::EditInteractionResponse::new()
                                .content("Failed to create thread. Please try again."),
                        )
                        .await;
                    return;
                }
            };

            // Initialize conversation history for this thread
            let mut conversations = self.state.conversations.write().await;
            conversations.insert(thread.id.get(), Vec::new());
            drop(conversations);

            // Edit the deferred response
            if let Err(e) = command
                .edit_response(
                    &ctx.http,
                    serenity::all::EditInteractionResponse::new()
                        .content(format!(
                            "Started a new conversation in <#{}>! Send messages there to chat with the AI.",
                            thread.id
                        ))
                )
                .await
            {
                eprintln!("Failed to edit response: {}", e);
            }

            // Send welcome message to the thread
            if let Err(e) = thread
                .send_message(
                    &ctx.http,
                    serenity::all::CreateMessage::new()
                        .content("Hello! I'm ready to help. What would you like to talk about?"),
                )
                .await
            {
                eprintln!("Failed to send welcome message: {}", e);
            }
        }
    }

    async fn handle_thread_message(&self, ctx: &Context, msg: &Message) {
        let thread_id = msg.channel_id.get();

        // Add user message to history
        {
            let mut conversations = self.state.conversations.write().await;
            if let Some(history) = conversations.get_mut(&thread_id) {
                history.push(RigMessage::User {
                    content: OneOrMany::one(UserContent::text(msg.content.clone())),
                });
            }
        }

        // Show typing indicator
        let _ = msg.channel_id.broadcast_typing(&ctx.http).await;

        // Get conversation history
        let conversations = self.state.conversations.read().await;
        let history = if let Some(history) = conversations.get(&thread_id) {
            history.clone()
        } else {
            vec![]
        };
        drop(conversations);

        // Generate response using the agent with conversation history
        let response = match self.state.agent.chat(&msg.content, history).await {
            Ok(resp) => resp,
            Err(e) => {
                eprintln!("Agent error: {}", e);
                let _ = msg
                    .channel_id
                    .say(
                        &ctx.http,
                        "Sorry, I encountered an error processing your message.",
                    )
                    .await;
                return;
            }
        };

        // Add assistant response to history
        {
            let mut conversations = self.state.conversations.write().await;
            if let Some(history) = conversations.get_mut(&thread_id) {
                history.push(RigMessage::Assistant {
                    content: OneOrMany::one(AssistantContent::text(msg.content.clone())),
                    id: None,
                });
            }
        }

        // Send response (split if too long for Discord's 2000 char limit)
        let chunks: Vec<String> = response
            .chars()
            .collect::<Vec<_>>()
            .chunks(1900)
            .map(|c| c.iter().collect())
            .collect();

        for chunk in chunks {
            if let Err(e) = msg.channel_id.say(&ctx.http, &chunk).await {
                eprintln!("Failed to send message: {}", e);
            }
        }
    }
}

/// A trait for turning a type into a `serenity` client.
///
pub trait DiscordExt: Sized + Send + Sync
where
    Self: 'static,
{
    fn into_discord_bot(
        self,
        token: &str,
    ) -> impl std::future::Future<Output = serenity::Client> + Send;

    fn into_discord_bot_from_env(
        self,
    ) -> impl std::future::Future<Output = serenity::Client> + Send {
        let token = std::env::var("DISCORD_BOT_TOKEN")
            .expect("DISCORD_BOT_TOKEN should exist as an env var");

        async move { DiscordExt::into_discord_bot(self, &token).await }
    }
}

impl<M> DiscordExt for Agent<M>
where
    M: CompletionModel + Send + Sync + 'static,
{
    async fn into_discord_bot(self, token: &str) -> serenity::Client {
        let intents = GatewayIntents::GUILDS
            | GatewayIntents::GUILD_MESSAGES
            | GatewayIntents::MESSAGE_CONTENT;

        let state = Arc::new(BotState::new(self));
        let handler = Handler {
            state: state.clone(),
        };

        serenity::Client::builder(token, intents)
            .event_handler(handler)
            .await
            .expect("Failed to create Discord client")
    }
}
