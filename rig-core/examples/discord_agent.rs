use anyhow::Result;
use rig::{
    completion::Message,
    embeddings::{EmbeddingsBuilder, OneOrMany},
    providers::openai::{Client, GPT_4O, TEXT_EMBEDDING_ADA_002},
    vector_store::in_memory_store::InMemoryVectorStore,
};
use serenity::{
    async_trait,
    model::{channel::Message as DiscordMessage, gateway::Ready, id::ChannelId},
    prelude::*,
};
use std::{env, sync::Arc, time::Duration};
use tokio::{sync::Mutex, time::interval};

// Error types
#[derive(Debug, thiserror::Error)]
#[error("Discord bot error: {0}")]
struct DiscordError(String);

// Types for vector store
#[derive(Clone, serde::Serialize)]
struct StoredMessage {
    content: String,
    author: String,
    timestamp: String,
}

// Discord bot state
struct DiscordBot {
    openai_client: Arc<Client>,
    channel_id: ChannelId,
    post_interval: Duration,
    message_history: Arc<Mutex<Vec<Message>>>,
    vector_store: Arc<InMemoryVectorStore<StoredMessage>>,
}

impl DiscordBot {
    async fn new(
        openai_api_key: String,
        channel_id: ChannelId,
        post_interval: Duration,
    ) -> Result<Self, DiscordError> {
        let openai_client = Arc::new(Client::new(&openai_api_key));
        let embedding_model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

        // Initialize vector store for message embeddings
        let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
            .documents(vec![])
            .build()
            .await
            .map_err(|e| DiscordError(e.to_string()))?;

        let vector_store = Arc::new(InMemoryVectorStore::from_documents_with_id_f(
            embeddings,
            |msg: &StoredMessage| msg.content.clone(),
        ));

        Ok(Self {
            openai_client,
            channel_id,
            post_interval,
            message_history: Arc::new(Mutex::new(Vec::new())),
            vector_store,
        })
    }

    async fn process_message(&self, msg: &DiscordMessage) -> Result<Option<String>, DiscordError> {
        // Convert Discord message to our Message format
        let rig_message = Message {
            role: "user".into(),
            content: msg.content.clone(),
        };

        // Add to message history
        self.message_history.lock().await.push(rig_message.clone());

        // Create stored message for vector store
        let stored_message = StoredMessage {
            content: msg.content.clone(),
            author: msg.author.name.clone(),
            timestamp: msg.timestamp.to_string(),
        };

        // Create embeddings for the message
        let embedding_model = self.openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
        let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
            .documents(vec![stored_message])
            .build()
            .await
            .map_err(|e| DiscordError(e.to_string()))?;

        // Store in vector store
        self.vector_store
            .add_embeddings(embeddings)
            .await
            .map_err(|e| DiscordError(e.to_string()))?;

        // Generate response using GPT-4
        let chat_history = self.message_history.lock().await.clone();
        let agent = self
            .openai_client
            .agent(GPT_4O)
            .preamble(
                "You are a helpful Discord bot assistant. Keep responses concise and engaging.
                Use the chat history and similar messages from your memory to provide relevant responses.
                If you don't have a good response, it's okay to stay quiet.",
            )
            .build();

        let should_respond = msg.mentions_me().unwrap_or(false) || chat_history.len() % 5 == 0;

        if should_respond {
            // Get similar messages from vector store
            let similar_messages = self
                .vector_store
                .search(&msg.content, 3)
                .await
                .map_err(|e| DiscordError(e.to_string()))?;

            // Add context from similar messages
            let context = format!(
                "Similar messages from history:\n{}",
                similar_messages
                    .iter()
                    .map(|m| format!("- {} (by {})", m.content, m.author))
                    .collect::<Vec<_>>()
                    .join("\n")
            );

            let response = agent
                .chat(&format!("{}\n\nUser message: {}", context, msg.content), chat_history)
                .await
                .map_err(|e| DiscordError(e.to_string()))?;

            Ok(Some(response))
        } else {
            Ok(None)
        }
    }

    async fn periodic_post(&self) -> Result<String, DiscordError> {
        let chat_history = self.message_history.lock().await.clone();
        
        // Get the most discussed topics from vector store
        let topics = self
            .vector_store
            .search("", 5) // Empty query returns most representative messages
            .await
            .map_err(|e| DiscordError(e.to_string()))?;

        let agent = self
            .openai_client
            .agent(GPT_4O)
            .preamble(
                "You are a Discord bot that periodically posts engaging messages.
                Look at the recent topics of discussion and generate a thought-provoking
                message that continues or builds upon those conversations.
                Keep it concise and interesting.",
            )
            .build();

        let context = format!(
            "Recent topics of discussion:\n{}",
            topics
                .iter()
                .map(|m| format!("- {} (by {})", m.content, m.author))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let response = agent
            .chat(&context, chat_history)
            .await
            .map_err(|e| DiscordError(e.to_string()))?;

        Ok(response)
    }
}

#[async_trait]
impl EventHandler for DiscordBot {
    async fn message(&self, ctx: Context, msg: DiscordMessage) {
        if msg.author.bot || msg.channel_id != self.channel_id {
            return;
        }

        if let Ok(Some(response)) = self.process_message(&msg).await {
            if let Err(e) = msg.channel_id.say(&ctx.http, response).await {
                eprintln!("Error sending message: {}", e);
            }
        }
    }

    async fn ready(&self, ctx: Context, ready: Ready) {
        println!("{} is connected!", ready.user.name);

        // Start periodic posting
        let ctx = Arc::new(ctx);
        let bot = self.clone();
        tokio::spawn(async move {
            let mut interval = interval(bot.post_interval);
            loop {
                interval.tick().await;
                match bot.periodic_post().await {
                    Ok(message) => {
                        if let Err(e) = bot.channel_id.say(&ctx.http, message).await {
                            eprintln!("Error sending periodic message: {}", e);
                        }
                    }
                    Err(e) => eprintln!("Error generating periodic message: {}", e),
                }
            }
        });
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Load environment variables
    let discord_token = env::var("DISCORD_API_TOKEN").expect("DISCORD_API_TOKEN not set");
    let channel_id = ChannelId(
        env::var("DISCORD_CHANNEL_ID")
            .expect("DISCORD_CHANNEL_ID not set")
            .parse()?,
    );
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    
    // Configure post interval (default: 1 hour)
    let post_interval = Duration::from_secs(
        env::var("POST_INTERVAL_SECS")
            .unwrap_or_else(|_| "3600".to_string())
            .parse()?,
    );

    // Create Discord bot
    let bot = DiscordBot::new(openai_api_key, channel_id, post_interval).await?;

    // Set up Discord client
    let mut client = Client::builder(&discord_token)
        .event_handler(bot)
        .await
        .expect("Error creating client");

    // Start bot
    println!("Starting Discord bot...");
    if let Err(why) = client.start().await {
        eprintln!("Client error: {:?}", why);
    }

    Ok(())
}
