# Discord Agent Example

This example demonstrates how to create a Discord bot using Rig's core features for AI-powered interactions. The bot combines several key capabilities:

## Features

- **Message Processing**: Processes Discord messages using GPT-4 and maintains conversation context
- **Vector Storage**: Stores message embeddings for semantic search and context retrieval
- **Periodic Posting**: Automatically generates and posts messages at configurable intervals
- **Selective Responses**: Responds to direct mentions and maintains natural conversation flow
- **Memory & Context**: Uses previous messages and similar discussions to inform responses

## Configuration

Set the following environment variables:

```bash
DISCORD_API_TOKEN=your_discord_bot_token
DISCORD_CHANNEL_ID=target_channel_id
OPENAI_API_KEY=your_openai_api_key
POST_INTERVAL_SECS=3600  # Optional, defaults to 1 hour
```

## Usage

1. Install dependencies:
```toml
[dependencies]
serenity = "0.11"
tokio = { version = "1.0", features = ["full"] }
```

2. Run the example:
```bash
cargo run --example discord_agent
```

## Architecture

The implementation follows a modular design with clear separation of concerns:

1. **DiscordBot Structure**
   - Manages bot state and core functionality
   - Handles message processing and periodic posting
   - Maintains message history and vector store

2. **Message Processing**
   - Converts Discord messages to Rig format
   - Creates embeddings for semantic search
   - Generates contextual responses using GPT-4

3. **Vector Store Integration**
   - Stores message embeddings for semantic search
   - Retrieves similar messages for context
   - Helps maintain conversation coherence

4. **Periodic Posting**
   - Runs on configurable intervals
   - Analyzes recent discussions
   - Generates relevant conversation starters

## Response Strategy

The bot uses a smart response strategy:
- Responds to direct mentions
- Periodically joins conversations (every 5 messages)
- Uses similar message context for relevant responses
- Maintains conversation flow without being overly chatty

## Error Handling

Comprehensive error handling is implemented:
- Discord API errors
- OpenAI API errors
- Message processing errors
- Periodic posting errors

## Future Enhancements

Potential improvements to consider:
1. Multi-channel support
2. Custom response triggers
3. Role-based permissions
4. Command system
5. Enhanced context management
