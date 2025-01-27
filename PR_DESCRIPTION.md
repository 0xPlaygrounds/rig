# Add EchoChambers Integration Example

This PR adds a comprehensive example demonstrating integration with the EchoChambers API, showcasing Rig's capabilities for building AI agents that can interact with external chat platforms.

## Features

### Core Functionality
- **Message Sending**: Send messages to EchoChambers rooms with proper formatting and sender information
- **Message History**: Retrieve and analyze message history from rooms
- **Room Metrics**: Get various metrics about rooms and agent performance
- **Dynamic Tool Selection**: Uses Rig's dynamic tool system for semantic tool selection

### Tools Implemented
1. `SendMessage`
   - Sends formatted messages to specified rooms
   - Handles proper content wrapping and sender metadata
   - Includes error handling for API responses

2. `GetHistory`
   - Retrieves message history with configurable limits
   - Supports pagination and filtering
   - Preserves message metadata and timestamps

3. `GetRoomMetrics`
   - Fetches overall room statistics
   - Tracks engagement and participation metrics
   - Monitors room activity patterns

4. `GetAgentMetrics`
   - Analyzes agent performance in rooms
   - Tracks response patterns and engagement
   - Measures effectiveness of interactions

5. `GetMetricsHistory`
   - Historical analysis of room metrics
   - Trend identification and pattern recognition
   - Long-term performance tracking

### Implementation Details
- Uses vector embeddings for semantic tool selection
- Implements proper error handling and API response validation
- Follows EchoChambers API best practices
- Maintains secure handling of API keys and sensitive data

## Usage Example

```rust
// Initialize the agent with EchoChambers tools
let echochambers_agent = openai_client
    .agent(GPT_4O)
    .preamble("You are an assistant...")
    .dynamic_tools(3, index, toolset)
    .build();

// Example interaction
Tool: send_message
Inputs: {
    'room_id': 'philosophy',
    'content': 'Your message here',
    'sender': {
        'username': 'Rig_Assistant',
        'model': 'gpt-4'
    }
}
```

## Configuration
Required environment variables:
```env
OPENAI_API_KEY=your_openai_key
ECHOCHAMBERS_API_KEY=your_echochambers_key
ECHOCHAMBERS_USERNAME=your_username
ECHOCHAMBERS_MODEL=gpt-4
ECHOCHAMBERS_ROOMS=general,philosophy
ECHOCHAMBERS_MESSAGE_LIMIT=10
ECHOCHAMBERS_POLL_INTERVAL=60
```

## Benefits
1. **Enhanced AI Interaction**
   - Natural language processing for message generation
   - Context-aware responses using message history
   - Dynamic tool selection based on user intent

2. **Robust Integration**
   - Comprehensive error handling
   - Rate limiting and API best practices
   - Secure credential management

3. **Extensible Design**
   - Modular tool implementation
   - Easy to add new capabilities
   - Flexible configuration options

## Testing
- Unit tests for each tool implementation
- Integration tests with mock API responses
- Error handling verification
- Rate limiting tests

## Future Enhancements
1. Add support for more EchoChambers features
2. Implement advanced message filtering
3. Add real-time metrics monitoring
4. Enhance context management
5. Add support for multimedia content

## Breaking Changes
None. This is a new example implementation that doesn't modify existing functionality.

## Dependencies Added
- None (uses existing project dependencies)

## Security Considerations
- API keys stored in environment variables
- Proper error handling for sensitive data
- Input validation for all API calls
- Rate limiting implementation

## Documentation
- Added comprehensive README
- Included usage examples
- Documented configuration options
- Added inline code documentation

## Related Issues
Closes #XXX - Add EchoChambers integration example
