//! User-facing integrations for classic agents.
//!
//! ```
//! use rig_agent::{Agent, integrations::cli_chatbot::ChatBot};
//! fn chatbot(agent: Agent) -> ChatBot {
//!     ChatBot::new(agent).max_turns(3)
//! }
//! ```
/// A terminal chat loop over an agent.
pub mod cli_chatbot;
