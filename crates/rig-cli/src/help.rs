//! `rig help-agents [TOPIC]` and `rig help VERB|--OPTION`. The texts are
//! [`rig_harness_protocol::guide`]'s, which the launcher prints too.

pub use rig_harness_protocol::guide::{FIRST_SCREEN, TOPICS, topic};
use rig_harness_protocol::guide::{option, option_help, verb, verb_help};

use crate::out::{self, Failure};

/// `rig help-agents [TOPIC]`: a topic, or a verb's or option's help.
pub fn agents(name: Option<&str>) -> Result<u8, Failure> {
    let Some(name) = name else {
        out::text(FIRST_SCREEN);
        return Ok(0);
    };
    let text = rig_harness_protocol::guide::help_agents(name)
        .map_err(|why| Failure::usage(why, "help-agents").next_is("rig help-agents"))?;
    out::text(&text);
    Ok(0)
}

/// `rig help VERB` or `rig help --OPTION`.
pub fn help(name: Option<&str>) -> Result<u8, Failure> {
    let Some(name) = name else {
        out::text(FIRST_SCREEN);
        return Ok(0);
    };
    if let Some(verb) = verb(name) {
        out::text(&verb_help(verb));
        return Ok(0);
    }
    if let Some(option) = option(name) {
        out::text(&option_help(option));
        return Ok(0);
    }
    Err(
        Failure::usage(format!("no verb or option `{name}`"), "help-agents")
            .next_is("rig help-agents"),
    )
}
