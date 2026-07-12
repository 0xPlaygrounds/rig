//! Durable interactive coding-agent host composition.
//! Security: run only inside a trusted project; the command allow-list and
//! protected paths are host policy, not a substitute for an OS sandbox.

use anyhow::Context;
use rig::{
    agent::Agent,
    client::{CompletionClient, ProviderClient},
    code_mode::{CodeModeTool, CodePermissions, LocalCommandBackend},
    providers::openai,
    skills::{InMemorySkillCatalog, Skill, SkillProvenance},
};
use rig_sqlite::{SqliteConversationMemory, SqliteSessionStore};
use std::collections::HashSet;
use tokio::io::{AsyncBufReadExt, BufReader};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let project = std::env::current_dir()?.canonicalize()?;
    let state = project.join(".rig-agent.sqlite");
    let memory = SqliteConversationMemory::open(&state).await?;
    let sessions = SqliteSessionStore::open(&state).await?;
    let skills = InMemorySkillCatalog::default();
    skills.insert(Skill {
        name: "coding".into(),
        description: "Inspect and modify this trusted project".into(),
        instructions: "Prefer read-only inspection; explain mutations and validate them.".into(),
        provenance: SkillProvenance::BuiltIn,
        assets: vec![],
        allowed_tools: vec!["code_mode".into()],
    });
    let backend = LocalCommandBackend::new(
        CodePermissions {
            programs: HashSet::from(["git".into(), "cargo".into(), "rg".into()]),
            trusted_root: project.clone(),
        },
        project.join(".rig-artifacts"),
    )
    .protect_paths([state.clone(), project.join(".git")]);
    let client = openai::Client::from_env()?;
    let agent: Agent<_> = client
        .agent(openai::GPT_4O)
        .preamble("You are a cautious interactive coding agent.")
        .memory(memory)
        .conversation("interactive-user")
        .session(sessions, "interactive-user")
        .skills(skills)
        .tool(CodeModeTool::new(backend))
        .build();

    eprintln!("Durable coding agent ready. Ctrl-D to exit.");
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        let response = agent
            .runner(line)
            .skill("coding")
            .map_err(anyhow::Error::msg)?
            .run()
            .await
            .context("agent run failed")?;
        println!("{}", response.output);
    }
    Ok(())
}
