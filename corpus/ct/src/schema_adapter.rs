//! CT-03: tool schemas adjusted per model, in the outgoing request only.
//!
//! The `[schema_adapter]` table of `settings.toml` maps model patterns to
//! profiles. Every model call gets a `schema-adapter` [`Adapter`]; at Start,
//! on the model the call finally goes to, a matching rule wraps the
//! model's handler: a layer rewrites the request's tools as the profiles
//! say (`strict-subset`, `short-descriptions`, `few-tools`), and a stream
//! rewrite maps each tool call's arguments back to the schema its tool
//! declares, so the kernel checks them against that schema and the
//! conversation keeps calls in the declared shape. Tool definitions, the
//! request the kernel built and the conversation never see the adapted
//! schemas. A model call that fails naming a tool or a schema gets a
//! warning with the schema the model was sent, once per tool, and
//! `/schema-adapter show <tool> [model]` prints what a model receives.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, PoisonError};

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::ModelSpec;
use rig_harness::rig_core::completion::{Message, ToolDefinition};
use rig_harness::rig_core::effect::{Answered, EffectId, EffectKind};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::serve::{self, AfterServe, Intercept};
use rig_harness::rig_core::streaming::{Item, Relayed, StreamEvent};
use rig_harness::rig_ecs::effects::Handler;
use serde::Deserialize;

pub mod schema;

use schema::{Changes, glob_matches};

/// The adapter's name, and its layers' on the handler.
pub const NAME: &str = "schema-adapter";
/// How many tools `few-tools` sends when a rule names no `max_tools`.
pub const DEFAULT_MAX_TOOLS: usize = 16;

/// An adaptation profile.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Profile {
    /// Refs inlined, unions of objects flattened into one object, and no
    /// `format`, `pattern`, `default`, `examples` or `additionalProperties`.
    StrictSubset,
    /// Tool and parameter descriptions cut to their first sentence.
    ShortDescriptions,
    /// At most `max_tools` tools, by recent use and `always`; the others
    /// named in one line of the system prompt.
    FewTools,
}

/// One rule: the models it is for and how their tools are adapted.
#[derive(Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
pub struct ModelRule {
    /// The models, as `vendor/model` or `model`, `*` matching any run of
    /// characters.
    pub pattern: String,
    /// The profiles applied, in any order.
    pub profiles: Vec<Profile>,
    /// For `few-tools`: how many tools are sent.
    pub max_tools: Option<usize>,
    /// For `few-tools`: tools always sent, when offered.
    pub always: Vec<String>,
}

impl ModelRule {
    /// Whether the rule is for the model `spec`.
    pub fn matches(&self, spec: &ModelSpec) -> bool {
        glob_matches(&self.pattern, &spec.reference()) || glob_matches(&self.pattern, &spec.id)
    }

    fn has(&self, profile: Profile) -> bool {
        self.profiles.contains(&profile)
    }

    /// `tool` as this rule sends it, and what changed.
    pub fn adapt_tool(&self, tool: &ToolDefinition) -> (ToolDefinition, Changes) {
        let mut changes = Changes::default();
        let mut adapted = tool.clone();
        // Shortened first, so the notes `strict-subset` adds are kept.
        if self.has(Profile::ShortDescriptions) {
            let short = schema::first_sentence(&adapted.description);
            if short != adapted.description {
                adapted.description = short;
                changes.note("shortened description");
            }
            adapted.parameters = schema::short_descriptions(&adapted.parameters, &mut changes);
        }
        if self.has(Profile::StrictSubset) {
            adapted.parameters = schema::strict_subset(&adapted.parameters, &mut changes);
        }
        (adapted, changes)
    }
}

/// The `[schema_adapter]` table of `settings.toml`:
///
/// ```toml
/// [[schema_adapter.models]]
/// pattern = "openrouter/*"
/// profiles = ["strict-subset", "few-tools"]
/// max_tools = 8
/// always = ["read", "edit"]
/// ```
///
/// The first rule whose pattern matches a model is its rule.
#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
pub struct SchemaAdapterSettings {
    /// The rules, in order.
    pub models: Vec<ModelRule>,
}

impl SchemaAdapterSettings {
    /// The rule for the model `spec`.
    pub fn rule_for(&self, spec: &ModelSpec) -> Option<&ModelRule> {
        self.models.iter().find(|rule| rule.matches(spec))
    }
}

/// The tools whose rejected schema was already reported.
#[derive(Resource, Default)]
struct Reported(HashSet<String>);

/// Adapts the tools of model requests per model.
#[derive(Default)]
pub struct SchemaAdapterPlugin;

/// The plugin's stage system, in [`PrepareSystems::Adapt`]: it adapts each
/// model call a rule is for.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchemaAdapterSystems;

impl Plugin for SchemaAdapterPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<SchemaAdapterSettings>("schema_adapter")
            .init_resource::<Reported>()
            .configure_sets(
                AgentPass,
                SchemaAdapterSystems.in_set(PrepareSystems::Adapt),
            )
            .add_systems(AgentPass, adapt.in_set(SchemaAdapterSystems))
            .add_observer(report_rejections)
            .add_command(
                "schema-adapter",
                "show <tool> [model]: the tool's schema as the model receives it",
                show,
            );
    }
}

/// Adapts each model call admitted this pass that a rule is for.
fn adapt(
    mut calls: Query<&mut Adapted, (Added<Admitted>, ModelCalls)>,
    settings: Res<SchemaAdapterSettings>,
) {
    for mut call in &mut calls {
        call.0 = adapted(call.0.clone(), &settings);
    }
}

/// `connection`, when a rule is for its model, with its requests' tools
/// adapted and its replies' calls mapped back.
fn adapted(mut connection: Connection, settings: &SchemaAdapterSettings) -> Connection {
    let Some(rule) = settings.rule_for(&connection.spec).cloned() else {
        return connection;
    };
    // The tools `strict-subset` changed, as declared, for the reply.
    let declared: Arc<Mutex<Vec<Arc<ToolDefinition>>>> = Arc::default();
    let reading = declared.clone();
    let maps_back = connection.handler.0.clone().map_stream(NAME, move |_| {
        let declared = reading.clone();
        move |item| vec![map_back(item, &declared)]
    });
    connection.handler = Handler(maps_back.layered(AdaptRequest { rule, declared }));
    connection
}

/// Rewrites a completion request's tools by a rule.
struct AdaptRequest {
    rule: ModelRule,
    declared: Arc<Mutex<Vec<Arc<ToolDefinition>>>>,
}

impl Intercept for AdaptRequest {
    fn name(&self) -> String {
        NAME.to_owned()
    }

    async fn before(&self, _id: EffectId, kind: &EffectKind) -> serve::BeforeServe {
        let EffectKind::Completion { request, stream } = kind else {
            return serve::BeforeServe::Proceed;
        };
        let mut request = (**request).clone();
        if self.rule.has(Profile::FewTools) {
            let sent = few_tools(&self.rule, &request.tools, &request.chat_history);
            let (kept, left): (Vec<_>, Vec<_>) = std::mem::take(&mut request.tools)
                .into_iter()
                .partition(|tool| sent.contains(&tool.name.as_str().to_owned()));
            request.tools = kept;
            if !left.is_empty()
                && let Some(Message::System { content }) = request
                    .chat_history
                    .first_mut()
                    .map(std::sync::Arc::make_mut)
            {
                let names: Vec<&str> = left.iter().map(|tool| tool.name.as_str()).collect();
                content.push_str(&format!(
                    "\n\nMore tools exist, left out of this request: {}.",
                    names.join(", ")
                ));
            }
        }
        let mut changed = Vec::new();
        for tool in &mut request.tools {
            let (adapted, _) = self.rule.adapt_tool(tool);
            if self.rule.has(Profile::StrictSubset) && adapted.parameters != tool.parameters {
                changed.push(tool.clone());
            }
            *tool = adapted.into();
        }
        *self.declared.lock().unwrap_or_else(PoisonError::into_inner) = changed;
        serve::BeforeServe::Patch(EffectKind::Completion {
            request: request.into(),
            stream: *stream,
        })
    }

    async fn after(
        &self,
        _id: EffectId,
        _kind: &EffectKind,
        _outcome: &Result<Answered, ErrorReport>,
    ) -> AfterServe {
        AfterServe::Keep
    }
}

/// The names of the tools `few-tools` sends of `tools`: those `always`
/// names, then the most recently called in `history`, then the others in
/// order, up to `max_tools`.
pub fn few_tools(
    rule: &ModelRule,
    tools: &[Arc<ToolDefinition>],
    history: &[std::sync::Arc<Message>],
) -> Vec<String> {
    let offered = |name: &str| tools.iter().any(|tool| tool.name.as_str() == name);
    let recent = history.iter().rev().flat_map(|message| match &**message {
        Message::Assistant(assistant) => {
            let names = assistant
                .tool_calls()
                .map(|call| call.function.name.as_str().to_owned());
            names.collect::<Vec<_>>().into_iter().rev().collect()
        }
        _ => Vec::new(),
    });
    let rest = tools.iter().map(|tool| tool.name.as_str().to_owned());
    let mut sent: Vec<String> = Vec::new();
    for name in rule.always.iter().cloned().chain(recent).chain(rest) {
        if sent.len() >= rule.max_tools.unwrap_or(DEFAULT_MAX_TOOLS) {
            break;
        }
        if offered(&name) && !sent.contains(&name) {
            sent.push(name);
        }
    }
    sent
}

/// `item` with the arguments of each finished call of a `declared` tool
/// mapped back to its declared schema, when they can be; else as sent,
/// and the kernel refuses them against that schema.
fn map_back(item: Relayed, declared: &Mutex<Vec<Arc<ToolDefinition>>>) -> Relayed {
    let declared = declared.lock().unwrap_or_else(PoisonError::into_inner);
    let remap = |call: &mut ToolCall| {
        let tool = declared.iter().find(|tool| tool.name == call.function.name);
        let arguments = serde_json::Value::Object(call.function.arguments.clone());
        let mapped = tool.and_then(|tool| schema::map_back(&tool.parameters, &arguments));
        if let Some(serde_json::Value::Object(mapped)) = mapped {
            call.function.arguments = mapped;
        }
    };
    match item {
        Relayed::Item(Item::Event(StreamEvent::End {
            part,
            content: AssistantContent::ToolCall(mut call),
        })) => {
            remap(&mut call);
            let content = AssistantContent::ToolCall(call);
            Relayed::Item(Item::Event(StreamEvent::End { part, content }))
        }
        Relayed::Done(mut response) => {
            for content in &mut Arc::make_mut(&mut response).choice {
                if let AssistantContent::ToolCall(call) = content {
                    remap(call);
                }
            }
            Relayed::Done(response)
        }
        other => other,
    }
}

/// A model call that failed naming a tool of its request, or a schema:
/// a warning per tool, once, with the schema the model was sent.
fn report_rejections(
    answered: On<Insert<Outcome>>,
    calls: Query<(&Outcome, &ModelRequest, &SentTo, Option<&ForAgent>), ModelCalls>,
    settings: Res<SchemaAdapterSettings>,
    mut reported: ResMut<Reported>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((Outcome(Err(report)), request, SentTo(connection), agent)) = calls.get(answered.entity)
    else {
        return;
    };
    let text = std::iter::once(&report.message)
        .chain(&report.source_chain)
        .cloned()
        .collect::<Vec<_>>()
        .join(": ");
    let tools = &request.request.tools;
    let named: Vec<&ToolDefinition> = tools
        .iter()
        .filter(|tool| text.contains(tool.name.as_str()))
        .map(Arc::as_ref)
        .collect();
    let rejected = match named.is_empty() && text.to_lowercase().contains("schema") {
        true => tools.iter().map(Arc::as_ref).collect(),
        false => named,
    };
    let rule = settings.rule_for(&connection.spec);
    for tool in rejected {
        if !reported.0.insert(tool.name.as_str().to_owned()) {
            continue;
        }
        let sent = rule.map_or_else(|| tool.clone(), |rule| rule.adapt_tool(tool).0);
        notices.write(Notice::warn(
            agent.map(|agent| agent.0),
            format!(
                "{NAME}: {} rejected the request over tool `{}`: {text}\nSchema sent: {}",
                connection.spec.reference(),
                tool.name.as_str(),
                sent.parameters
            ),
        ));
    }
}

/// `/schema-adapter show <tool> [model]`: the tool's schema as the model
/// (the agent's own by default) receives it, and what was changed.
fn show(
    In(args): In<CommandArgs>,
    tools: Query<&ToolDef>,
    agents: Query<&Connection>,
    models: Res<Models>,
    settings: Res<SchemaAdapterSettings>,
    mut notices: MessageWriter<Notice>,
) {
    let words: Vec<&str> = args.args.split_whitespace().collect();
    let (Some(&"show"), Some(name)) = (words.first(), words.get(1)) else {
        notices.write(Notice::error(
            args.agent,
            "usage: /schema-adapter show <tool> [model]",
        ));
        return;
    };
    let Some(tool) = tools.iter().find(|tool| tool.0.name.as_str() == *name) else {
        notices.write(Notice::error(args.agent, format!("no tool named `{name}`")));
        return;
    };
    let spec = match words.get(2) {
        Some(model) => models
            .0
            .catalog()
            .resolve(model)
            .map(|found| found.spec.clone())
            .ok(),
        None => agents
            .get(args.agent)
            .ok()
            .map(|it| it.spec.as_ref().clone()),
    };
    let Some(spec) = spec else {
        notices.write(Notice::error(
            args.agent,
            "no such model, and the agent has none",
        ));
        return;
    };
    let text = match settings.rule_for(&spec) {
        None => format!(
            "{} has no schema-adapter rule; `{name}` goes as declared:\n{:#}",
            spec.reference(),
            tool.0.parameters
        ),
        Some(rule) => {
            let (adapted, changes) = rule.adapt_tool(&tool.0);
            let left_out = match rule.has(Profile::FewTools) {
                true => "\n(few-tools: sent only when among the tools chosen for the request)",
                false => "",
            };
            format!(
                "`{name}` as {} receives it ({:?}); changed: {}{left_out}\n{}\n{:#}",
                spec.reference(),
                rule.profiles,
                changes.summary(),
                adapted.description,
                adapted.parameters
            )
        }
    };
    notices.write(Notice::info(args.agent, text));
}

#[cfg(test)]
mod tests;
