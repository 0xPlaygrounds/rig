//! Long-horizon Gemini 3.8 Flash runs, one per program of design record 0002.
//! P7 lives in `p7_testing`. Every recorded run re-sends each signature and
//! native part unchanged, and every turn's usage reconciles with billing.

use std::collections::BTreeMap;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::{FutureExt, StreamExt};
use rig::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, MultiTurnStreamItem,
};
use rig::cassette::gemini::{Exchange, Exchanges, Scripted};
use rig::cassette::http::{CassetteSpec, ProviderCassette, cassette_path};
use rig::message::{
    AssistantContent, Document, DocumentMediaType, DocumentSourceKind, Image, ImageMediaType,
    MediaDetail, Message, Text, UserContent,
};
use rig::providers::gemini::{
    self, CacheExpiry, CachedPrefix, Gemini, GeminiConfig, NewCachedContent, api,
};
use rig::streaming::{Item, StreamEvent, StreamedUserContent};
use rig::tool::{PortableTool, tool_definition};
use rig::{AgentBuilder, AgentRun, NonEmpty};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const CONFIG_PREAMBLE: &str = include_str!("config_agent_preamble.md");
const HANDBOOK: &str =
    include_str!("../../../../../../examples/gemini_frozen_api/src/bin/contract_handbook.md");
const CONTRACT_PDF: &[u8] = include_bytes!("inputs/contract.pdf");
const SIGNATURE_PAGE_PNG: &[u8] = include_bytes!("inputs/signature-page.png");

/// Runs `body` against a Gemini client on the scenario's cassette, then reads
/// the recording back and checks that it reconciles.
async fn with_gemini_long_horizon_cassette<F, Fut>(scenario: &'static str, body: F) -> Exchanges
where
    F: FnOnce(Gemini) -> Fut,
    Fut: Future<Output = ()>,
{
    let root = crate::cassettes::cassette_root();
    let cassette = ProviderCassette::start(
        &root,
        "gemini",
        CassetteSpec::new(scenario),
        gemini::BASE_URL,
    )
    .await;
    let client = GeminiConfig::new(cassette.api_key(gemini::API_KEY_ENV))
        .with_base_url(cassette.base_url())
        .client();
    let result = AssertUnwindSafe(body(client)).catch_unwind().await;
    crate::cassettes::checkpoint_attempt(&cassette, "gemini", scenario).await;
    cassette.finish_after_test(result).await;

    let exchanges = Exchanges::from_cassette(&cassette_path(&root, "gemini", scenario))
        .expect("every body parses as Google's schema");
    assert_reconciled(&exchanges);
    exchanges
}

/// Signatures and native parts are re-sent unchanged, and each turn's usage
/// reconciles: cached within input, thoughts within output, total their sum.
fn assert_reconciled(exchanges: &Exchanges) {
    assert!(
        !exchanges.is_empty(),
        "no generateContent exchange was recorded"
    );
    exchanges
        .assert_replayed_verbatim()
        .expect("every signature and native part is re-sent unchanged");
    for (turn, exchange) in exchanges.turns().iter().enumerate() {
        let usage = exchange
            .usage()
            .unwrap_or_else(|| panic!("turn {turn} reported no usage"));
        let input = usage.input_tokens.unwrap_or(0);
        let output = usage.output_tokens.unwrap_or(0);
        assert!(
            usage.cached_input_tokens.unwrap_or(0) <= input,
            "turn {turn}"
        );
        assert!(usage.reasoning_tokens.unwrap_or(0) <= output, "turn {turn}");
        assert_eq!(usage.total_tokens, Some(input + output), "turn {turn}");
    }
}

/// Every turn reads `cache`, sends none of the prefix it owns, and hits it.
fn assert_reads_cache(exchanges: &Exchanges, cache: &str) {
    for (turn, exchange) in exchanges.turns().iter().enumerate() {
        let request = &exchange.request;
        assert_eq!(
            request.cached_content.as_deref(),
            Some(cache),
            "turn {turn} skipped the cache"
        );
        assert!(
            request.tools.is_empty()
                && request.system_instruction.is_none()
                && request.tool_config.is_none(),
            "turn {turn} re-sent a prefix the cache owns"
        );
        let cached = exchange
            .usage()
            .and_then(|usage| usage.cached_input_tokens)
            .unwrap_or(0);
        assert!(cached > 0, "turn {turn} missed the cache");
    }
}

/// Implicit caching is Google's choice per request, so hits are reported,
/// not asserted.
fn report_cache_hits(scenario: &str, exchanges: &Exchanges) {
    for (turn, exchange) in exchanges.turns().iter().enumerate() {
        let usage = exchange.usage().unwrap_or_default();
        eprintln!(
            "{scenario} turn {turn}: input={:?} cached={:?}",
            usage.input_tokens, usage.cached_input_tokens
        );
    }
}

/// The parts of every candidate Google returned in `exchange`, in order.
fn reply_parts(exchange: &Exchange) -> impl Iterator<Item = &api::Part> {
    exchange
        .responses
        .iter()
        .flat_map(|response| response.candidates.iter())
        .filter_map(|candidate| candidate.content.as_ref())
        .flat_map(|content| content.parts.iter())
}

/// The parts of the newest turn of `exchange`'s request.
fn last_turn_parts(exchange: &Exchange) -> &[api::Part] {
    exchange
        .request
        .contents
        .last()
        .map(|content| content.parts.as_slice())
        .unwrap_or_default()
}

fn sets_server_side_invocations(exchange: &Exchange) -> bool {
    exchange
        .request
        .tool_config
        .as_ref()
        .and_then(|config| config.include_server_side_tool_invocations)
        == Some(true)
}

#[derive(Deserialize)]
struct CityArgs {
    city: String,
}

#[derive(Deserialize)]
struct CelsiusArgs {
    celsius: f64,
}

struct Weather;

impl PortableTool for Weather {
    const NAME: &'static str = "get_weather";
    type Args = CityArgs;
    type Output = Value;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Current weather for a city, in Celsius.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] })
    }

    async fn call(&self, args: CityArgs) -> Result<Value, Self::Error> {
        let celsius = 12.0 + (args.city.len() % 7) as f64 * 2.5;
        Ok(json!({ "city": args.city, "celsius": celsius, "sky": "clear" }))
    }
}

struct LocalTime;

impl PortableTool for LocalTime {
    const NAME: &'static str = "local_time";
    type Args = CityArgs;
    type Output = Value;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "The current local time in a city, 24h clock.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] })
    }

    async fn call(&self, args: CityArgs) -> Result<Value, Self::Error> {
        let hour = (args.city.len() * 5) % 24;
        Ok(json!({ "city": args.city, "time": format!("{hour:02}:05") }))
    }
}

struct ToFahrenheit;

impl PortableTool for ToFahrenheit {
    const NAME: &'static str = "to_fahrenheit";
    type Args = CelsiusArgs;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Convert Celsius to Fahrenheit.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "celsius": { "type": "number" } }, "required": ["celsius"] })
    }

    async fn call(&self, args: CelsiusArgs) -> Result<f64, Self::Error> {
        Ok(args.celsius * 9.0 / 5.0 + 32.0)
    }
}

// P1: a multi-tool agent over a conversation. Every call a reply makes is
// answered, in order, by the next request.
#[tokio::test]
async fn travel_assistant_answers_every_call() {
    const CITIES: [&str; 5] = ["Lisbon", "Tokyo", "Nairobi", "Reykjavik", "Buenos Aires"];
    let exchanges = with_gemini_long_horizon_cassette("long_horizon/travel_assistant", |gemini| async move {
        let agent = AgentBuilder::new(gemini.completion(gemini::GEMINI_3_8_FLASH))
            .preamble("You are a travel assistant. Use the tools; never guess weather or time.")
            .tool(Weather)
            .tool(LocalTime)
            .tool(ToFahrenheit)
            .default_max_turns(6)
            .build();
        let mut history = Vec::new();
        for city in CITIES {
            let question = format!(
                "Is it a sensible hour to call my friend in {city}, and how warm is it there in Fahrenheit?"
            );
            let response = agent.chat(question, &mut history).await.expect("turn succeeds");
            assert!(!response.output.is_empty(), "{city} got no answer");
        }
    })
    .await;
    assert!(
        exchanges.unmodeled().is_empty(),
        "{:?}",
        exchanges.unmodeled()
    );

    let mut calls = 0;
    for pair in exchanges.turns().windows(2) {
        let [reply, next] = pair else { continue };
        let called: Vec<_> = reply_parts(reply)
            .filter_map(|part| part.function_call.as_ref())
            .map(|call| call.name.clone())
            .collect();
        if called.is_empty() {
            continue;
        }
        let answered: Vec<_> = last_turn_parts(next)
            .iter()
            .filter_map(|part| part.function_response.as_ref())
            .map(|response| response.name.clone())
            .collect();
        assert_eq!(
            called, answered,
            "the next request answers every call in order"
        );
        calls += called.len();
    }
    assert!(calls >= CITIES.len() * 2, "only {calls} tool calls");
}

// P2: a chat whose history is saved to JSON and reloaded before every turn.
#[tokio::test]
async fn pair_programmer_survives_reloads() {
    const QUESTIONS: [&str; 8] = [
        "I'm writing a Rust function that parses `key=value` lines into a HashMap<String, String>. Sketch it.",
        "Make it skip blank lines and lines starting with #.",
        "Now return an error with the line number for a line without '='.",
        "Should the error type be an enum or a struct? Pick one and show it.",
        "Trim whitespace around keys and values.",
        "Reject duplicate keys, reporting both line numbers.",
        "Write two unit tests for the duplicate-key case.",
        "Summarize the final function's contract in three bullet points.",
    ];
    let exchanges =
        with_gemini_long_horizon_cassette("long_horizon/pair_programmer", |gemini| async move {
            let agent = AgentBuilder::new(gemini.completion(gemini::GEMINI_3_8_FLASH))
                .preamble("You are a concise pair programmer. Keep answers under ten lines.")
                .build();
            let mut saved = "[]".to_owned();
            for question in QUESTIONS {
                let mut history: Vec<Message> =
                    serde_json::from_str(&saved).expect("history reloads");
                agent
                    .chat(question, &mut history)
                    .await
                    .expect("turn succeeds");
                saved = serde_json::to_string(&history).expect("history saves");
                let reloaded: Vec<Message> = serde_json::from_str(&saved).expect("history reloads");
                assert_eq!(
                    serde_json::to_string(&reloaded).expect("history saves"),
                    saved
                );
            }
        })
        .await;
    assert!(
        exchanges.unmodeled().is_empty(),
        "{:?}",
        exchanges.unmodeled()
    );
    assert_eq!(exchanges.len(), QUESTIONS.len());
}

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
}

struct SearchIssues;

impl PortableTool for SearchIssues {
    const NAME: &'static str = "search_issues";
    type Args = SearchArgs;
    type Output = Value;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Search open issues. Returns id, title and labels.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "query": { "type": "string" } }, "required": ["query"] })
    }

    async fn call(&self, args: SearchArgs) -> Result<Value, Self::Error> {
        Ok(json!([
            { "id": 41, "title": format!("{} panics on empty input", args.query), "labels": ["bug"] },
            { "id": 57, "title": format!("{} is slow on large files", args.query), "labels": ["perf", "bug"] },
            { "id": 63, "title": format!("{} drops trailing comments", args.query), "labels": ["bug", "good first issue"] }
        ]))
    }
}

// P3: a streamed agent with thoughts, a function tool and code execution
// across a conversation.
#[tokio::test]
async fn streamed_triage_surfaces_every_part() {
    const QUESTIONS: [&str; 3] = [
        "Which open issues mention the parser, and how many are there per label?",
        "Do the same for the lexer, then compare the two label counts with code.",
        "Which single label is most common across both searches?",
    ];
    #[derive(Default)]
    struct Seen {
        thoughts: usize,
        text: usize,
        tool_calls: usize,
        tool_results: usize,
        hosted_code: usize,
    }
    let seen = Arc::new(Mutex::new(Seen::default()));
    let recorded = Arc::clone(&seen);
    let exchanges =
        with_gemini_long_horizon_cassette("long_horizon/streamed_triage", |gemini| async move {
            let model =
                gemini
                    .completion(gemini::GEMINI_3_8_FLASH)
                    .settings(api::RequestSettings {
                        generation_config: api::GenerationSettings {
                            thinking_config: Some(api::ThinkingConfig {
                                include_thoughts: Some(true),
                                ..Default::default()
                            }),
                            ..Default::default()
                        },
                        tools: vec![api::HostedTool {
                            code_execution: Some(api::CodeExecution::default()),
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
            let agent = AgentBuilder::new(model)
                .preamble("You triage issues. Use search_issues, then count with code.")
                .tool(SearchIssues)
                .build();
            let mut history: Vec<Message> = Vec::new();
            for question in QUESTIONS {
                let mut stream = agent
                    .prompt(question)
                    .history(history.clone())
                    .max_turns(8)
                    .stream();
                let mut finished = false;
                while let Some(item) = stream.next().await {
                    let mut seen = recorded.lock().expect("unpoisoned");
                    match item.expect("stream succeeds") {
                        MultiTurnStreamItem::StreamAssistantItem(Item::Event(event)) => match event
                        {
                            StreamEvent::Text { .. } => seen.text += 1,
                            StreamEvent::Reasoning { .. } => seen.thoughts += 1,
                            StreamEvent::End {
                                content: AssistantContent::ToolCall(_),
                                ..
                            } => seen.tool_calls += 1,
                            StreamEvent::End {
                                content: AssistantContent::Native(native),
                                ..
                            } => {
                                let part = api::Part::try_from(&native).expect("a Gemini part");
                                if part.executable_code.is_some() {
                                    seen.hosted_code += 1;
                                }
                            }
                            _ => {}
                        },
                        MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                            ..
                        }) => seen.tool_results += 1,
                        MultiTurnStreamItem::FinalResponse(response) => {
                            history.extend(response.messages.unwrap_or_default());
                            finished = true;
                        }
                        _ => {}
                    }
                }
                assert!(finished, "{question} ended without a final response");
            }
        })
        .await;
    assert!(
        exchanges.unmodeled().is_empty(),
        "{:?}",
        exchanges.unmodeled()
    );

    let seen = seen.lock().expect("unpoisoned");
    assert!(
        seen.thoughts > 0 && seen.text > 0,
        "thoughts and text stream"
    );
    assert!(seen.tool_calls > 0 && seen.tool_calls == seen.tool_results);
    assert!(
        seen.hosted_code > 0,
        "code execution surfaces as a native part"
    );
    for (turn, exchange) in exchanges.turns().iter().enumerate() {
        assert!(
            exchange.path.contains(":streamGenerateContent"),
            "turn {turn}"
        );
        assert!(
            sets_server_side_invocations(exchange),
            "turn {turn} mixes tools unannounced"
        );
    }
}

/// An in-memory repository of service configuration files.
#[derive(Clone)]
struct Workspace(Arc<Mutex<BTreeMap<String, String>>>);

impl Workspace {
    fn seeded() -> Self {
        let files = [
            (
                "RULES.md",
                "# Validation rules\n\n\
                 Every file under `config/` is one JSON object describing a service.\n\n\
                 - `name`: lowercase letters, digits and hyphens.\n\
                 - `port`: an integer from 1024 to 65535, used by no other service.\n\
                 - `replicas`: an integer from 1 to 10.\n\
                 - `depends_on`: an array of names of services declared under `config/`.\n\
                 - `timeout_ms`: an integer from 100 to 30000, a multiple of 100.\n",
            ),
            (
                "README.md",
                "Service configuration. Files: config/api.json, config/db.json, \
                 config/queue.json, config/worker.json. `run_tests` checks RULES.md.\n",
            ),
            (
                "config/api.json",
                "{\n  \"name\": \"api\",\n  \"port\": 80,\n  \"replicas\": 3,\n  \"depends_on\": [\"db\", \"cache\"],\n  \"timeout_ms\": 5000\n}\n",
            ),
            (
                "config/db.json",
                "{\n  \"name\": \"DB\",\n  \"port\": 5432,\n  \"replicas\": 0,\n  \"depends_on\": [],\n  \"timeout_ms\": 2500\n}\n",
            ),
            (
                "config/queue.json",
                "{\n  \"name\": \"queue\",\n  \"port\": 5672,\n  \"replicas\": 1,\n  \"depends_on\": [],\n  \"timeout_ms\": 1250\n}\n",
            ),
            (
                "config/worker.json",
                "{\n  \"name\": \"worker\",\n  \"port\": 5432,\n  \"replicas\": 2,\n  \"depends_on\": [\"db\", \"queue\"],\n  \"timeout_ms\": 45000\n}\n",
            ),
        ];
        Self(Arc::new(Mutex::new(
            files
                .into_iter()
                .map(|(path, contents)| (path.to_owned(), contents.to_owned()))
                .collect(),
        )))
    }

    fn files(&self) -> BTreeMap<String, String> {
        self.0.lock().expect("unpoisoned").clone()
    }

    /// The validation suite's report, and how many of its tests failed.
    fn validate(&self, filter: Option<&str>) -> (String, usize) {
        let files = self.files();
        let services: Vec<(&String, Option<serde_json::Map<String, Value>>)> = files
            .iter()
            .filter(|(path, _)| path.starts_with("config/") && path.ends_with(".json"))
            .map(|(path, contents)| (path, serde_json::from_str(contents).ok()))
            .collect();
        let names: Vec<&str> = services
            .iter()
            .filter_map(|(_, service)| service.as_ref()?.get("name")?.as_str())
            .collect();
        let mut ports: BTreeMap<u64, &String> = BTreeMap::new();
        let mut results: Vec<(String, Result<(), String>)> = Vec::new();
        for (path, service) in &services {
            let Some(service) = service else {
                results.push((
                    format!("{path}::parse"),
                    Err("not a JSON object".to_owned()),
                ));
                continue;
            };
            let int = |key: &str| service.get(key).and_then(Value::as_u64);
            let name = service
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let name_ok = !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
            results.push((
                format!("{path}::name"),
                name_ok
                    .then_some(())
                    .ok_or(format!("name {name:?} is not lowercase")),
            ));
            let port = match int("port") {
                Some(port) if !(1024..=65535).contains(&port) => {
                    Err(format!("port {port} is outside 1024..=65535"))
                }
                Some(port) => match ports.insert(port, path) {
                    Some(other) => {
                        ports.insert(port, other);
                        Err(format!("port {port} is also used by {other}"))
                    }
                    None => Ok(()),
                },
                None => Err("port is not an integer".to_owned()),
            };
            results.push((format!("{path}::port"), port));
            let replicas = match int("replicas") {
                Some(1..=10) => Ok(()),
                other => Err(format!("replicas {other:?} is outside 1..=10")),
            };
            results.push((format!("{path}::replicas"), replicas));
            let depends_on = match service.get("depends_on").and_then(Value::as_array) {
                Some(dependencies) => {
                    dependencies
                        .iter()
                        .try_for_each(|dependency| match dependency.as_str() {
                            Some(dependency) if names.contains(&dependency) => Ok(()),
                            _ => Err(format!("{dependency} is not a declared service")),
                        })
                }
                None => Err("depends_on is not an array".to_owned()),
            };
            results.push((format!("{path}::depends_on"), depends_on));
            let timeout = match int("timeout_ms") {
                Some(timeout) if (100..=30000).contains(&timeout) && timeout % 100 == 0 => Ok(()),
                other => Err(format!(
                    "timeout_ms {other:?} is not a multiple of 100 in 100..=30000"
                )),
            };
            results.push((format!("{path}::timeout_ms"), timeout));
        }
        let mut report = String::new();
        let (mut passed, mut failed) = (0, 0);
        for (test, result) in results {
            if filter.is_some_and(|filter| !test.contains(filter)) {
                continue;
            }
            match result {
                Ok(()) => {
                    passed += 1;
                    report.push_str(&format!("test {test} ... ok\n"));
                }
                Err(reason) => {
                    failed += 1;
                    report.push_str(&format!("test {test} ... FAILED: {reason}\n"));
                }
            }
        }
        report.push_str(&format!("test result: {passed} passed; {failed} failed\n"));
        (report, failed)
    }
}

#[derive(Deserialize)]
struct PathArgs {
    path: String,
}

#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    contents: String,
}

#[derive(Deserialize)]
struct TestArgs {
    filter: Option<String>,
}

struct ReadFile(Workspace);

impl PortableTool for ReadFile {
    const NAME: &'static str = "read_file";
    type Args = PathArgs;
    type Output = String;
    type Error = std::io::Error;

    fn description(&self) -> String {
        "Read a UTF-8 file from the repository.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] })
    }

    async fn call(&self, args: PathArgs) -> Result<String, std::io::Error> {
        self.0.files().remove(&args.path).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{} does not exist", args.path),
            )
        })
    }
}

struct WriteFile(Workspace);

impl PortableTool for WriteFile {
    const NAME: &'static str = "write_file";
    type Args = WriteArgs;
    type Output = String;
    type Error = std::io::Error;

    fn description(&self) -> String {
        "Replace a file's contents.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" }, "contents": { "type": "string" } },
            "required": ["path", "contents"]
        })
    }

    async fn call(&self, args: WriteArgs) -> Result<String, std::io::Error> {
        let mut files = (self.0).0.lock().expect("unpoisoned");
        files.insert(args.path.clone(), args.contents);
        Ok(format!("wrote {}", args.path))
    }
}

struct RunTests(Workspace);

impl PortableTool for RunTests {
    const NAME: &'static str = "run_tests";
    type Args = TestArgs;
    type Output = String;
    type Error = std::io::Error;

    fn description(&self) -> String {
        "Run the validation suite, optionally filtered by test name, and return its output."
            .to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "filter": { "type": "string" } } })
    }

    async fn call(&self, args: TestArgs) -> Result<String, std::io::Error> {
        Ok(self.0.validate(args.filter.as_deref()).0)
    }
}

const CONFIG_TASK: &str = "Make `run_tests` pass in this repository without changing RULES.md.";
const CONFIG_MAX_TURNS: usize = 40;

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    run: AgentRun,
    cache: Option<CachedPrefix>,
}

/// Saves the run and its cache before every model call, and stops the run
/// before call `crash_at`, as a crashed process would.
struct SaveRun {
    cache: Option<CachedPrefix>,
    saved: Arc<Mutex<Option<String>>>,
    calls: AtomicUsize,
    crash_at: Option<usize>,
}

impl AgentHook for SaveRun {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let checkpoint = Checkpoint {
            run: AgentRun::new(event.prompt.clone())
                .with_history(event.history.to_vec())
                .max_turns(CONFIG_MAX_TURNS),
            cache: self.cache.clone(),
        };
        let json = serde_json::to_string(&checkpoint).expect("checkpoint serializes");
        *self.saved.lock().expect("unpoisoned") = Some(json);
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.crash_at == Some(call) {
            return CompletionCallAction::stop("simulated crash");
        }
        CompletionCallAction::continue_run()
    }
}

fn config_agent(
    gemini: &Gemini,
    workspace: &Workspace,
    cache: Option<CachedPrefix>,
    saved: &Arc<Mutex<Option<String>>>,
    crash_at: Option<usize>,
) -> rig::Agent {
    let model = gemini.completion(gemini::GEMINI_3_8_FLASH);
    let model = match &cache {
        Some(cache) => model.cached_content(cache.clone()),
        None => model,
    };
    AgentBuilder::new(model)
        .preamble(CONFIG_PREAMBLE)
        .tool(ReadFile(workspace.clone()))
        .tool(WriteFile(workspace.clone()))
        .tool(RunTests(workspace.clone()))
        .default_max_turns(CONFIG_MAX_TURNS)
        .add_hook(SaveRun {
            cache,
            saved: Arc::clone(saved),
            calls: AtomicUsize::new(0),
            crash_at,
        })
        .build()
}

fn config_tools(workspace: &Workspace) -> [rig::completion::ToolDefinition; 3] {
    [
        tool_definition(&ReadFile(workspace.clone())),
        tool_definition(&WriteFile(workspace.clone())),
        tool_definition(&RunTests(workspace.clone())),
    ]
}

// P4 with explicit caching: the run crashes before its fourth model call,
// resumes from its checkpoint, and every turn on either side reads the cache.
#[tokio::test]
async fn config_agent_resumes_on_its_explicit_cache() {
    let workspace = Workspace::seeded();
    assert_eq!(
        workspace.validate(None).1,
        8,
        "the seeded repository fails eight tests"
    );
    let cache_name = Arc::new(Mutex::new(String::new()));
    let (repository, name) = (workspace.clone(), Arc::clone(&cache_name));
    let exchanges = with_gemini_long_horizon_cassette(
        "prompt_caching/config_agent_explicit",
        |gemini| async move {
            let caches = gemini.cached_contents();
            let cache = caches
                .create(
                    NewCachedContent::new(gemini::GEMINI_3_8_FLASH)
                        .system_instruction(CONFIG_PREAMBLE)
                        .tools(config_tools(&repository))
                        .expiry(CacheExpiry::ttl(Duration::from_secs(60 * 60))),
                )
                .await
                .expect("cache created");
            *name.lock().expect("unpoisoned") = cache.name().to_owned();

            let saved = Arc::new(Mutex::new(None));
            let crashed = config_agent(&gemini, &repository, Some(cache.clone()), &saved, Some(4))
                .prompt(CONFIG_TASK)
                .tool_concurrency(8)
                .await;
            assert!(crashed.is_err(), "the run stops before its fourth call");

            let json = saved
                .lock()
                .expect("unpoisoned")
                .clone()
                .expect("a checkpoint");
            let checkpoint: Checkpoint = serde_json::from_str(&json).expect("checkpoint loads");
            let cache = caches
                .ensure(&checkpoint.cache.expect("the checkpoint keeps its cache"))
                .await
                .expect("cache alive");
            let response = config_agent(&gemini, &repository, Some(cache.clone()), &saved, None)
                .resume(checkpoint.run)
                .tool_concurrency(8)
                .await
                .expect("resumed run succeeds");
            assert!(!response.output.is_empty());
            caches.delete(cache.name()).await.expect("cache deleted");
        },
    )
    .await;
    assert!(
        exchanges.unmodeled().is_empty(),
        "{:?}",
        exchanges.unmodeled()
    );
    assert_eq!(
        workspace.validate(None).1,
        0,
        "the agent fixed every failure"
    );
    assert!(exchanges.len() >= 8, "a long run, got {}", exchanges.len());
    assert_reads_cache(&exchanges, &cache_name.lock().expect("unpoisoned"));
}

// P4 with implicit caching: the same agent with no cache. Hits are reported.
#[tokio::test]
async fn config_agent_on_implicit_caching() {
    let workspace = Workspace::seeded();
    let repository = workspace.clone();
    let exchanges = with_gemini_long_horizon_cassette(
        "prompt_caching/config_agent_implicit",
        |gemini| async move {
            let saved = Arc::new(Mutex::new(None));
            config_agent(&gemini, &repository, None, &saved, None)
                .prompt(CONFIG_TASK)
                .tool_concurrency(8)
                .await
                .expect("run succeeds");
        },
    )
    .await;
    assert!(
        exchanges.unmodeled().is_empty(),
        "{:?}",
        exchanges.unmodeled()
    );
    assert_eq!(
        workspace.validate(None).1,
        0,
        "the agent fixed every failure"
    );
    for exchange in exchanges.turns() {
        assert!(exchange.request.cached_content.is_none());
        assert!(
            exchange.request.system_instruction.is_some() && !exchange.request.tools.is_empty()
        );
    }
    report_cache_hits("config_agent_implicit", &exchanges);
}

#[derive(Deserialize)]
struct AccountArgs {
    account: String,
}

struct Holdings;

impl PortableTool for Holdings {
    const NAME: &'static str = "portfolio_holdings";
    type Args = AccountArgs;
    type Output = Value;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Tickers and share counts held in an account.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "account": { "type": "string" } }, "required": ["account"] })
    }

    async fn call(&self, args: AccountArgs) -> Result<Value, Self::Error> {
        Ok(json!({ "account": args.account, "holdings": [
            { "ticker": "GOOGL", "shares": 12 },
            { "ticker": "MSFT", "shares": 5 }
        ]}))
    }
}

// P5: hosted code execution and Google Search beside a function tool. Their
// parts come back native, type through the mirror, and are re-sent verbatim.
#[tokio::test]
async fn portfolio_agent_keeps_hosted_parts() {
    let hosted = Arc::new(Mutex::new((0, 0, 0)));
    let counted = Arc::clone(&hosted);
    let exchanges = with_gemini_long_horizon_cassette("long_horizon/portfolio_agent", |gemini| async move {
        let model = gemini.completion(gemini::GEMINI_3_8_FLASH).settings(api::RequestSettings {
            tools: vec![
                api::HostedTool {
                    code_execution: Some(api::CodeExecution::default()),
                    ..Default::default()
                },
                api::HostedTool {
                    google_search: Some(api::GoogleSearch::default()),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let agent = AgentBuilder::new(model)
            .preamble("Use portfolio_holdings for positions, Google Search for prices, code for arithmetic.")
            .tool(Holdings)
            .default_max_turns(8)
            .build();

        let mut history = Vec::new();
        for question in [
            "What is account ACC-9 worth at today's closing prices?",
            "And if every price drops 7%?",
            "Which holding accounts for more of that drop, in dollars?",
        ] {
            agent.chat(question, &mut history).await.expect("turn succeeds");
        }
        let mut counted = counted.lock().expect("unpoisoned");
        for message in &history {
            let Message::Assistant { content, .. } = message else { continue };
            for item in content.iter() {
                if let AssistantContent::Native(native) = item {
                    let part = api::Part::try_from(native).expect("a Gemini part");
                    counted.0 += usize::from(part.executable_code.is_some());
                    counted.1 += usize::from(part.code_execution_result.is_some());
                    counted.2 += usize::from(part.tool_call.is_some());
                }
            }
        }
    })
    .await;
    assert!(
        exchanges.unmodeled().is_empty(),
        "{:?}",
        exchanges.unmodeled()
    );
    let (code, results, server_calls) = *hosted.lock().expect("unpoisoned");
    assert!(
        code > 0 && results > 0,
        "code ran: {code} code parts, {results} results"
    );
    assert!(
        server_calls > 0,
        "Google Search surfaced as a server-side tool call"
    );
    for (turn, exchange) in exchanges.turns().iter().enumerate() {
        assert!(
            sets_server_side_invocations(exchange),
            "turn {turn} mixes tools unannounced"
        );
    }
}

// P6 on scripted replies: a model id rig has never heard of, settings keys the
// mirror does not type, and a part kind it does not know. The keys reach the
// wire, the part round-trips, and typed or rig-owned keys are refused.
#[tokio::test]
async fn unknown_models_keys_and_parts_pass_through() {
    let unknown_part = json!({ "enterpriseSearchResult": { "hits": [{ "title": "Cafe Lumen" }] } });
    let first: api::GenerateContentResponse = serde_json::from_value(json!({
        "candidates": [{
            "content": { "role": "model", "parts": [unknown_part.clone(), { "text": "Cafe Lumen is quiet." }] },
            "finishReason": "STOP"
        }],
        "usageMetadata": { "promptTokenCount": 30, "candidatesTokenCount": 6, "totalTokenCount": 36 }
    }))
    .expect("a reply");
    let scripted = Scripted::new([first, rig::cassette::gemini::reply::text("It opens at 8.")]);
    let gemini = GeminiConfig::new("offline").connect(scripted.clone());

    let settings = api::RequestSettings {
        tools: vec![api::HostedTool {
            unmodeled: api::Unmodeled::new()
                .with("enterpriseSearch", json!({}))
                .expect("an untyped hosted tool"),
            ..Default::default()
        }],
        generation_config: api::GenerationSettings {
            thinking_config: Some(api::ThinkingConfig {
                thinking_level: Some(api::ThinkingLevel::Medium),
                ..Default::default()
            }),
            unmodeled: api::Unmodeled::new()
                .with("responseVerbosity", json!("LOW"))
                .expect("an untyped generation field"),
            ..Default::default()
        },
        ..Default::default()
    };
    let agent = AgentBuilder::new(gemini.completion("gemini-3.9-flash").settings(settings))
        .preamble("You recommend places to work from. Prefer quiet, well-reviewed cafes.")
        .build();
    let mut history = Vec::new();
    agent
        .chat("Find a quiet cafe near Alexanderplatz.", &mut history)
        .await
        .expect("first turn");
    agent
        .chat("When does it open?", &mut history)
        .await
        .expect("second turn");

    let requests = scripted.requests();
    let first = serde_json::to_value(&requests[0]).expect("a request");
    assert_eq!(first["tools"][0]["enterpriseSearch"], json!({}));
    assert_eq!(first["generationConfig"]["responseVerbosity"], json!("LOW"));
    assert_eq!(
        first["generationConfig"]["thinkingConfig"]["thinkingLevel"],
        json!("MEDIUM")
    );
    let replayed = serde_json::to_value(&requests[1].contents[1].parts[0]).expect("a part");
    assert_eq!(
        replayed, unknown_part,
        "the unknown part is re-sent as it arrived"
    );

    let refused =
        api::Unmodeled::<api::GenerationSettings>::new().with("thinkingConfig", json!({}));
    assert!(refused.is_err());
    let removed =
        api::Unmodeled::<api::GenerationSettings>::new().with("max_output_tokens", json!(64));
    assert!(removed.is_err());
}

// P6 live: the model is named by a string rig has no constant for, thinking
// is set in its typed home, and a hosted tool rides along. Google rejects
// keys it does not know, so the `Unmodeled` keys stay in the scripted test.
#[tokio::test]
async fn a_model_named_by_string_needs_no_release() {
    let exchanges =
        with_gemini_long_horizon_cassette("long_horizon/model_by_name", |gemini| async move {
            let settings = api::RequestSettings {
                tools: vec![api::HostedTool {
                    google_search: Some(api::GoogleSearch::default()),
                    ..Default::default()
                }],
                generation_config: api::GenerationSettings {
                    thinking_config: Some(api::ThinkingConfig {
                        thinking_level: Some(api::ThinkingLevel::Medium),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                ..Default::default()
            };
            let agent = AgentBuilder::new(gemini.completion("gemini-3.8-flash").settings(settings))
                .preamble("You recommend places to work from. Prefer quiet, well-reviewed cafes.")
                .build();
            let mut history = Vec::new();
            for question in [
                "Find a quiet cafe near Alexanderplatz.",
                "Which of those opens earliest?",
                "Summarize your pick in one line.",
            ] {
                agent
                    .chat(question, &mut history)
                    .await
                    .expect("turn succeeds");
            }
        })
        .await;
    assert!(
        exchanges.unmodeled().is_empty(),
        "{:?}",
        exchanges.unmodeled()
    );
    assert_eq!(exchanges.len(), 3);
    for (turn, exchange) in exchanges.turns().iter().enumerate() {
        assert!(
            exchange.path.contains("/models/gemini-3.8-flash:"),
            "turn {turn}"
        );
        let level = exchange
            .request
            .generation_config
            .as_ref()
            .and_then(|config| config.thinking_config.as_ref())
            .and_then(|thinking| thinking.thinking_level.clone());
        assert_eq!(level, Some(api::ThinkingLevel::Medium), "turn {turn}");
    }
}

struct SearchClauses;

impl PortableTool for SearchClauses {
    const NAME: &'static str = "search_clauses";
    type Args = SearchArgs;
    type Output = Value;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Full-text search over the clause index.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "query": { "type": "string" } }, "required": ["query"] })
    }

    async fn call(&self, args: SearchArgs) -> Result<Value, Self::Error> {
        Ok(json!([
            { "clause": "14.2", "text": format!("Termination for convenience: {}", args.query) },
            { "clause": "15.1", "text": "Notices in writing, by registered mail or email." }
        ]))
    }
}

// P8: every setting in its typed home, an explicit cache, per-part media
// resolution, and a PDF and an image in one prompt, over a review session.
#[tokio::test]
#[ignore = "not recorded: Google answers 503 to every flex-tier request that reads a cachedContent on gemini-3.8-flash; flex alone and cache alone both answer 200"]
async fn contract_review_uses_every_typed_setting() {
    let cache_name = Arc::new(Mutex::new(String::new()));
    let name = Arc::clone(&cache_name);
    let exchanges = with_gemini_long_horizon_cassette("long_horizon/contract_review", |gemini| async move {
        use base64::Engine;
        use base64::engine::general_purpose::STANDARD;

        let settings = api::RequestSettings {
            generation_config: api::GenerationSettings {
                thinking_config: Some(api::ThinkingConfig {
                    thinking_level: Some(api::ThinkingLevel::Low),
                    ..Default::default()
                }),
                media_resolution: Some(api::MediaResolution::High),
                ..Default::default()
            },
            service_tier: Some(api::ServiceTier::Flex),
            safety_settings: vec![
                api::SafetySetting {
                    category: Some(api::HarmCategory::DangerousContent),
                    threshold: Some(api::HarmBlockThreshold::BlockOnlyHigh),
                    ..Default::default()
                },
                api::SafetySetting {
                    category: Some(api::HarmCategory::Harassment),
                    threshold: Some(api::HarmBlockThreshold::BlockMediumAndAbove),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let caches = gemini.cached_contents();
        let prefix = caches
            .create(
                NewCachedContent::new(gemini::GEMINI_3_8_FLASH)
                    .system_instruction(HANDBOOK)
                    .tools([tool_definition(&SearchClauses)])
                    .expiry(CacheExpiry::ttl(Duration::from_secs(60 * 60))),
            )
            .await
            .expect("cache created");
        *name.lock().expect("unpoisoned") = prefix.name().to_owned();
        let agent = AgentBuilder::new(
            gemini
                .completion(gemini::GEMINI_3_8_FLASH)
                .settings(settings)
                .cached_content(prefix.clone()),
        )
        .preamble(HANDBOOK)
        .max_tokens(4096)
        .tool(SearchClauses)
        .default_max_turns(4)
        .build();

        let contract = UserContent::Document(Document {
            data: DocumentSourceKind::Base64(STANDARD.encode(CONTRACT_PDF)),
            media_type: Some(DocumentMediaType::PDF),
            detail: Some(MediaDetail::Medium),
            ..Default::default()
        });
        let signature_page = UserContent::Image(Image {
            data: DocumentSourceKind::Base64(STANDARD.encode(SIGNATURE_PAGE_PNG)),
            media_type: Some(ImageMediaType::PNG),
            detail: Some(MediaDetail::Low),
            ..Default::default()
        });
        let first = Message::User {
            content: NonEmpty::with_rest(
                UserContent::Text(Text::new(
                    "Which clause covers early termination, and is the contract signed?",
                )),
                [contract, signature_page],
            ),
        };
        let mut history = Vec::new();
        agent.chat(first, &mut history).await.expect("first turn");
        for question in [
            "What notice and fee apply if the Customer terminates for convenience ten months after the Effective Date?",
            "Does that fee apply if the Customer terminates for cause instead?",
            "Can the Customer give the notice by email?",
        ] {
            agent.chat(question, &mut history).await.expect("turn succeeds");
        }
        caches.delete(prefix.name()).await.expect("cache deleted");
    })
    .await;
    assert!(
        exchanges.unmodeled().is_empty(),
        "{:?}",
        exchanges.unmodeled()
    );
    assert_reads_cache(&exchanges, &cache_name.lock().expect("unpoisoned"));

    for (turn, exchange) in exchanges.turns().iter().enumerate() {
        let request = &exchange.request;
        assert_eq!(
            request.service_tier,
            Some(api::ServiceTier::Flex),
            "turn {turn}"
        );
        assert_eq!(request.safety_settings.len(), 2, "turn {turn}");
        let config = request
            .generation_config
            .as_ref()
            .expect("a generation config");
        assert_eq!(config.max_output_tokens, Some(4096), "turn {turn}");
        assert_eq!(
            config.media_resolution,
            Some(api::MediaResolution::High),
            "turn {turn}"
        );
        let level = config
            .thinking_config
            .as_ref()
            .and_then(|thinking| thinking.thinking_level.clone());
        assert_eq!(level, Some(api::ThinkingLevel::Low), "turn {turn}");
    }
    let first = exchanges.turns().first().expect("a first turn");
    let resolution = |mime: &str| {
        last_turn_parts(first)
            .iter()
            .find(|part| {
                part.inline_data
                    .as_ref()
                    .is_some_and(|blob| blob.mime_type.as_deref() == Some(mime))
            })
            .and_then(|part| part.media_resolution.as_ref())
            .and_then(|resolution| resolution.level.clone())
    };
    assert_eq!(
        resolution("application/pdf"),
        Some(api::MediaResolutionLevel::Medium)
    );
    assert_eq!(
        resolution("image/png"),
        Some(api::MediaResolutionLevel::Low)
    );
}
