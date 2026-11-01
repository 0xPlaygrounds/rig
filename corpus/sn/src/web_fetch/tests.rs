use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex, PoisonError};

use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness_test_support::{
    AnswerAsks, Answers, Ended, ToolOutputs, app_in, calls, first_agent, log_ended,
    log_tool_outputs, reply, run_until, scratch_home, scripted_models,
};
use serde_json::json;

use super::*;

/// The page `/doc` serves.
const DOC: &str = r#"<!doctype html><html><head><title>Doc</title>
<script>alert(1)</script><style>body { color: red }</style></head>
<body><nav><a href="/">Home</a> | <a href="/blog">Site menu</a></nav>
<main><h1>Serde</h1><p>A <b>framework</b> for <a href="/ser">serializing</a>.</p>
<h2>Install</h2>
<pre><code class="language-sh">cargo add serde --features derive
</code></pre>
<ul><li>fast</li><li>generic</li></ul></main>
<footer>Copyright footer text</footer></body></html>"#;

/// A local HTTP server: `/old` moves to `/doc`, and the requests it saw by
/// path.
fn serve(long: String) -> (String, Arc<Mutex<HashMap<String, usize>>>) {
    let Ok(listener) = TcpListener::bind("127.0.0.1:0") else {
        return (String::new(), Arc::default());
    };
    let base = listener
        .local_addr()
        .map(|address| format!("http://{address}"))
        .unwrap_or_default();
    let seen: Arc<Mutex<HashMap<String, usize>>> = Arc::default();
    let counted = seen.clone();
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut request = String::new();
            let mut reader = BufReader::new(&stream);
            while reader.read_line(&mut request).is_ok_and(|read| read > 2) {}
            let path = request.split_whitespace().nth(1).unwrap_or("/").to_owned();
            *counted
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(path.clone())
                .or_default() += 1;
            let png = b"\x89PNG\r\n\x1a\n0000000000".to_vec();
            let (status, extra, kind, body) = match path.as_str() {
                "/old" => (
                    "301 Moved Permanently",
                    "Location: /doc\r\n",
                    "text/plain",
                    Vec::new(),
                ),
                "/doc" => (
                    "200 OK",
                    "",
                    "text/html; charset=utf-8",
                    DOC.as_bytes().to_vec(),
                ),
                "/data.json" => (
                    "200 OK",
                    "",
                    "application/json",
                    br#"{"name":"serde","versions":[1,2]}"#.to_vec(),
                ),
                "/logo.png" => ("200 OK", "", "image/png", png),
                "/long" => ("200 OK", "", "text/plain", long.as_bytes().to_vec()),
                _ => ("404 Not Found", "", "text/plain", b"missing".to_vec()),
            };
            let head = format!(
                "HTTP/1.1 {status}\r\n{extra}Content-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes()).ok();
            stream.write_all(&body).ok();
        }
    });
    (base, seen)
}

/// The text a result shows between its header and where to go on.
fn body(output: &str) -> &str {
    let text = output.split_once("\n\n").map_or("", |(_, text)| text);
    match text.rsplit_once("\n\n") {
        Some((text, tail)) if tail.contains("call again with offset=") => text,
        _ => text,
    }
}

/// An app with the plugin on the scripted `model`, under a home whose
/// `fetch.toml` is `config`.
fn app_with(
    name: &str,
    config: &str,
    model: &MockCompletionModel,
) -> (App, std::sync::mpsc::Receiver<()>) {
    let home = scratch_home(name);
    std::fs::create_dir_all(home.root()).ok();
    std::fs::write(home.root().join(CONFIG), config).ok();
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(scripted_models([("test/model", model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, WebFetchPlugin))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    (app, wakes)
}

fn run(app: &mut App, wakes: &std::sync::mpsc::Receiver<()>) {
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    app.world_mut()
        .trigger(Deliver::new(agent, "Read the docs.", DeliveryMode::Steer));
    run_until(app, wakes, |world| !world.resource::<Ended>().0.is_empty());
}

#[test]
fn fetches_pages_as_clean_text_within_the_policy() {
    let long: String = (0..60_000u32)
        .map(|at| {
            if at % 80 == 79 {
                '\n'
            } else {
                char::from(b'a' + (at % 26) as u8)
            }
        })
        .collect();
    let (base, seen) = serve(long.clone());
    let fetch = |id: &str, args: serde_json::Value| calls(id, WEB_FETCH, args);
    let model = MockCompletionModel::from_stream_turns([
        fetch("old", json!({ "url": format!("{base}/old") })),
        fetch("doc", json!({ "url": format!("{base}/doc") })),
        fetch("json", json!({ "url": format!("{base}/data.json") })),
        fetch("png", json!({ "url": format!("{base}/logo.png") })),
        fetch("long1", json!({ "url": format!("{base}/long") })),
        fetch(
            "long2",
            json!({ "url": format!("{base}/long"), "offset": 20_000, "max_chars": 50_000 }),
        ),
        fetch("unlisted", json!({ "url": "http://unlisted.invalid/" })),
        fetch("private", json!({ "url": "http://10.0.0.1/" })),
        reply("done"),
    ]);
    let (mut app, wakes) = app_with("sn-web-fetch", r#"allow = ["127.0.0.1"]"#, &model);
    run(&mut app, &wakes);

    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = |id: &str| outputs.get(id).cloned().unwrap_or_default();
    let old = output("old");
    assert!(old.contains(&format!("URL: {base}/doc")), "{old}");
    assert!(
        old.contains("Status: 200") && old.contains("text/html"),
        "{old}"
    );
    assert!(old.contains("## Install"), "{old}");
    assert!(
        old.contains("```sh\ncargo add serde --features derive\n```"),
        "{old}"
    );
    assert!(old.contains(&format!("[serializing]({base}/ser)")), "{old}");
    assert!(
        !old.contains("alert(1)") && !old.contains("Site menu"),
        "{old}"
    );
    assert!(
        !old.contains("footer text") && !old.contains("color: red"),
        "{old}"
    );

    let doc = output("doc");
    assert!(
        doc.contains("(cached)") && doc.contains("## Install"),
        "{doc}"
    );
    let seen = seen.lock().unwrap_or_else(PoisonError::into_inner).clone();
    assert_eq!(seen.get("/doc"), Some(&1), "{seen:?}");

    let data = output("json");
    assert!(
        data.contains("{\n  \"name\": \"serde\",\n  \"versions\": [\n    1,"),
        "{data}"
    );
    let png = output("png");
    assert!(
        png.contains("image/png") && png.contains("18 bytes"),
        "{png}"
    );

    let first = output("long1");
    assert!(
        first.contains("40000 more characters; call again with offset=20000"),
        "{first}"
    );
    let second = output("long2");
    assert!(!second.contains("call again"), "{second}");
    assert_eq!(format!("{}{}", body(&first), body(&second)), long);

    let unlisted = output("unlisted");
    assert!(
        unlisted.contains("neither list") && unlisted.contains("nobody can allow"),
        "{unlisted}"
    );
    let private = output("private");
    assert!(
        private.contains("10.0.0.1 is a loopback, link-local or private address"),
        "{private}"
    );
}

#[test]
fn asks_once_per_domain() {
    let fetch = |id: &str, url: &str| calls(id, WEB_FETCH, json!({ "url": url }));
    let model = MockCompletionModel::from_stream_turns([
        fetch("first", "http://asked.invalid/a"),
        fetch("again", "http://asked.invalid/b"),
        reply("done"),
    ]);
    let (mut app, wakes) = app_with("sn-web-fetch-asks", "", &model);
    app.insert_resource(Answers::new([Answer::choice(DENY)]))
        .add_plugins(AnswerAsks);
    run(&mut app, &wakes);

    let asked = &app.world().resource::<Answers>().asked;
    assert_eq!(asked, &[title("asked.invalid")]);
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let first = outputs.get("first").cloned().unwrap_or_default();
    assert!(
        first.contains("did not allow fetching from asked.invalid"),
        "{first}"
    );
    let again = outputs.get("again").cloned().unwrap_or_default();
    assert!(
        again.contains("denied asked.invalid for this session"),
        "{again}"
    );
    let denied = &app.world().resource::<FetchDomains>().denied;
    assert_eq!(denied, &["asked.invalid".to_owned()]);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(WebFetchPlugin);
}
