use super::*;

fn exchange(content_type: &str, body: &str) -> Exchange {
    Exchange {
        path: String::new(),
        method: String::new(),
        status: 200,
        header: vec![NameValue {
            name: "content-type".to_owned(),
            value: content_type.to_owned(),
        }],
        body: Some(body.to_owned()),
        body_encoding: None,
    }
}

#[test]
fn path_templates_erase_models_and_ids() {
    for (path, template) in [
        (
            "/v1beta/models/gemini-2.5-flash:streamGenerateContent",
            "/v1beta/models/{model}:streamGenerateContent",
        ),
        (
            "/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse-stream",
            "/model/{model}/converse-stream",
        ),
        (
            "/v1/responses/resp_0ff9b3a3728736b2006ab2ea67575087d0",
            "/v1/responses/{id}",
        ),
        (
            "/v1beta/cachedContents/pkucxaxnfhgp583g861gbmm8mn7kerhsx0fbdkjq",
            "/v1beta/cachedContents/{id}",
        ),
        ("/v1/chat/completions", "/v1/chat/completions"),
        ("/v1beta/models", "/v1beta/models"),
    ] {
        assert_eq!(path_template(path), template, "{path}");
    }
}

#[test]
fn a_skeleton_keeps_types_and_keys_and_collapses_arrays() {
    let value = serde_json::json!({
        "model": "gpt",
        "stream": true,
        "max_tokens": 5,
        "messages": [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": [{"type": "text", "text": "c"}]}
        ],
        "tools": [],
        "stop": null
    });
    assert_eq!(
        skeleton(&value, &[]),
        "{max_tokens:num,messages:[{content:[{text:str,type:str}],role:str}|{content:str,role:str}],\
         model:str,stop:null,stream:bool,tools:[]}"
    );
}

#[test]
fn a_reply_skeleton_keeps_discriminator_values() {
    let value = serde_json::json!({"type": "message", "stop_reason": "end_turn", "text": "hi"});
    assert_eq!(
        skeleton(&value, DISCRIMINATORS),
        "{stop_reason:\"end_turn\",text:str,type:\"message\"}"
    );
}

fn root(object: &str) -> Facts {
    Facts::from([Fact {
        path: "$".to_owned(),
        object: object.to_owned(),
    }])
}

fn texts(facts: &Facts) -> Vec<String> {
    facts.iter().map(ToString::to_string).collect()
}

#[test]
fn request_bodies_are_classified_before_their_json_is_read() {
    let mut empty = exchange("application/json", "");
    assert_eq!(request_facts(&empty), root("empty"));
    empty.body = None;
    assert_eq!(request_facts(&empty), root("empty"));
    let mut binary = exchange("application/octet-stream", "AAAA");
    binary.body_encoding = Some("base64".to_owned());
    assert_eq!(request_facts(&binary), root("binary"));
    let multipart = exchange(
        "multipart/form-data; boundary=x",
        "--x\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n\
         --x\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a\"\r\n\r\nb\r\n--x--",
    );
    assert_eq!(request_facts(&multipart), root("multipart[file,model]"));
    assert_eq!(
        request_facts(&exchange("text/plain", "hello")),
        root("text")
    );
    assert_eq!(
        request_facts(&exchange("application/json", "{\"a\":[1,2]}")),
        root("{a:[num]}")
    );
    assert_eq!(
        texts(&request_facts(&exchange(
            "application/json",
            "[[1],{\"b\":2}]"
        ))),
        ["$ [[num]|{}]", "$[] {b:num}"]
    );
}

#[test]
fn a_fact_is_one_objects_keys_and_kinds_at_its_path() {
    let value = serde_json::json!({
        "model": "gpt",
        "stream": true,
        "max_tokens": 5,
        "messages": [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": [{"type": "text", "text": "c"}]},
            {"role": "user", "content": [{"type": "text", "text": "d"}]}
        ],
        "tools": [],
        "stop": null
    });
    assert_eq!(
        texts(&facts(&value, DISCRIMINATORS)),
        [
            "$ {max_tokens:num,messages:[{}],model:str,stop:null,stream:bool,tools:[]}",
            "$.messages[] {content:[{}],role:\"user\"}",
            "$.messages[] {content:str,role:\"system\"}",
            "$.messages[] {content:str,role:\"user\"}",
            "$.messages[].content[] {text:str,type:\"text\"}",
        ]
    );
}

#[test]
fn schema_property_names_and_tool_values_are_not_facts() {
    let value = serde_json::json!({
        "tools": [{
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                    "properties": {"type": "object", "properties": {"x": {"type": "number"}}}
                },
                "$defs": {"Unit": {"enum": ["c", "f"]}}
            }
        }],
        "messages": [{"content": [{"type": "tool_use", "input": {"city": "Paris"}}]}],
        "input": [{"type": "function_result", "result": {"sum": 3}}]
    });
    assert_eq!(
        texts(&facts(&value, DISCRIMINATORS)),
        [
            "$ {input:[{}],messages:[{}],tools:[{}]}",
            "$.input[] {result:{},type:\"function_result\"}",
            "$.messages[] {content:[{}]}",
            "$.messages[].content[] {input:{},type:\"tool_use\"}",
            "$.tools[] {input_schema:{}}",
            "$.tools[].input_schema {$defs:{},properties:{},type:\"object\"}",
            "$.tools[].input_schema.$defs.* {enum:[str]}",
            "$.tools[].input_schema.properties.* {properties:{},type:\"object\"}",
            "$.tools[].input_schema.properties.* {type:\"string\"}",
            "$.tools[].input_schema.properties.*.properties.* {type:\"number\"}",
        ]
    );
}

#[test]
fn a_stream_reply_is_the_set_of_its_event_skeletons() {
    let body = "event: delta\ndata: {\"type\":\"text\",\"text\":\"a\"}\n\n\
                event: delta\ndata: {\"type\":\"text\",\"text\":\"b\"}\n\n\
                data: [DONE]\n\n";
    assert_eq!(
        reply_shape(&exchange("text/event-stream; charset=utf-8", body)),
        "200 sse[=[DONE]|delta={text:str,type:\"text\"}]"
    );
    let mut failed = exchange("application/json", "{\"error\":{\"type\":\"overloaded\"}}");
    failed.status = 529;
    assert_eq!(
        reply_shape(&failed),
        "529 json{error:{type:\"overloaded\"}}"
    );
}

fn frame(event: &str, payload: &str) -> Vec<u8> {
    let mut headers = Vec::new();
    let name = ":event-type";
    headers.push(u8::try_from(name.len()).unwrap());
    headers.extend_from_slice(name.as_bytes());
    headers.push(7);
    headers.extend_from_slice(&u16::try_from(event.len()).unwrap().to_be_bytes());
    headers.extend_from_slice(event.as_bytes());
    let total = 12 + headers.len() + payload.len() + 4;
    let mut out = Vec::new();
    out.extend_from_slice(&u32::try_from(total).unwrap().to_be_bytes());
    out.extend_from_slice(&u32::try_from(headers.len()).unwrap().to_be_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&headers);
    out.extend_from_slice(payload.as_bytes());
    out.extend_from_slice(&[0; 4]);
    out
}

#[test]
fn an_aws_event_stream_is_read_frame_by_frame() {
    let mut bytes = frame("messageStart", "{\"role\":\"assistant\"}");
    bytes.extend(frame("contentBlockDelta", "{\"delta\":{\"text\":\"hi\"}}"));
    assert_eq!(
        event_stream(&bytes).unwrap(),
        [
            "messageStart={role:\"assistant\"}",
            "contentBlockDelta={delta:{text:str}}"
        ]
    );
    assert_eq!(event_stream(&bytes[..10]), None);
}

#[test]
fn base64_decodes_with_and_without_padding() {
    assert_eq!(decode_base64("aGVsbG8=").unwrap(), b"hello");
    assert_eq!(decode_base64("aGVsbG8").unwrap(), b"hello");
    assert_eq!(decode_base64("aGVs\nbG8=").unwrap(), b"hello");
    assert_eq!(decode_base64("a*b"), None);
}

#[test]
fn the_hash_is_fnv1a() {
    assert_eq!(hash(""), "cbf29ce484222325");
    assert_eq!(hash("a"), "af63dc4c8601ec8c");
}

fn key(provider: &str, kind: Kind, hash: &str) -> ShapeKey {
    ShapeKey {
        provider: provider.to_owned(),
        encoder: "POST /v1/messages".to_owned(),
        kind,
        shape: hash.to_owned(),
    }
}

#[test]
fn the_baseline_round_trips_and_reports_lost_shapes() {
    let mut shapes = Shapes::new();
    shapes.insert(
        key("anthropic", Kind::Request, "00"),
        Recorded {
            recordings: 2,
            example: "anthropic/a.yaml#0".to_owned(),
        },
    );
    shapes.insert(
        key("anthropic", Kind::Reply, "01"),
        Recorded {
            recordings: 1,
            example: "anthropic/a.yaml#1".to_owned(),
        },
    );
    let text = render(&shapes);
    assert_eq!(parse(&text).unwrap(), shapes);
    let mut current = shapes.clone();
    current.remove(&key("anthropic", Kind::Reply, "01"));
    let lost = lost(&shapes, &current);
    assert_eq!(lost.len(), 1);
    assert!(lost[0].contains("reply") && lost[0].contains("anthropic/a.yaml#1"));
    assert!(super::lost(&shapes, &shapes).is_empty());
    assert!(parse(&format!("{HEADER}\nbroken")).is_err());
}

#[test]
fn the_corpus_is_read_document_by_document() {
    let dir = std::env::temp_dir().join(format!("xtask-shapes-{}", std::process::id()));
    let provider = dir.join("groq/scenario");
    std::fs::create_dir_all(&provider).unwrap();
    std::fs::write(
        provider.join("a.yaml"),
        "when:\n  path: /openai/v1/chat/completions\n  method: POST\n  query_param: []\n  \
         header: []\n  body: '{\"model\":\"m\"}'\nthen:\n  status: 200\n  header:\n  \
         - name: content-type\n    value: application/json\n  body: '{\"object\":\"chat.completion\"}'\n\
         ---\nwhen:\n  path: /openai/v1/chat/completions\n  method: POST\n  query_param: []\n  \
         header: []\n  body: '{\"model\":\"n\"}'\nthen:\n  status: 200\n  header: []\n  body: ''\n",
    )
    .unwrap();
    let shapes = collect(&dir).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(summary(&shapes).get("groq"), Some(&(1, 2)));
    let request = shapes
        .iter()
        .find(|(key, _)| key.kind == Kind::Request)
        .unwrap();
    assert_eq!(request.0.encoder, "POST /openai/v1/chat/completions");
    assert_eq!(request.0.shape, "$ {model:str}");
    assert_eq!(request.1.recordings, 2);
    assert_eq!(request.1.example, "groq/scenario/a.yaml#0");
}

#[test]
fn a_banked_reply_shape_counts_only_where_no_cassette_records_it() {
    let key = |kind: Kind, shape: &str| ShapeKey {
        provider: "openai".to_owned(),
        encoder: "POST /v1/responses".to_owned(),
        kind,
        shape: shape.to_owned(),
    };
    let recorded = Recorded {
        recordings: 3,
        example: "openai/a.yaml#0".to_owned(),
    };
    let mut shapes = Shapes::from([(key(Kind::Reply, "00aa"), recorded.clone())]);
    add_banked(
        &mut shapes,
        BTreeMap::from([
            (key(Kind::Reply, "00aa"), "openai/b.yaml#1".to_owned()),
            (key(Kind::Reply, "00bb"), "openai/c.yaml#2".to_owned()),
            (key(Kind::Request, "$ {}"), "openai/d.yaml#0".to_owned()),
        ]),
    );
    assert_eq!(shapes.get(&key(Kind::Reply, "00aa")), Some(&recorded));
    assert_eq!(
        shapes.get(&key(Kind::Reply, "00bb")),
        Some(&Recorded {
            recordings: 1,
            example: "bank:openai/c.yaml#2".to_owned(),
        })
    );
    assert!(!shapes.contains_key(&key(Kind::Request, "$ {}")));
    let baseline = shapes.clone();
    let mut current = Shapes::new();
    assert_eq!(lost(&baseline, &current).len(), 2);
    add_banked(
        &mut current,
        BTreeMap::from([
            (key(Kind::Reply, "00aa"), "openai/b.yaml#1".to_owned()),
            (key(Kind::Reply, "00bb"), "openai/c.yaml#2".to_owned()),
        ]),
    );
    assert!(lost(&baseline, &current).is_empty());
}
