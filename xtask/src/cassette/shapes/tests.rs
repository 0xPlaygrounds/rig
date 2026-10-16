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

#[test]
fn an_ndjson_reply_is_the_set_of_its_record_skeletons() {
    let body = "{\"message\":{\"content\":\"a\"},\"done\":false}\n\
                {\"message\":{\"content\":\"b\"},\"done\":false}\n\
                {\"done\":true,\"done_reason\":\"stop\"}\n";
    let reply = exchange("application/x-ndjson", body);
    assert_eq!(
        reply_shape(&reply),
        "200 ndjson[{done:bool,done_reason:\"stop\"}|{done:bool,message:{content:str}}]"
    );
    assert_eq!(reply_documents(&reply).len(), 3);
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
    let fixtures = corpus(&dir).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let [fixture] = fixtures.as_slice() else {
        panic!("one fixture, not {fixtures:?}");
    };
    assert_eq!(fixture.provider, "groq");
    assert_eq!(fixture.relative, "groq/scenario/a.yaml");
    let [first, second] = fixture.interactions.as_slice() else {
        panic!("two interactions, not {:?}", fixture.interactions);
    };
    assert_eq!(first.when.encoder(), "POST /openai/v1/chat/completions");
    assert_eq!(texts(&request_facts(&first.when)), ["$ {model:str}"]);
    assert_eq!(request_facts(&first.when), request_facts(&second.when));
    assert_ne!(reply_shape(&first.then), reply_shape(&second.then));
}
