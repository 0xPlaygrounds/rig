//! `gem` against an in-process fake Gemini gRPC service on loopback,
//! scripted per test.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rig_core::providers::gemini::GeminiConfig;
use rig_gemini_grpc::proto::{
    self, Candidate, Content, ContentEmbedding, EmbedContentRequest, EmbedContentResponse,
    GenerateContentRequest, GenerateContentResponse, Part, part::Data,
};
use rig_memory::MemoryStore;
use tonic::codegen::{BoxFuture, Service, http};
use tonic::transport::server::TcpIncoming;
use tonic::transport::{Endpoint, Server};

use super::*;

/// The value of an `Ok`, else a failed assertion that ends the test.
macro_rules! ok {
    ($result:expr) => {{
        let result = $result;
        assert!(result.is_ok(), "{:?}", result.as_ref().err());
        let Ok(value) = result else { return };
        value
    }};
}

const SERVICE: &str = "google.ai.generativelanguage.v1beta.GenerativeService";
const STOP: i32 = proto::candidate::FinishReason::Stop as i32;

/// One scripted reply: its messages, or a status that ends it.
type Reply = Vec<Result<GenerateContentResponse, tonic::Status>>;

/// The fake: answers each `GenerateContent`/`StreamGenerateContent` with the
/// next scripted reply and records the request; answers `EmbedContent` with
/// a vector whose first value is the number in the text.
#[derive(Clone, Default)]
struct Fake {
    script: Arc<Mutex<VecDeque<Reply>>>,
    seen: Arc<Mutex<Vec<GenerateContentRequest>>>,
    embeds: Arc<Mutex<Vec<EmbedContentRequest>>>,
}

impl Fake {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        let fake = Self::default();
        if let Ok(mut script) = fake.script.lock() {
            script.extend(replies);
        }
        fake
    }

    fn next(&self, request: GenerateContentRequest) -> Reply {
        if let Ok(mut seen) = self.seen.lock() {
            seen.push(request);
        }
        self.script
            .lock()
            .ok()
            .and_then(|mut script| script.pop_front())
            .unwrap_or_else(|| vec![Err(tonic::Status::internal("script ran out"))])
    }

    fn seen(&self) -> Vec<GenerateContentRequest> {
        self.seen
            .lock()
            .map(|seen| seen.clone())
            .unwrap_or_default()
    }

    /// Serve on loopback and connect `gem` to it.
    async fn serve(
        &self,
    ) -> Result<(Gem, tokio::task::JoinHandle<()>), Box<dyn std::error::Error>> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let service = self.clone();
        let server = tokio::spawn(async move {
            let _ = Server::builder()
                .add_service(service)
                .serve_with_incoming(TcpIncoming::from(listener))
                .await;
        });
        let channel = Endpoint::from_shared(format!("http://{address}"))?
            .connect()
            .await?;
        Ok((
            Gem::with_channel(&GeminiConfig::new("dummy"), channel)?,
            server,
        ))
    }
}

impl tonic::server::NamedService for Fake {
    const NAME: &'static str = SERVICE;
}

/// `GenerateContent`.
struct Unary(Fake);

impl tonic::server::UnaryService<GenerateContentRequest> for Unary {
    type Response = GenerateContentResponse;
    type Future = std::future::Ready<Result<tonic::Response<Self::Response>, tonic::Status>>;

    fn call(&mut self, request: tonic::Request<GenerateContentRequest>) -> Self::Future {
        let reply = self.0.next(request.into_inner()).into_iter().next();
        std::future::ready(match reply {
            Some(reply) => reply.map(tonic::Response::new),
            None => Err(tonic::Status::internal("empty reply")),
        })
    }
}

/// `StreamGenerateContent`.
struct Streaming(Fake);

impl tonic::server::ServerStreamingService<GenerateContentRequest> for Streaming {
    type Response = GenerateContentResponse;
    type ResponseStream =
        futures::stream::Iter<std::vec::IntoIter<Result<Self::Response, tonic::Status>>>;
    type Future = std::future::Ready<Result<tonic::Response<Self::ResponseStream>, tonic::Status>>;

    fn call(&mut self, request: tonic::Request<GenerateContentRequest>) -> Self::Future {
        let reply = self.0.next(request.into_inner());
        std::future::ready(Ok(tonic::Response::new(futures::stream::iter(reply))))
    }
}

/// `EmbedContent`.
struct Embed(Fake);

impl tonic::server::UnaryService<EmbedContentRequest> for Embed {
    type Response = EmbedContentResponse;
    type Future = std::future::Ready<Result<tonic::Response<Self::Response>, tonic::Status>>;

    fn call(&mut self, request: tonic::Request<EmbedContentRequest>) -> Self::Future {
        let request = request.into_inner();
        let text = request
            .content
            .iter()
            .flat_map(|content| &content.parts)
            .find_map(|part| match &part.data {
                Some(Data::Text(text)) => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let number: f32 = text.trim_start_matches("note ").parse().unwrap_or(-1.0);
        let dims = request.output_dimensionality.unwrap_or(3072).max(1) as usize;
        let mut values = vec![0.5; dims];
        if let Some(first) = values.first_mut() {
            *first = number;
        }
        if let Ok(mut embeds) = self.0.embeds.lock() {
            embeds.push(request);
        }
        std::future::ready(Ok(tonic::Response::new(EmbedContentResponse {
            embedding: Some(ContentEmbedding { values }),
        })))
    }
}

impl<B> Service<http::Request<B>> for Fake
where
    B: tonic::codegen::Body + Send + 'static,
    B::Error: Into<tonic::codegen::StdError> + Send,
{
    type Response = http::Response<tonic::body::Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<B>) -> Self::Future {
        let fake = self.clone();
        Box::pin(async move {
            let method = request
                .uri()
                .path()
                .rsplit('/')
                .next()
                .unwrap_or("")
                .to_owned();
            Ok(match method.as_str() {
                "GenerateContent" => grpc().unary(Unary(fake), request).await,
                "StreamGenerateContent" => grpc().server_streaming(Streaming(fake), request).await,
                "EmbedContent" => grpc().unary(Embed(fake), request).await,
                _ => tonic::Status::unimplemented(method).into_http(),
            })
        })
    }
}

/// A server that encodes replies of any size, as Gemini's does.
fn grpc<T, U>() -> tonic::server::Grpc<tonic_prost::ProstCodec<T, U>>
where
    T: prost::Message + Send + 'static,
    U: prost::Message + Default + Send + 'static,
{
    tonic::server::Grpc::new(tonic_prost::ProstCodec::default())
        .max_encoding_message_size(usize::MAX)
}

fn part(data: Data) -> Part {
    Part {
        data: Some(data),
        ..Part::default()
    }
}

fn thought(text: &str) -> Part {
    Part {
        thought: true,
        ..part(Data::Text(text.to_owned()))
    }
}

fn message(parts: Vec<Part>, finish: i32) -> GenerateContentResponse {
    GenerateContentResponse {
        candidates: vec![Candidate {
            content: Some(Content {
                parts,
                role: "model".to_owned(),
            }),
            finish_reason: finish,
            ..Candidate::default()
        }],
        ..GenerateContentResponse::default()
    }
}

fn usage(prompt: i32, candidates: i32, thoughts: i32, cached: i32) -> proto::UsageMetadata {
    proto::UsageMetadata {
        prompt_token_count: prompt,
        candidates_token_count: candidates,
        thoughts_token_count: thoughts,
        cached_content_token_count: cached,
        total_token_count: prompt + candidates + thoughts,
        ..proto::UsageMetadata::default()
    }
}

fn settings(thinking: Thinking) -> ChatSettings {
    ChatSettings {
        model: "gemini-2.5-flash".to_owned(),
        thinking,
        conversation: "chat".to_owned(),
    }
}

/// The thinking config the request carried.
fn thinking_of(request: &GenerateContentRequest) -> Option<proto::ThinkingConfig> {
    request
        .generation_config
        .as_ref()
        .and_then(|config| config.thinking_config)
}

/// Every thought signature in the request's history.
fn signatures(request: &GenerateContentRequest) -> Vec<Vec<u8>> {
    request
        .contents
        .iter()
        .flat_map(|content| &content.parts)
        .filter(|part| !part.thought_signature.is_empty())
        .map(|part| part.thought_signature.clone())
        .collect()
}

/// R1, R2, R5: thinking at budget 1024 reaches the wire, thoughts stream
/// apart from the answer, and the signature of a function call survives the
/// tool round and a reload of the conversation from its serialized log.
#[tokio::test]
async fn thinking_tool_calls_and_signatures_survive_a_reload() {
    let signature = b"sig-\x00\x01\xfe-opaque".to_vec();
    let call = Part {
        thought_signature: signature.clone(),
        ..part(Data::FunctionCall(proto::FunctionCall {
            name: "search_notes".to_owned(),
            args: None,
            id: String::new(),
        }))
    };
    let mut first = message(vec![thought("Looking in the notes."), call], STOP);
    first.usage_metadata = Some(usage(100, 10, 40, 0));
    let mut answer = message(
        vec![
            thought("Found nothing."),
            part(Data::Text("No notes yet.".to_owned())),
        ],
        STOP,
    );
    answer.usage_metadata = Some(usage(150, 5, 20, 64));
    let fake = Fake::new([
        vec![Ok(first)],
        vec![Ok(answer)],
        vec![Ok(message(
            vec![part(Data::Text("Still none.".to_owned()))],
            STOP,
        ))],
    ]);
    let (gem, server) = ok!(fake.serve().await);
    let store = MemoryStore::default();

    let mut events = Vec::new();
    let reply = ok!(gem
        .chat(settings(Thinking::Budget(1024)), store.clone(), None)
        .send("What do my notes say about fog?", |event| events
            .push(event))
        .await);
    assert_eq!(reply.answer, "No notes yet.");
    assert_eq!(reply.thoughts, "Looking in the notes.Found nothing.");
    assert!(events.contains(&ChatEvent::Thought("Found nothing.".to_owned())));
    assert!(events.contains(&ChatEvent::ToolCall("search_notes".to_owned())));
    assert!(!events.contains(&ChatEvent::Answer("Found nothing.".to_owned())));
    let expected_usage = [
        CallUsage {
            input: Some(100),
            output: Some(50),
            thinking: Some(40),
            cached: None,
        },
        CallUsage {
            input: Some(150),
            output: Some(25),
            thinking: Some(20),
            cached: Some(64),
        },
    ];
    assert_eq!(reply.usage, expected_usage);

    // A new chat over the same store reads the conversation back from its
    // serialized log, as a restarted program does.
    ok!(gem
        .chat(settings(Thinking::Budget(1024)), store, None)
        .send("And now?", |_| {})
        .await);
    server.abort();

    let seen = fake.seen();
    assert_eq!(seen.len(), 3);
    for request in &seen {
        let thinking = thinking_of(request);
        assert_eq!(
            thinking.and_then(|config| config.thinking_budget),
            Some(1024)
        );
        assert_eq!(
            thinking.and_then(|config| config.include_thoughts),
            Some(true)
        );
    }
    for request in seen.iter().skip(1) {
        assert_eq!(signatures(request), vec![signature.clone()]);
    }
}

/// R3: an 8 MiB image and a 3 KiB caption in one reply arrive intact and
/// in order, and save as a file with the same bytes.
#[tokio::test]
async fn a_large_image_reply_arrives_intact() {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend((0..8 * 1024 * 1024).map(|i: u32| (i % 251) as u8));
    let caption = "fog ".repeat(768);
    let reply = message(
        vec![
            part(Data::Text(caption.clone())),
            part(Data::InlineData(proto::Blob {
                mime_type: "image/png".to_owned(),
                data: png.clone(),
            })),
        ],
        STOP,
    );
    let fake = Fake::new([vec![Ok(reply)]]);
    let (gem, server) = ok!(fake.serve().await);
    let drawing = ok!(gem
        .draw("gemini-2.5-flash-image", "a lighthouse in fog", None)
        .await);
    server.abort();

    assert_eq!(
        drawing.parts,
        vec![
            DrawnPart::Text(caption),
            DrawnPart::Image {
                mime: "image/png".to_owned(),
                bytes: png.clone()
            },
        ]
    );
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/rf-06-test");
    ok!(std::fs::create_dir_all(&dir));
    let saved = ok!(drawing.save(&dir, &stem_for("a lighthouse in fog")));
    let read = ok!(saved.first().map(std::fs::read).transpose());
    ok!(std::fs::remove_dir_all(&dir));
    assert_eq!(
        saved.first().and_then(|path| path.file_name()),
        Some("lighthouse-1.png".as_ref())
    );
    assert_eq!(read, Some(png));

    let modalities = fake
        .seen()
        .first()
        .and_then(|request| request.generation_config.clone())
        .map(|config| config.response_modalities)
        .unwrap_or_default();
    assert_eq!(modalities.len(), 2);
}

/// R4: 250 texts at 768 dimensions come back as 250 vectors of 768, in
/// input order.
#[tokio::test]
async fn embeddings_keep_order_and_width() {
    let fake = Fake::default();
    let (gem, server) = ok!(fake.serve().await);
    let texts: Vec<String> = (0..250).map(|i| format!("note {i}")).collect();
    let vectors = ok!(gem.embed("gemini-embedding-001", 768, texts).await);
    server.abort();

    assert_eq!(vectors.len(), 250);
    for (i, vector) in vectors.iter().enumerate() {
        assert_eq!(vector.len(), 768);
        assert_eq!(vector.first().copied(), Some(i as f64));
    }
    let embeds = fake
        .embeds
        .lock()
        .map(|embeds| embeds.clone())
        .unwrap_or_default();
    assert!(
        embeds
            .iter()
            .all(|request| request.output_dimensionality == Some(768))
    );
}

/// R6: quota exhaustion is retryable, a safety block reports its reason
/// and categories, and a stream cut after some text fails in bounded time
/// with that text.
#[tokio::test]
async fn failures_are_typed() {
    let mut blocked = message(
        Vec::new(),
        proto::candidate::FinishReason::ImageSafety as i32,
    );
    if let Some(candidate) = blocked.candidates.first_mut() {
        candidate.safety_ratings = vec![proto::SafetyRating {
            category: proto::HarmCategory::DangerousContent as i32,
            probability: proto::safety_rating::HarmProbability::High as i32,
            blocked: true,
        }];
    }
    let fake = Fake::new([
        vec![Err(tonic::Status::resource_exhausted(
            "Quota exceeded for generate_content.",
        ))],
        vec![Ok(blocked)],
        vec![
            Ok(message(
                vec![part(Data::Text("The lighthouse ".to_owned()))],
                0,
            )),
            Err(tonic::Status::unavailable("connection reset")),
        ],
    ]);
    let (gem, server) = ok!(fake.serve().await);

    let quota = gem.draw("gemini-2.5-flash-image", "x", None).await;
    assert!(
        matches!(&quota, Err(GemError::RateLimited { .. })),
        "{quota:?}"
    );
    assert!(quota.as_ref().err().is_some_and(GemError::is_retryable));

    let block = gem.draw("gemini-2.5-flash-image", "x", None).await;
    assert!(
        matches!(&block, Err(GemError::Blocked { categories, .. })
            if categories == &["HARM_CATEGORY_DANGEROUS_CONTENT"]),
        "{block:?}"
    );

    let cut = tokio::time::timeout(
        Duration::from_secs(10),
        gem.chat(settings(Thinking::Off), MemoryStore::default(), None)
            .send("Describe it.", |_| {}),
    )
    .await;
    server.abort();
    let cut = ok!(cut);
    assert!(
        matches!(&cut, Err(GemError::Cut { partial, .. }) if partial == "The lighthouse "),
        "{cut:?}"
    );
}
