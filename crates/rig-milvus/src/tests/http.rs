use std::io::{self, BufRead, Read, Write};
use std::net::TcpListener;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rig_core::providers::openai::OpenAIConfig;
use serde_json::Value;

#[derive(Debug)]
pub struct Request {
    pub path: String,
    pub body: Value,
}

pub struct Server {
    pub url: String,
    task: JoinHandle<io::Result<Vec<Request>>>,
}

impl Server {
    pub fn start(replies: Vec<(u16, String)>) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let url = format!("http://{}", listener.local_addr()?);
        let task = std::thread::spawn(move || {
            let mut requests = Vec::new();
            for (status, body) in replies {
                let deadline = Instant::now() + Duration::from_secs(10);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            if Instant::now() >= deadline {
                                return Err(io::Error::new(io::ErrorKind::TimedOut, "no request"));
                            }
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => return Err(error),
                    }
                };
                stream.set_read_timeout(Some(Duration::from_secs(10)))?;
                stream.set_write_timeout(Some(Duration::from_secs(10)))?;
                let mut reader = io::BufReader::new(&mut stream);
                let mut line = String::new();
                reader.read_line(&mut line)?;
                let path = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or_default()
                    .to_owned();
                let mut length = 0;
                loop {
                    line.clear();
                    if reader.read_line(&mut line)? == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "request headers",
                        ));
                    }
                    if line == "\r\n" {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':')
                        && name.eq_ignore_ascii_case("content-length")
                    {
                        length = value.trim().parse().map_err(io::Error::other)?;
                    }
                }
                let mut bytes = vec![0; length];
                reader.read_exact(&mut bytes)?;
                requests.push(Request {
                    path,
                    body: serde_json::from_slice(&bytes).map_err(io::Error::other)?,
                });
                write!(
                    stream,
                    "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )?;
            }
            Ok(requests)
        });
        Ok(Self { url, task })
    }

    pub fn store(&self) -> crate::MilvusVectorStore {
        let model = OpenAIConfig::new("fixture")
            .with_base_url(&self.url)
            .client()
            .embedding("fixture", Some(2));
        crate::MilvusVectorStore::new(
            model,
            self.url.clone(),
            "default".into(),
            "documents".into(),
        )
    }

    pub fn finish(self) -> io::Result<Vec<Request>> {
        self.task
            .join()
            .map_err(|_| io::Error::other("HTTP fixture panicked"))?
    }
}
