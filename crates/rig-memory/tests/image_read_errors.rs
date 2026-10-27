#![cfg(feature = "fs")]

use std::future::Future;
use std::io;
use std::path::PathBuf;
use std::task::{Context, Poll, Waker};

use rig_core::conversation::Conversation;
use rig_core::message::{DocumentSourceKind, ImageMediaType, Message, UserContent};
use rig_memory::{DirStore, Files, SessionError, load, load_async, save, store_images};

type TestResult = Result<(), Box<dyn std::error::Error>>;

struct Fixture {
    dir: PathBuf,
    store: DirStore,
    first: PathBuf,
    second: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Fixture {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "rig-image-read-audit-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir(&dir)?;
        let store = DirStore::open(&dir)?;
        let first = UserContent::image_base64(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRz8AAAAASUVORK5CYII=",
            Some(ImageMediaType::PNG),
            None,
        );
        let second = UserContent::image_base64(
            "R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==",
            Some(ImageMediaType::GIF),
            None,
        );
        let forward = Message::User {
            content: vec![first.clone(), second.clone()],
        };
        let stored = store_images(&forward, &store)?;
        let mut names = stored.images().filter_map(|image| match &image.data {
            DocumentSourceKind::Url(url) => url.strip_prefix("blob:"),
            _ => None,
        });
        let first_path = names
            .next()
            .ok_or_else(|| io::Error::other("missing first blob"))?;
        let second_path = names
            .next()
            .ok_or_else(|| io::Error::other("missing second blob"))?;
        let first_path = dir.join("files/blobs").join(first_path);
        let second_path = dir.join("files/blobs").join(second_path);
        drop(names);
        drop(stored);
        let reverse = Message::User {
            content: vec![second.clone(), first],
        };
        let second_only = Message::User {
            content: vec![second],
        };
        for (log, message) in [
            ("forward", forward),
            ("reverse", reverse),
            ("second", second_only),
        ] {
            let mut conversation = Conversation::from_messages([message]);
            save(&store, log, &mut conversation)?;
        }
        Ok(Self {
            dir,
            store,
            first: first_path,
            second: second_path,
        })
    }

    fn make_second_unreadable(&self) -> io::Result<()> {
        // Keep the original bytes so the obstruction is reversible.
        std::fs::rename(&self.second, self.dir.join("second-image-backup"))?;
        std::fs::create_dir(&self.second)
    }
}

fn show(label: &str, result: &Result<Conversation, SessionError>) {
    match result {
        Ok(conversation) => {
            let images: Vec<_> = conversation
                .messages()
                .iter()
                .flat_map(|message| message.images())
                .map(|image| match image.data {
                    DocumentSourceKind::Unknown => "Unknown",
                    DocumentSourceKind::Base64(_) => "Base64",
                    _ => "other",
                })
                .collect();
            println!("{label}: Ok, images={images:?}");
        }
        Err(error) => println!("{label}: Err({error:?})"),
    }
}

fn require_read_error(result: Result<Conversation, SessionError>) -> TestResult {
    match result {
        Err(SessionError::Io(error)) if error.kind() != io::ErrorKind::NotFound => Ok(()),
        _ => Err(
            io::Error::other("load suppressed the existing image's filesystem read error").into(),
        ),
    }
}

#[test]
fn missing_first_image_must_not_hide_second_image_read_error() -> TestResult {
    let fixture = Fixture::new()?;
    std::fs::remove_file(&fixture.first)?;
    fixture.make_second_unreadable()?;
    let result = load(&fixture.store, "forward");
    show("missing image then unreadable image", &result);
    require_read_error(result)
}

#[test]
fn async_missing_first_image_must_not_hide_second_image_read_error() -> TestResult {
    let fixture = Fixture::new()?;
    std::fs::remove_file(&fixture.first)?;
    fixture.make_second_unreadable()?;
    let mut future = std::pin::pin!(load_async(&fixture.store, "forward"));
    let mut context = Context::from_waker(Waker::noop());
    let Poll::Ready(result) = future.as_mut().poll(&mut context) else {
        return Err(io::Error::other("synchronous store unexpectedly yielded").into());
    };
    show("async missing image then unreadable image", &result);
    require_read_error(result)
}

#[test]
fn reverse_order_and_single_unreadable_image_fail_the_load() -> TestResult {
    let fixture = Fixture::new()?;
    std::fs::remove_file(&fixture.first)?;
    fixture.make_second_unreadable()?;
    for log in ["reverse", "second"] {
        let result = load(&fixture.store, log);
        show(log, &result);
        require_read_error(result)?;
    }
    Ok(())
}

#[test]
fn genuinely_missing_image_is_a_placeholder_and_other_image_survives() -> TestResult {
    let fixture = Fixture::new()?;
    std::fs::remove_file(&fixture.first)?;
    let result = load(&fixture.store, "forward");
    show("missing image then readable image", &result);
    let conversation = result?;
    let images: Vec<_> = conversation
        .messages()
        .iter()
        .flat_map(|message| message.images())
        .collect();
    if !matches!(
        images.first().map(|image| &image.data),
        Some(DocumentSourceKind::Unknown)
    ) || !matches!(
        images.get(1).map(|image| &image.data),
        Some(DocumentSourceKind::Base64(_))
    ) {
        return Err(io::Error::other("missing-only control failed").into());
    }
    // The second file remains present in this control.
    let name = fixture
        .second
        .file_name()
        .ok_or_else(|| io::Error::other("missing filename"))?;
    fixture
        .store
        .read_file(&format!("blobs/{}", name.to_string_lossy()))?;
    Ok(())
}
