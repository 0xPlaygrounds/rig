use std::borrow::Cow;

use rig_core::completion::Message;
use rig_core::message::{
    CallId, DocumentSourceKind, ImageMediaType, ToolName, ToolResultContent, UserContent,
};

use super::{JournalStore, MemoryStore, load_images, store_images};

#[test]
fn images_are_logged_as_blobs_and_read_back() {
    let store = MemoryStore::default();
    let text = Message::user("no images");
    assert!(matches!(store_images(&text, &store), Ok(Cow::Borrowed(_))));
    // The same bytes as a user image and in a tool result.
    let png = DocumentSourceKind::Base64("cG5n".to_owned());
    let image = UserContent::image_raw(b"png".to_vec(), Some(ImageMediaType::PNG), None);
    let UserContent::Image(shot) = image.clone() else {
        panic!("an image");
    };
    let (call, name) = (
        CallId::from_wire("c"),
        ToolName::new("shot").expect("a name"),
    );
    let result = UserContent::tool_result(call, name, vec![ToolResultContent::Image(shot)]);
    let message = Message::User {
        content: vec![image, result],
    };

    let logged = store_images(&message, &store).expect("stored").into_owned();
    let names: Vec<_> = logged.images().map(|image| image.data.clone()).collect();
    let Some(DocumentSourceKind::Url(url)) = names.first() else {
        panic!("a blob name, got {names:?}");
    };
    let blob = url.strip_prefix("blob:").expect("a blob");
    assert!(names.len() == 2 && names.iter().all(|name| Some(name) == names.first()));
    assert!(blob.ends_with(".png") && store.blob(blob).expect("the blob") == b"png");

    let mut read = logged.clone();
    load_images(&mut read, &store).expect("loaded");
    assert!(read.images().all(|image| image.data == png));
    // Without its blob, an image has no data.
    let mut gone = logged;
    assert!(load_images(&mut gone, &MemoryStore::default()).is_err());
    assert!(
        gone.images()
            .all(|image| image.data == DocumentSourceKind::Unknown)
    );
}
