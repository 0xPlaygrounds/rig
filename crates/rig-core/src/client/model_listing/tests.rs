use super::*;
use crate::model::Model;
use crate::test_utils::MockModelLister;

#[tokio::test]
async fn test_model_lister_list_all() {
    let models = vec![
        Model::new("gpt-4", "GPT-4"),
        Model::new("gpt-3.5-turbo", "GPT-3.5 Turbo"),
    ];
    let lister = MockModelLister::new(models);

    let result =
        <MockModelLister as ModelLister<crate::test_utils::RecordingHttpClient>>::list_all(&lister)
            .await
            .unwrap();
    assert_eq!(result.len(), 2);
}
