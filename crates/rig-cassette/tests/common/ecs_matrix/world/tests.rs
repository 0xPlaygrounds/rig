use super::join_logs;
use rig_cassette::effect_log::{EffectLog, RecordedStreamError};
use rig_core::{
    effect::{Delivery, DeliveryKind, EffectId},
    error::{ErrorKind, ErrorReport},
};

#[test]
fn joining_a_restored_tail_keeps_deliveries_and_stream_errors() {
    let mut head = EffectLog::default();
    head.header.deliveries = Some(vec![Delivery {
        batch: 7,
        id: EffectId::from_raw(0),
        kind: DeliveryKind::Outcome,
    }]);
    let mut tail = EffectLog::default();
    tail.header.deliveries = Some(vec![
        Delivery {
            batch: 2,
            id: EffectId::from_raw(1),
            kind: DeliveryKind::Stream { items: 1 },
        },
        Delivery {
            batch: 2,
            id: EffectId::from_raw(2),
            kind: DeliveryKind::Outcome,
        },
        Delivery {
            batch: 3,
            id: EffectId::from_raw(1),
            kind: DeliveryKind::Outcome,
        },
    ]);
    let error = RecordedStreamError {
        item: 0,
        error: ErrorReport::new(ErrorKind::ProviderResponse, "stream error"),
    };
    tail.header
        .stream_errors
        .insert(EffectId::from_raw(1), vec![error.clone()]);
    tail.header
        .delivery_limitations
        .push("unobserved delivery".into());
    let log = join_logs(head, tail);
    let deliveries = log.header.deliveries.expect("both delivery traces");
    assert_eq!(
        deliveries
            .iter()
            .map(|delivery| (delivery.batch, delivery.id.as_u64()))
            .collect::<Vec<_>>(),
        [(7, 0), (9, 1), (9, 2), (10, 1)]
    );
    assert_eq!(log.header.stream_errors[&EffectId::from_raw(1)], [error]);
    assert_eq!(log.header.delivery_limitations, ["unobserved delivery"]);
}
