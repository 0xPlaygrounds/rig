use super::{Below, Scroll};

const WIDTH: u16 = 80;

#[test]
fn a_new_view_follows_the_bottom() {
    let mut scroll = Scroll::default();
    assert_eq!(scroll.place(&[30, 20], 10, WIDTH), (40, Below::Nothing));
    assert_eq!(scroll.place(&[30, 25], 10, WIDTH), (45, Below::Nothing));
}

#[test]
fn a_view_scrolled_up_stays_put_while_rows_arrive() {
    let mut scroll = Scroll::default();
    scroll.up(10);
    assert_eq!(scroll.place(&[30, 20], 10, WIDTH), (30, Below::More));
    // The streaming reply grows, and becomes a message of its own.
    assert_eq!(scroll.place(&[30, 28], 10, WIDTH), (30, Below::New));
    assert_eq!(scroll.place(&[30, 28, 5], 10, WIDTH), (30, Below::New));
}

#[test]
fn scrolling_back_to_the_bottom_follows_again() {
    let mut scroll = Scroll::default();
    scroll.up(10);
    assert_eq!(scroll.place(&[50], 10, WIDTH), (30, Below::More));
    scroll.down(4);
    assert_eq!(scroll.place(&[60], 10, WIDTH), (34, Below::New));
    scroll.down(100);
    assert_eq!(scroll.place(&[60], 10, WIDTH), (50, Below::Nothing));
    assert_eq!(scroll.place(&[70], 10, WIDTH), (60, Below::Nothing));
}

#[test]
fn scrolling_up_stops_at_the_top() {
    let mut scroll = Scroll::default();
    scroll.up(100);
    assert_eq!(scroll.place(&[50], 10, WIDTH), (0, Below::More));
}

#[test]
fn a_resize_keeps_the_same_part_on_top() {
    let mut scroll = Scroll::default();
    scroll.up(25);
    // Row 5 of the second part.
    assert_eq!(scroll.place(&[20, 10, 30], 10, WIDTH), (25, Below::More));
    // At half the width every part takes twice the rows.
    assert_eq!(
        scroll.place(&[40, 20, 60], 10, WIDTH / 2),
        (50, Below::More)
    );
}

#[test]
fn following_again_drops_the_held_row() {
    let mut scroll = Scroll::default();
    scroll.up(10);
    assert_eq!(scroll.place(&[50], 10, WIDTH), (30, Below::More));
    scroll.follow();
    assert_eq!(scroll.place(&[55], 10, WIDTH), (45, Below::Nothing));
}
