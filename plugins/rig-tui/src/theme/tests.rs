use super::*;

#[test]
fn every_band_is_its_own_color_on_dark_and_light_terminals() {
    for background in [DARK, (0xfa, 0xfa, 0xfa)] {
        for truecolor in [true, false] {
            let theme = Theme::on(background, truecolor);
            let bands = [
                theme.user,
                theme.tool,
                theme.added,
                theme.added_word,
                theme.removed,
                theme.removed_word,
                theme.selected,
            ];
            let mut distinct = bands.to_vec();
            distinct.sort_by_key(|color| format!("{color:?}"));
            distinct.dedup();
            assert_eq!(distinct.len(), bands.len(), "{theme:?}");
        }
    }
}

#[test]
fn a_band_is_a_tint_of_the_background() {
    let Color::Rgb(r, g, b) = Theme::on((0, 0, 0), true).tool else {
        return;
    };
    assert!(r < 40 && g < 40 && b < 40, "{r} {g} {b}");
    let Color::Rgb(r, g, b) = Theme::on((0xff, 0xff, 0xff), true).tool else {
        return;
    };
    assert!(r > 215 && g > 215 && b > 215, "{r} {g} {b}");
}
