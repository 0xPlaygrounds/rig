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
                theme.panel_surface,
                theme.raised_surface,
                theme.call_waiting,
                theme.call_running,
                theme.call_done,
                theme.call_failed,
                theme.input,
                theme.code,
                theme.note,
                theme.important,
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

#[test]
fn a_surface_leaves_what_a_border_would() {
    let area = ratatui::layout::Rect::new(0, 0, 20, 10);
    let bordered = Block::bordered().inner(area);
    let theme = Theme::default();
    assert_eq!(theme.panel("files").inner(area), bordered);
    assert_eq!(theme.raised("").inner(area), bordered);
}

#[test]
fn a_plain_theme_marks_with_reverse_and_underline_instead_of_tints() {
    use ratatui::style::Modifier;

    let theme = Theme::default();
    assert_eq!(theme.selection().bg, Some(theme.selected));
    let plain = theme.without_color();
    assert!(plain.selection().add_modifier.contains(Modifier::REVERSED));
    assert_eq!(plain.selection().bg, None);
    let word = plain.changed_word(plain.added_word);
    assert!(word.add_modifier.contains(Modifier::UNDERLINED));
}

/// WCAG's relative luminance of `rgb`.
fn luminance((r, g, b): Rgb) -> f64 {
    let channel = |value: u8| {
        let value = f64::from(value) / 255.0;
        match value <= 0.039_28 {
            true => value / 12.92,
            false => ((value + 0.055) / 1.055).powf(2.4),
        }
    };
    0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
}

/// WCAG's contrast ratio of `a` and `b`, from 1 to 21.
fn contrast(a: Rgb, b: Rgb) -> f64 {
    let (a, b) = (luminance(a), luminance(b));
    (a.max(b) + 0.05) / (a.min(b) + 0.05)
}

#[test]
fn text_reads_on_every_tint_of_dark_and_light_terminals() {
    let backgrounds = [
        DARK,
        (0, 0, 0),
        (0x28, 0x2c, 0x34),
        (0xfa, 0xfa, 0xfa),
        (0xff, 0xff, 0xff),
        (0xfd, 0xf6, 0xe3),
    ];
    for background in backgrounds {
        let theme = Theme::on(background, true);
        let toward = match lightness(background) < 0.5 {
            true => (0xff, 0xff, 0xff),
            false => (0, 0, 0),
        };
        // A terminal's own foreground: most of the way from its background.
        let text = mix(background, toward, 0.8);
        let tints = [
            theme.user,
            theme.tool,
            theme.added,
            theme.added_word,
            theme.removed,
            theme.removed_word,
            theme.selected,
            theme.panel_surface,
            theme.raised_surface,
            theme.input,
            theme.call_waiting,
            theme.call_running,
            theme.call_done,
            theme.call_failed,
            theme.code,
            theme.note,
            theme.important,
        ];
        for tint in tints {
            let Color::Rgb(r, g, b) = tint else {
                continue;
            };
            let ratio = contrast(text, (r, g, b));
            assert!(
                ratio >= 4.5,
                "{background:?}: {text:?} on {tint:?} is {ratio:.2}"
            );
        }
    }
}

#[test]
fn a_shade_leans_from_the_background_toward_the_far_lightness() {
    let level = |color: Color| match color {
        Color::Rgb(r, _, _) => i32::from(r),
        _ => -1,
    };
    let dark = Theme::on(DARK, true);
    assert!(level(dark.shade(0.2)) < level(dark.shade(0.8)));
    let light = Theme::on((0xfa, 0xfa, 0xfa), true);
    assert!(level(light.shade(0.2)) > level(light.shade(0.8)));
    let indexed = Theme::on(DARK, false).shade(0.85);
    assert!(matches!(indexed, Color::Indexed(232..=255)), "{indexed:?}");
}
