from mm2026.utils.ids import format_matchup_id, parse_matchup_id


def test_parse_matchup_id_orders_team_ids() -> None:
    mid = parse_matchup_id("2026_1103_1101")
    assert mid.season == 2026
    assert mid.team_low == 1101
    assert mid.team_high == 1103


def test_format_matchup_id_orders_team_ids() -> None:
    assert format_matchup_id(2026, 1103, 1101) == "2026_1101_1103"
