"""The parser: formats it accepts, what it refuses, and the rows it leaves unparsed."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic_cmapss import cycles_of, rows_text, train_text

from closed_loop_v2.data.errors import DataError
from closed_loop_v2.data.parse import N_COLUMNS, parse_rul, parse_test, parse_training
from closed_loop_v2.data.roles import role_of
from closed_loop_v2.types import Role, Trajectory

N = 30
NAME = "train_FD001.txt"


def _units_with(role: str) -> list[int]:
    return [u for u in range(1, N + 1) if role_of("FD001", u) == role]


def _parse(
    text: str, roles: tuple[Role, ...] = ("dev", "calib"), n: int = N
) -> dict[Role, list[Trajectory]]:
    return parse_training(text.encode("ascii"), NAME, "FD001", n, roles)


def test_every_role_is_present_among_the_synthetic_units() -> None:
    assert all(len(_units_with(r)) >= 2 for r in ("dev", "calib", "eval"))


def test_parse_training_returns_the_units_of_the_asked_roles_in_unit_order() -> None:
    parsed = _parse(train_text(N, seed=1))
    assert set(parsed) == {"dev", "calib"}
    assert [t.unit for t in parsed["dev"]] == _units_with("dev")
    assert [t.unit for t in parsed["calib"]] == _units_with("calib")
    for traj in parsed["dev"] + parsed["calib"]:
        assert traj.subset == "FD001"
        assert traj.source == "train"
        assert traj.n == cycles_of(traj.unit)
        assert traj.rul_at_end == 0
        assert traj.rul()[-1] == 0
        assert traj.settings.shape == (traj.n, 3)
        assert traj.sensors.shape == (traj.n, 21)
        assert traj.settings.dtype == np.float64 and traj.sensors.dtype == np.float64


def test_values_are_the_values_in_the_file() -> None:
    text = "1  1  0.5  -0.25  100.0  " + "  ".join(str(float(i)) for i in range(21)) + "  \r\n"
    parsed = parse_training(text.encode(), NAME, "FD001", 1, ("dev", "calib", "eval"))
    (traj,) = sum(parsed.values(), [])
    assert traj.settings.tolist() == [[0.5, -0.25, 100.0]]
    assert traj.sensors.tolist() == [[float(i) for i in range(21)]]


def test_arrays_are_read_only() -> None:
    traj = _parse(train_text(N, seed=1))["dev"][0]
    for array in (traj.cycles, traj.settings, traj.sensors):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = array[0]


def test_line_endings_and_trailing_spaces_and_blank_lines_are_accepted() -> None:
    text = train_text(N, seed=1)
    for variant in (text, text.replace("\r\n", "\n"), text + "\r\n\r\n", "\n" + text):
        assert [t.unit for t in _parse(variant)["dev"]] == _units_with("dev")


def test_eval_rows_are_not_parsed_when_only_dev_and_calib_are_asked_for() -> None:
    eval_units = set(_units_with("eval"))
    broken = []
    for line in train_text(N, seed=1).splitlines(keepends=True):
        if int(line.split()[0]) in eval_units:
            head = " ".join(line.split()[:2])
            line = head + "  not-a-number  1.0\r\n"  # wrong width, and not numbers
        broken.append(line)
    text = "".join(broken)
    parsed = _parse(text)
    assert [t.unit for t in parsed["dev"]] == _units_with("dev")
    with pytest.raises(DataError):
        _parse(text, roles=("eval",))


def test_only_the_unit_column_of_every_row_is_read_for_other_roles() -> None:
    """A row of an eval unit may be wrong anywhere but its first column; the unit column is read."""
    eval_unit = _units_with("eval")[0]
    text = train_text(N, seed=1) + f"{eval_unit}  junk\r\n"  # after the last unit: not together
    with pytest.raises(DataError, match=rf"unit {eval_unit} are not together"):
        _parse(text)


def test_a_non_integer_unit_is_refused_naming_file_and_line_without_the_value() -> None:
    lines = train_text(N, seed=1).splitlines(keepends=True)
    lines[3] = lines[3].replace(lines[3].split()[0], "1.5", 1)
    with pytest.raises(DataError) as info:
        _parse("".join(lines))
    message = str(info.value)
    assert NAME in message and "line 4" in message and "column 1" in message
    assert "1.5" not in message


def test_a_non_integer_cycle_is_refused_for_a_parsed_unit() -> None:
    dev_unit = _units_with("dev")[0]
    lines = train_text(N, seed=1).splitlines(keepends=True)
    index = next(i for i, ln in enumerate(lines) if int(ln.split()[0]) == dev_unit)
    parts = lines[index].split()
    parts[1] = "2.0"
    lines[index] = "  ".join(parts) + "\r\n"
    with pytest.raises(DataError, match=rf"line {index + 1}: column 2"):
        _parse("".join(lines))


@pytest.mark.parametrize("token", ["-1", "+1", "1e3", "0x10", "1_0", "nan"])
def test_unit_column_must_be_plain_digits(token: str) -> None:
    lines = train_text(N, seed=1).splitlines(keepends=True)
    lines[0] = lines[0].replace(lines[0].split()[0], token, 1)
    with pytest.raises(DataError, match="column 1"):
        _parse("".join(lines))


def test_a_wrong_column_count_is_refused_for_a_parsed_unit() -> None:
    dev_unit = _units_with("dev")[0]
    lines = train_text(N, seed=1).splitlines(keepends=True)
    index = next(i for i, ln in enumerate(lines) if int(ln.split()[0]) == dev_unit)
    lines[index] = "  ".join(lines[index].split()[:-1]) + "\r\n"
    with pytest.raises(DataError, match=rf"line {index + 1}: {N_COLUMNS - 1} columns"):
        _parse("".join(lines))


def test_a_non_numeric_or_non_finite_sensor_value_is_refused() -> None:
    dev_unit = _units_with("dev")[0]
    for bad in ("abc", "nan", "inf"):
        lines = train_text(N, seed=1).splitlines(keepends=True)
        index = next(i for i, ln in enumerate(lines) if int(ln.split()[0]) == dev_unit)
        parts = lines[index].split()
        parts[10] = bad
        lines[index] = "  ".join(parts) + "\r\n"
        with pytest.raises(DataError, match=rf"unit {dev_unit}"):
            _parse("".join(lines))


def test_cycles_must_run_1_to_n() -> None:
    dev_unit = _units_with("dev")[0]
    lines = train_text(N, seed=1).splitlines(keepends=True)
    index = next(i for i, ln in enumerate(lines) if int(ln.split()[0]) == dev_unit)
    parts = lines[index].split()
    parts[1] = "9"
    lines[index] = "  ".join(parts) + "\r\n"
    with pytest.raises(DataError, match=rf"unit {dev_unit}.*cycles are not 1"):
        _parse("".join(lines))


def test_missing_unit_numbers_are_refused() -> None:
    text = train_text(N, seed=1)
    kept = "".join(ln for ln in text.splitlines(keepends=True) if int(ln.split()[0]) != 7)
    with pytest.raises(DataError, match=r"missing \[7\]"):
        _parse(kept)


def test_extra_unit_numbers_are_refused() -> None:
    text = train_text(N + 1, seed=1)
    with pytest.raises(DataError, match=rf"unexpected \[{N + 1}\]"):
        _parse(text)


def test_units_out_of_order_are_refused() -> None:
    text = train_text(N, seed=1)
    lines = text.splitlines(keepends=True)
    swapped = "".join(lines[len(lines) // 2 :] + lines[: len(lines) // 2])
    with pytest.raises(DataError, match="not together"):
        _parse(swapped)


def test_an_empty_file_is_refused() -> None:
    with pytest.raises(DataError, match="no rows"):
        _parse("\r\n\r\n")


def test_a_non_ascii_byte_is_refused() -> None:
    data = train_text(N, seed=1).encode("ascii") + b"\xff"
    with pytest.raises(DataError, match="not ASCII"):
        parse_training(data, NAME, "FD001", N, ("dev",))


def test_parse_rul() -> None:
    assert parse_rul(b"112\r\n98\r\n69\r\n", "RUL_FD001.txt", 3) == [112, 98, 69]
    assert parse_rul(b"112 \r\n\r\n98 \r\n", "RUL_FD001.txt", 2) == [112, 98]


@pytest.mark.parametrize("data", [b"1\r\n2\r\n", b"1\r\n2\r\n3\r\n4\r\n", b""])
def test_parse_rul_refuses_a_wrong_count(data: bytes) -> None:
    with pytest.raises(DataError, match=r"RUL_FD001\.txt"):
        parse_rul(data, "RUL_FD001.txt", 3)


@pytest.mark.parametrize("data", [b"1\r\n-2\r\n3\r\n", b"1\r\n2.5\r\n3\r\n", b"1\r\n2 3\r\n3\r\n"])
def test_parse_rul_refuses_a_bad_value(data: bytes) -> None:
    with pytest.raises(DataError, match=r"RUL_FD001\.txt line 2"):
        parse_rul(data, "RUL_FD001.txt", 3)


def test_parse_test_assigns_rul_by_unit_order() -> None:
    text = "".join(rows_text(u, 6, np.random.default_rng(u)) for u in range(1, 4))
    trajs = parse_test(text.encode(), "test_FD001.txt", "FD001", 3, [30, 20, 10])
    assert [(t.unit, t.source, t.rul_at_end, t.n) for t in trajs] == [
        (1, "test", 30, 6),
        (2, "test", 20, 6),
        (3, "test", 10, 6),
    ]
    assert trajs[0].failure_cycle == 36
    assert trajs[0].rul()[0] == 35 and trajs[0].rul()[-1] == 30
