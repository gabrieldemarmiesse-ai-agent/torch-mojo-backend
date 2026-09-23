"""Real rocprof export schemas, without executing a profiler or GPU kernels."""

import csv
import sqlite3
from pathlib import Path

import pytest

from scripts.rocprof_kernels import load_results, render, shorten


def _write_csv(path: Path):
    fields = [
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "Private_Segment_Size",
        "Group_Segment_Size",
        "Workgroup_Size_X",
        "Workgroup_Size_Y",
        "Workgroup_Size_Z",
        "Grid_Size_X",
        "Grid_Size_Y",
        "Grid_Size_Z",
    ]
    # A quoted comma in the name and timestamps above fp64's exact-integer
    # range exercise the two lossy shortcuts a hand-written parser might take.
    start = 2**54
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for name, duration in (("gemm<A, B>", 1001), ("gemm<A, B>", 2003), ("copy", 9)):
            writer.writerow(
                [name, start, start + duration, 16, 32768, 256, 1, 1, 4096, 2, 1]
            )


def test_csv_results_preserve_names_time_geometry_and_missing_registers(tmp_path: Path):
    node = tmp_path / "node"
    node.mkdir()
    _write_csv(node / "123_kernel_trace.csv")
    stats = load_results(tmp_path, "gemm")
    assert set(stats) == {"gemm<A, B>"}
    entry = stats["gemm<A, B>"]
    assert entry.durations_ns == [1001, 2003]
    assert entry.grid == (4096, 2, 1)
    assert entry.block == (256, 1, 1)
    assert (entry.lds, entry.scratch) == (32768, 16)
    assert (entry.vgpr, entry.accum_vgpr, entry.sgpr) == (None, None, None)
    assert "n/a" in render(stats, 10, 64)
    assert load_results(tmp_path, "absent") == {}


def test_database_export_prevents_double_counting_csv(tmp_path: Path):
    _write_csv(tmp_path / "123_kernel_trace.csv")
    with sqlite3.connect(tmp_path / "123_results.db") as db:
        db.execute("""create table kernels (
            name text, duration integer, grid_x integer, grid_y integer, grid_z integer,
            workgroup_x integer, workgroup_y integer, workgroup_z integer,
            vgpr_count integer, accum_vgpr_count integer, sgpr_count integer,
            lds_size integer, scratch_size integer)""")
        db.execute(
            "insert into kernels values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("gemm<A, B>", 1001, 4096, 2, 1, 256, 1, 1, 64, 32, 48, 32768, 16),
        )
    stats = load_results(tmp_path, None)
    assert stats["gemm<A, B>"].durations_ns == [1001]
    assert stats["gemm<A, B>"].vgpr == 64
    assert "n/a" not in render(stats, 10, 64)


def test_csv_results_merge_processes(tmp_path: Path):
    _write_csv(tmp_path / "123_kernel_trace.csv")
    _write_csv(tmp_path / "456_kernel_trace.csv")
    assert load_results(tmp_path, None)["gemm<A, B>"].durations_ns == [1001, 2003] * 2


def test_missing_results_is_an_error(tmp_path: Path):
    with pytest.raises(SystemExit, match="no kernel results"):
        load_results(tmp_path, None)


def test_anonymous_namespace_kernel_keeps_a_visible_name():
    assert (
        shorten("(anonymous namespace)::layer_norm(float*)", 64)
        == "anonymous::layer_norm"
    )
