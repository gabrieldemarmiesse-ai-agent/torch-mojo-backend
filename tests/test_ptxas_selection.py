"""Choosing the assembler: which ptxas suits this driver and this GPU.

The rules under test are NVIDIA's, not ours -- a 13.x cubin needs r580, a
12.x one runs on any r525+ driver, CUDA 11.8 has no `sm_90a` and CUDA 13 has
no `sm_70` -- so the cases below are written as the machines they describe:
an r570 box with a torch wheel carrying a CUDA 13 ptxas, a Blackwell that
needs a newer assembler than the one pinned, a Volta on a driver new enough
that the newest assembler has dropped it.
"""

import os
import sys
from pathlib import Path

import pytest

from torch_mojo_backend import _ptxas

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the nvcc wheel is a Linux dev dependency"
)

WHEEL_12_8 = _ptxas.Ptxas(
    Path("/wheel/nvidia/cuda_nvcc/bin/ptxas"),
    "nvidia-cuda-nvcc-cu12 wheel",
    (12, 8, 93),
)
TORCH_13_0 = _ptxas.Ptxas(Path("/wheel/torch/bin/ptxas"), "torch wheel", (13, 0, 88))
SYSTEM_11_8 = _ptxas.Ptxas(
    Path("/usr/local/cuda-11.8/bin/ptxas"), "system CUDA", (11, 8, 89)
)
MISSING = _ptxas.Ptxas(Path("/nonexistent/ptxas"), _ptxas.ENV_VAR, None)
BUILTIN_13_1 = _ptxas.Ptxas(Path("/wheel/max"), _ptxas.BUILTIN_SOURCE, (13, 1, 0))
BUILTIN_12_9 = _ptxas.Ptxas(Path("/wheel/max"), _ptxas.BUILTIN_SOURCE, (12, 9, 0))

# What each of those really lists in `ptxas --help`, abbreviated to the
# architectures these tests reason about.
ARCHES = {
    SYSTEM_11_8.path: frozenset({"sm_70", "sm_80", "sm_86", "sm_89", "sm_90"}),
    WHEEL_12_8.path: frozenset(
        {"sm_70", "sm_80", "sm_90", "sm_90a", "sm_100a", "sm_120a"}
    ),
    TORCH_13_0.path: frozenset(
        {"sm_80", "sm_90", "sm_90a", "sm_100a", "sm_110a", "sm_120a"}
    ),
}


@pytest.fixture
def arches(monkeypatch):
    """`ptxas --help` without the binaries."""
    monkeypatch.setattr(_ptxas, "arches_of", lambda path: ARCHES.get(path, frozenset()))


def test_a_cuda_13_assembler_is_refused_on_an_r570_driver(arches):
    chosen, rejected = _ptxas.choose((12, 8), (90,), [WHEEL_12_8, TORCH_13_0])
    assert chosen is WHEEL_12_8
    assert [(p.path, r) for p, r in rejected] == [(TORCH_13_0.path, rejected[0][1])]
    assert "r580" in rejected[0][1] and "CUDA 12.8" in rejected[0][1]


def test_the_newest_that_fits_wins(arches):
    chosen, rejected = _ptxas.choose((13, 0), (90,), [WHEEL_12_8, TORCH_13_0])
    assert chosen is TORCH_13_0
    assert rejected == []
    # 13.1 is newer than the 13.0 the driver reports, and still wins: any
    # 13.x cubin loads on an r580 driver (minor version compatibility).
    chosen, _ = _ptxas.choose((13, 0), (90,), [WHEEL_12_8, TORCH_13_0, BUILTIN_13_1])
    assert chosen is BUILTIN_13_1


def test_a_gpu_the_pinned_wheel_cannot_target_moves_to_a_newer_one(arches):
    """sm_110 exists only from CUDA 13: the pin is not usable, torch's is."""
    chosen, rejected = _ptxas.choose((13, 0), (110,), [WHEEL_12_8, TORCH_13_0])
    assert chosen is TORCH_13_0
    assert (
        "too old for this GPU: cannot target sm_110a"
        in dict((p.path, r) for p, r in rejected)[WHEEL_12_8.path]
    )


def test_a_gpu_the_newest_assembler_dropped_moves_to_an_older_one(arches):
    """CUDA 13 dropped Volta; on an r580 driver sm_70 needs the 12.x wheel."""
    chosen, _ = _ptxas.choose((13, 0), (70,), [TORCH_13_0, WHEEL_12_8])
    assert chosen is WHEEL_12_8


def test_an_assembler_without_the_a_variant_is_refused(arches):
    """CUDA 11.8 knows sm_90 but not sm_90a, which is what MAX targets on an
    H100; taking the base name for support is how the build dies with
    `Value 'sm_90a' is not defined`."""
    chosen, rejected = _ptxas.choose((12, 8), (90,), [SYSTEM_11_8])
    assert chosen is None
    assert "too old for this GPU: cannot target sm_90a" in rejected[0][1]


def test_every_gpu_present_has_to_be_supported(arches):
    chosen, rejected = _ptxas.choose((13, 0), (90, 110), [WHEEL_12_8, TORCH_13_0])
    assert chosen is TORCH_13_0
    assert "sm_110a" in rejected[0][1]


def test_a_path_that_is_not_there_is_reported_as_such(arches):
    chosen, rejected = _ptxas.choose((12, 8), (90,), [MISSING])
    assert chosen is None
    assert rejected[0][1] == "no such file"


def test_an_unparsable_help_output_is_not_a_rejection(arches):
    """A future CUDA that words its --help differently must not lose the user
    their device."""
    unknown = _ptxas.Ptxas(Path("/unknown/ptxas"), "system CUDA", (12, 6, 0))
    chosen, _ = _ptxas.choose((12, 8), (90,), [unknown])
    assert chosen is unknown


# -- MAX's own compiler: what runs when nothing is set -------------------------


def test_the_builtin_compiler_is_judged_by_the_tables():
    """It has no --help to ask, so it is read off ARCH_MIN_CUDA and
    ARCH_LAST_MAJOR: a CUDA 13 release knows sm_110a and has dropped sm_70."""
    supported = _ptxas.table_arches((13, 1, 0))
    assert {"sm_90a", "sm_110a"} <= supported
    assert "sm_70" not in supported and "sm_101a" not in supported
    assert "sm_70" in _ptxas.table_arches((12, 9, 0))


def test_the_builtin_is_ranked_like_any_other(arches):
    """Where an assembler came from does not enter the ranking."""
    chosen, _ = _ptxas.choose((13, 0), (110,), [BUILTIN_13_1, TORCH_13_0])
    assert chosen is BUILTIN_13_1
    chosen, _ = _ptxas.choose((12, 9), (90,), [WHEEL_12_8, BUILTIN_12_9])
    assert chosen is BUILTIN_12_9
    chosen, _ = _ptxas.choose((13, 0), (90,), [BUILTIN_12_9, TORCH_13_0])
    assert chosen is TORCH_13_0


def test_the_builtin_is_bound_by_the_driver_like_any_other(arches):
    chosen, rejected = _ptxas.choose((12, 8), (90,), [BUILTIN_13_1])
    assert chosen is None
    assert rejected[0][1].startswith("too new for this driver")


def test_the_builtin_is_chosen_over_a_refusal(arches):
    """An r580 box with a Blackwell and only the cu12 wheel installed used to
    be refused, though leaving the variable unset would have worked."""
    chosen, _ = _ptxas.choose((13, 0), (110,), [WHEEL_12_8, BUILTIN_13_1])
    assert chosen is BUILTIN_13_1


def test_the_installed_max_has_a_builtin_entry():
    builtin = _ptxas.builtin_ptxas()
    assert builtin is not None and builtin.is_builtin
    assert builtin.version is not None and builtin.version >= (12, 9, 0)
    assert builtin in _ptxas.candidates()


# -- what the user is told when nothing works ---------------------------------


def test_the_advice_names_the_wheel_for_this_driver():
    assert _ptxas.install_advice((12, 8), (90,))[0].startswith(
        'pip install "nvidia-cuda-nvcc-cu12==12.8.*"'
    )
    assert (
        'pip install "nvidia-cuda-nvcc==13.0.*"'
        in _ptxas.install_advice((13, 0), (110,))[0]
    )


def test_the_advice_allows_a_wheel_newer_than_the_dev_pin():
    """The development pin does not constrain a user's runtime install."""
    advice = _ptxas.install_advice((12, 9), (103,))[0]
    assert 'pip install "nvidia-cuda-nvcc-cu12>=12.9,<13"' in advice
    assert 'uv add "nvidia-cuda-nvcc-cu12>=12.9,<13"' in advice
    assert "--no-deps" not in advice


def test_the_advice_asks_for_the_other_wheel_when_it_is_not_the_pinned_one():
    """The CUDA 13 wheel is a different project, so nothing forbids it."""
    advice = _ptxas.install_advice((13, 1), (110,))[0]
    assert 'pip install "nvidia-cuda-nvcc==13.0.*"' in advice
    assert "--no-deps" not in advice


def test_the_advice_says_when_two_gpus_have_no_assembler_in_common():
    """A Volta beside a Blackwell-next: CUDA 13 targets one and dropped the
    other. Blaming the driver would send the user to fix the wrong thing."""
    advice = " ".join(_ptxas.install_advice((13, 0), (70, 110)))
    assert "no CUDA release targets sm_70, sm_110 at once" in advice
    assert "CUDA_VISIBLE_DEVICES" in advice
    assert "Update the NVIDIA driver" not in advice


def test_the_advice_says_a_pre_cuda_12_driver_is_the_real_problem():
    advice = " ".join(_ptxas.install_advice((11, 4), (80,)))
    assert "nvidia-cuda-nvcc-cu11==11.8.*" in advice
    assert "MAX's own runtime may need" in advice


def test_the_advice_says_to_update_the_driver_when_no_wheel_can_help():
    advice = " ".join(_ptxas.install_advice((12, 8), (110,)))
    assert "sm_110 needs CUDA 13.0" in advice
    assert "Update the NVIDIA driver" in advice
    assert "pip install" not in advice


def test_the_report_shows_every_candidate_and_its_verdict(arches):
    text = _ptxas.report(
        (12, 8), ((90, "NVIDIA H100 80GB HBM3"),), [WHEEL_12_8, TORCH_13_0]
    )
    assert "H100" in text and "sm_90" in text
    assert "[USED] /wheel/nvidia/cuda_nvcc/bin/ptxas" in text
    assert "[  no] /wheel/torch/bin/ptxas" in text
    assert "CUDA 13.0, from torch wheel -- too new for this driver" in text


def test_the_report_lists_newest_first_and_names_what_would_not_answer(arches):
    silent = _ptxas.Ptxas(Path(__file__), "system CUDA", None)  # exists, no answer
    text = _ptxas.report(
        (13, 0), ((90, "NVIDIA H100 80GB HBM3"),), [WHEEL_12_8, TORCH_13_0, silent]
    )
    order = [text.index(str(p.path)) for p in (TORCH_13_0, WHEEL_12_8, silent)]
    assert order == sorted(order)
    assert f"{silent.path}\n           --version failed, from system CUDA" in text


def test_the_report_marks_nothing_as_used_when_the_setting_is_broken(arches):
    """With an unusable MODULAR_NVPTX_COMPILER_PATH nothing assembles at all;
    the fallback is what *would* run, not what is running."""
    text = _ptxas.report(
        (12, 8),
        ((90, "NVIDIA H100 80GB HBM3"),),
        [_ptxas.Ptxas(TORCH_13_0.path, _ptxas.ENV_VAR, TORCH_13_0.version), WHEEL_12_8],
    )
    assert "[USED]" not in text
    assert "[pick] /wheel/nvidia/cuda_nvcc/bin/ptxas" in text
    assert "  fix" not in text  # there is one that works; installing is not it


def test_the_report_ends_with_what_to_install_when_nothing_works(arches):
    text = _ptxas.report((12, 8), ((90, "NVIDIA H100 80GB HBM3"),), [TORCH_13_0])
    assert 'pip install "nvidia-cuda-nvcc-cu12==12.8.*"' in text
    assert _ptxas.ENV_VAR in text


def test_diagnose_only_speaks_up_about_assembler_failures():
    assert _ptxas.diagnose("error: unbound variable `x`") == ""
    assert "ptxas found" in _ptxas.diagnose(
        "ptxas fatal : Value 'sm_90a' is not defined"
    )


# -- the check at registration ------------------------------------------------


def _machine(monkeypatch, *, driver, devices, found, env=None, ours=None):
    """Stand the process in front of a machine, without touching `check`'s own
    state beyond the latch it starts from."""
    monkeypatch.setattr(_ptxas, "_checked", False)
    monkeypatch.setattr(_ptxas, "driver_cuda_version", lambda: driver)
    monkeypatch.setattr(_ptxas, "device_arches", lambda: devices)
    monkeypatch.setattr(_ptxas, "candidates", lambda: found)
    monkeypatch.setattr(_ptxas, "arches_of", lambda path: ARCHES.get(path, frozenset()))
    monkeypatch.delenv(_ptxas.ENV_VAR, raising=False)
    monkeypatch.delenv(_ptxas.AUTO_ENV_VAR, raising=False)
    if ours is not None:  # what a previous pick of ours left behind
        monkeypatch.setenv(_ptxas.ENV_VAR, ours)
        monkeypatch.setenv(_ptxas.AUTO_ENV_VAR, ours)
    if env is not None:
        monkeypatch.setenv(_ptxas.ENV_VAR, env)


def _check(monkeypatch, **machine):
    _machine(monkeypatch, **machine)
    _ptxas.check()


def test_an_inherited_automatic_choice_is_not_mistaken_for_the_users(monkeypatch):
    """A child process inherits the environment, not the module state. With
    only MODULAR_NVPTX_COMPILER_PATH to go on, every torchrun rank and every
    compile worker would read our own pick as a setting it must not touch --
    and refuse to move off it for the GPU it actually has."""
    monkeypatch.setenv(_ptxas.ENV_VAR, str(WHEEL_12_8.path))
    monkeypatch.setenv(_ptxas.AUTO_ENV_VAR, str(WHEEL_12_8.path))
    assert _ptxas.explicit_choice() is None
    monkeypatch.setenv(_ptxas.ENV_VAR, "/somewhere/else/ptxas")
    assert _ptxas.explicit_choice() == "/somewhere/else/ptxas"


def test_check_is_silent_without_an_nvidia_driver(monkeypatch):
    _check(monkeypatch, driver=None, devices=(), found=[])  # AMD, Apple, CPU


def test_check_rewrites_the_import_time_choice_for_the_gpu(monkeypatch):
    """Import time knows the driver but not the architecture: a Blackwell-next
    box gets one choice at import and the right one here."""
    _check(
        monkeypatch,
        driver=(13, 0),
        devices=((110, "NVIDIA B300"),),
        found=[WHEEL_12_8, TORCH_13_0],
        ours=str(WHEEL_12_8.path),
    )
    assert os.environ[_ptxas.ENV_VAR] == str(TORCH_13_0.path)


def test_a_switched_assembler_is_traced(monkeypatch, capsys):
    """Assembling with something other than the wheel we ship is what a bug
    report has to mention, so it is said once, on stderr."""
    _check(
        monkeypatch,
        driver=(13, 0),
        devices=((110, "NVIDIA B300"),),
        found=[WHEEL_12_8, TORCH_13_0],
        ours=str(WHEEL_12_8.path),
    )
    line = capsys.readouterr().err
    assert str(TORCH_13_0.path) in line
    assert "cannot target sm_110a" in line


def test_check_raises_before_the_first_kernel_when_nothing_fits(monkeypatch):
    with pytest.raises(_ptxas.PtxasError) as excinfo:
        _check(
            monkeypatch,
            driver=(12, 8),
            devices=((90, "NVIDIA H100 80GB HBM3"),),
            found=[TORCH_13_0],
        )
    message = str(excinfo.value)
    assert "no usable ptxas" in message
    assert 'pip install "nvidia-cuda-nvcc-cu12==12.8.*"' in message
    assert "torch-mojo-backend ptxas" in message


def test_a_refusal_is_repeated_on_a_retried_registration(monkeypatch):
    """`register_mojo_devices()` leaves a failure retryable, so the second
    attempt has to hit the same refusal -- not sail past a check that latched
    itself on the way out and die later on a message that explains nothing."""
    _machine(
        monkeypatch,
        driver=(12, 8),
        devices=((90, "NVIDIA H100 80GB HBM3"),),
        found=[TORCH_13_0],
    )
    for _ in range(2):  # the second one is the registration being retried
        with pytest.raises(_ptxas.PtxasError):
            _ptxas.check()


def test_check_explains_a_users_own_setting_rather_than_overriding_it(monkeypatch):
    with pytest.raises(_ptxas.PtxasError) as excinfo:
        _check(
            monkeypatch,
            driver=(12, 8),
            devices=((90, "NVIDIA H100 80GB HBM3"),),
            found=[
                _ptxas.Ptxas(TORCH_13_0.path, _ptxas.ENV_VAR, TORCH_13_0.version),
                WHEEL_12_8,
            ],
            env=str(TORCH_13_0.path),
        )
    message = str(excinfo.value)
    assert f"{_ptxas.ENV_VAR}={TORCH_13_0.path} cannot be used here" in message
    assert "r580" in message
    assert f"Unset it and {WHEEL_12_8.path}" in message


def test_check_accepts_a_users_own_setting_that_works(monkeypatch):
    _check(
        monkeypatch,
        driver=(12, 8),
        devices=((90, "NVIDIA H100 80GB HBM3"),),
        found=[_ptxas.Ptxas(WHEEL_12_8.path, _ptxas.ENV_VAR, WHEEL_12_8.version)],
        env=str(WHEEL_12_8.path),
    )


def test_the_refusal_can_be_turned_into_a_warning(monkeypatch):
    """For a machine whose rules we got wrong: the build then fails on its
    own terms instead of ours."""
    monkeypatch.setenv(_ptxas.CHECK_ENV_VAR, "0")
    with pytest.warns(RuntimeWarning, match="no usable ptxas"):
        _check(
            monkeypatch,
            driver=(12, 8),
            devices=((90, "NVIDIA H100 80GB HBM3"),),
            found=[TORCH_13_0],
        )


def test_choosing_the_builtin_unsets_the_variable_and_marks_it(monkeypatch):
    _check(
        monkeypatch,
        driver=(13, 0),
        devices=((110, "NVIDIA B300"),),
        found=[WHEEL_12_8, BUILTIN_13_1],
        ours=str(WHEEL_12_8.path),
    )
    assert _ptxas.ENV_VAR not in os.environ
    assert os.environ[_ptxas.AUTO_ENV_VAR] == _ptxas.BUILTIN_MARK


def test_an_inherited_builtin_mark_is_not_a_setting(monkeypatch):
    """A rank spawned after the parent settled on the built-in sees no
    variable and the mark -- and the mark alone must not read as a choice."""
    monkeypatch.delenv(_ptxas.ENV_VAR, raising=False)
    monkeypatch.setenv(_ptxas.AUTO_ENV_VAR, _ptxas.BUILTIN_MARK)
    assert _ptxas.explicit_choice() is None
    _check(
        monkeypatch,
        driver=(13, 0),
        devices=((90, "NVIDIA H100 80GB HBM3"),),
        found=[WHEEL_12_8, BUILTIN_12_9, TORCH_13_0],
    )
    assert os.environ[_ptxas.ENV_VAR] == str(TORCH_13_0.path)


def test_a_broken_setting_is_told_the_builtin_would_work(monkeypatch):
    with pytest.raises(_ptxas.PtxasError) as excinfo:
        _check(
            monkeypatch,
            driver=(13, 0),
            devices=((110, "NVIDIA B300"),),
            found=[
                _ptxas.Ptxas(WHEEL_12_8.path, _ptxas.ENV_VAR, WHEEL_12_8.version),
                BUILTIN_13_1,
            ],
            env=str(WHEEL_12_8.path),
        )
    assert "Unset it and MAX's built-in assembler (CUDA 13.1)" in str(excinfo.value)


# -- the tables, against the assemblers actually installed here ---------------


def _real_ptxas() -> dict[tuple[int, int], Path]:
    found = {}
    for candidate in _ptxas.candidates():
        if candidate.version is not None and _ptxas.arches_of(candidate.path):
            found.setdefault(candidate.version[:2], candidate.path)
    return found


@pytest.mark.parametrize("arch", sorted(_ptxas.ARCH_MIN_CUDA))
def test_the_arch_table_agrees_with_the_installed_assemblers(arch):
    """ARCH_MIN_CUDA and ARCH_A_VARIANT are advice tables, and advice that
    drifts from the binaries is worse than none: every real ptxas on this
    machine is asked whether it targets what the tables claim."""
    real = _real_ptxas()
    if not real:
        pytest.skip("no ptxas installed to check the table against")
    needed = _ptxas.ARCH_MIN_CUDA[arch]
    last = _ptxas.ARCH_LAST_MAJOR.get(arch, 99)
    for release, path in real.items():
        supported = _ptxas.target_name(arch) in _ptxas.arches_of(path)
        expected = needed <= release and release[0] <= last
        assert supported == expected, (
            f"CUDA {release[0]}.{release[1]} at {path} "
            f"{'targets' if supported else 'does not target'} "
            f"{_ptxas.target_name(arch)}, table says otherwise"
        )


@pytest.mark.parametrize("wheel", [WHEEL_12_8.path, None])
def test_no_assembler_is_run_at_import_without_an_nvidia_driver(monkeypatch, wheel):
    """On AMD, on Apple, on a CPU-only box, every ptxas found would be a
    subprocess for nothing -- and one hung toolkit wrapper on a network mount
    would hold up the import for its timeout."""
    monkeypatch.setattr(_ptxas, "driver_cuda_version", lambda: None)
    monkeypatch.setattr(_ptxas, "wheel_ptxas", lambda: wheel)
    monkeypatch.setattr(
        _ptxas,
        "_version_of",
        lambda path: pytest.fail(f"ran {path} with no NVIDIA driver"),
    )
    monkeypatch.delenv(_ptxas.ENV_VAR, raising=False)
    monkeypatch.delenv(_ptxas.AUTO_ENV_VAR, raising=False)
    _ptxas.apply_default()
    expected = str(wheel) if wheel is not None else None
    assert os.environ.get(_ptxas.ENV_VAR) == expected
    assert os.environ.get(_ptxas.AUTO_ENV_VAR) == expected


def test_the_wheels_ptxas_is_found_and_answers():
    """Discovery against this machine. Only the wheel is asserted on: a
    system ptxas that cannot run (a broken CUDA install, a foreign
    architecture) is a candidate we are meant to survive, not a test
    failure -- it comes back with no version and is rejected by name."""
    wheel = _ptxas.wheel_ptxas()
    assert wheel is not None, (
        "the nvidia-cuda-nvcc-cu12 wheel is a Linux dev dependency"
    )
    found = {p.path: p for p in _ptxas.candidates()}
    assert wheel in found
    assert found[wheel].version is not None
