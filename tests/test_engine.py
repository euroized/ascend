"""The ASCEND engine against the Brian2 reference, and its CUDA kernels against the CPU engine."""

import numpy as np
import pytest

from ascend import synthetic
from ascend.engine import CPUEngine, EventInput, Network, Params, PoissonInput, rng
from ascend.engine import emulate as emu
from ascend.reference import brian2_ref

P = Params.reference()


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    """Dense enough that neurons receive synchronous input and hit refractoriness often."""
    return synthetic.write(tmp_path_factory.mktemp("dense"), n_neurons=500, n_connections=40000, seed=3)


def kick_events(n_steps, n_stim, seed=0, rate_hz=150.0):
    draw = np.random.default_rng(seed).random((n_steps, n_stim)) < rate_hz * P.dt
    return np.nonzero(draw)


def assert_same(a: dict, b: dict):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(np.rint(a[k] / P.dt), np.rint(b[k] / P.dt), err_msg=f"neuron {k}")


def test_params_come_from_reference():
    assert (P.refractory_steps, P.delay_steps) == (22, 18)
    assert P.w_poi == pytest.approx(0.06875) and P.v_th == pytest.approx(-0.045)


@pytest.mark.parametrize("fixture", ["tiny_connectome", "dense"])
def test_cpu_engine_matches_brian2_spike_for_spike(fixture, request):
    pc, pn = request.getfixturevalue(fixture)
    exc = list(range(0, 200, 13))
    steps, pos = kick_events(2000, len(exc))
    eng = CPUEngine(Network.from_files(pc, pn, P.w_syn), P, exc)
    ours = eng.run(2000, EventInput(steps, pos)).trains()
    ref = brian2_ref.run_events(pc, pn, exc, steps, pos, n_steps=2000, target="numpy")
    assert sum(len(v) for v in ref.values()) > 500
    assert_same(ours, ref)


def test_run_in_pieces_equals_one_run(tiny_connectome):
    net = Network.from_files(*tiny_connectome, P.w_syn)
    a, b = CPUEngine(net, P, range(10)), CPUEngine(net, P, range(10))
    whole = a.run(1000, PoissonInput(150, 4))
    parts = [b.run(n, PoissonInput(150, 4)) for n in (300, 700)]
    np.testing.assert_array_equal(whole.steps, np.concatenate([p.steps for p in parts]))


def test_no_stimulus_is_silent(tiny_connectome):
    eng = CPUEngine(Network.from_files(*tiny_connectome, P.w_syn), P, [])
    assert len(eng.run(500)) == 0


def test_kick_rng_rate_and_determinism():
    neurons = np.arange(50)
    thr = rng.threshold(150, P.dt)
    m = np.array([rng.kicks(1, s, neurons, thr) for s in range(20000)])
    assert m.mean() == pytest.approx(0.015, abs=0.0006)
    np.testing.assert_array_equal(m[:100], [rng.kicks(1, s, neurons, thr) for s in range(100)])
    assert not np.array_equal(m[:100], [rng.kicks(2, s, neurons, thr) for s in range(100)])


def test_out_synapses_follow_row_order():
    net = Network.from_arrays(4, np.array([2, 0, 2, 1, 0]), np.array([1, 3, 0, 2, 1]), np.arange(1, 6), 1.0)
    np.testing.assert_array_equal(net.post[net.out_synapses(np.array([0, 2]))], [3, 1, 1, 0])


needs_cxx = pytest.mark.skipif(emu.compiler() is None, reason="no C++ compiler")


@needs_cxx
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_cuda_kernels_match_cpu_engine(dense, dtype):
    """kernels.cu, run thread by thread on the CPU, gives the CPU engine's spikes exactly."""
    net = Network.from_files(*dense, P.w_syn)
    exc = list(range(0, 500, 20))
    cpu = CPUEngine(net, P, exc, dtype=dtype).run(1500, PoissonInput(150, 9))
    k = emu.EmulatedEngine(net, P, exc, dtype=dtype)
    k.chunk_steps = 400  # several chunks, including a partial one
    kern = k.run(1500, PoissonInput(150, 9))
    assert len(cpu) > 1000
    np.testing.assert_array_equal(cpu.steps, kern.steps)
    np.testing.assert_array_equal(cpu.neurons, kern.neurons)


@needs_cxx
def test_kernel_engine_reset_and_record_overflow(tiny_connectome):
    net = Network.from_files(*tiny_connectome, P.w_syn)
    k = emu.EmulatedEngine(net, P, range(10))
    first = k.run(800, PoissonInput(150, 1))
    k.reset()
    np.testing.assert_array_equal(first.steps, k.run(800, PoissonInput(150, 1)).steps)
    small = emu.EmulatedEngine(net, P, range(10), rec_cap=5)
    with pytest.raises(RuntimeError, match="overflow"):
        small.run(800, PoissonInput(150, 1))


def test_kernels_compile_with_nvrtc():
    """The GPU build of kernels.cu compiles for Pascal through Ada, in float and double (no GPU needed)."""
    nvrtc = pytest.importorskip("cupy_backends.cuda.libs.nvrtc")
    from ascend.engine.kernel_engine import KERNELS

    try:
        nvrtc.getVersion()
    except Exception as e:  # library missing
        pytest.skip(f"NVRTC unavailable: {e}")
    for real in ("float", "double"):
        for arch in ("compute_60", "compute_75", "compute_86", "compute_89"):
            prog = nvrtc.createProgram(KERNELS.read_text(), "kernels.cu", [], [])
            nvrtc.compileProgram(prog, ["--std=c++14", "--fmad=false", f"-DREAL={real}", f"-arch={arch}"])
            ptx = nvrtc.getPTX(prog)
            ptx = ptx.decode() if isinstance(ptx, bytes | bytearray) else ptx
            for kernel in ("update", "deliver", "clear", "advance"):
                assert f".entry {kernel}(" in ptx
