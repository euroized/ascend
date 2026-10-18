"""Model constants for the ASCEND engine, taken from the reference model so the two cannot drift.

All values are SI floats (volts, seconds). The integration constants reproduce, operation for
operation, the scalar code Brian2 generates for the reference equations with method='linear'
(see ascend/reference/shiu_model.py):

    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau              : volt (unless refractory)

so that the float64 CPU engine can match Brian2 bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT = 1e-4  # Brian2's default clock: 0.1 ms


def _timestep(t: float, dt: float) -> int:
    """Brian2's timestep(): shift by 1e-3 dt before flooring, so multiples of dt land on their own step."""
    return int(np.array((t + 1e-3 * dt) / dt, dtype=np.int64))


@dataclass(frozen=True)
class Params:
    dt: float
    v_0: float
    v_rst: float
    v_th: float
    t_mbr: float
    tau: float
    t_rfc: float
    t_dly: float
    w_syn: float
    w_poi: float

    @classmethod
    def reference(cls, dt: float = DT) -> Params:
        """The reference model's default_params, as floats."""
        import brian2 as b2  # noqa: PLC0415

        from ..reference import shiu_model  # noqa: PLC0415

        p = shiu_model.default_params
        # Brian2 renders the Poisson weight into generated code as a literal in mV, e.g. "68.75 * mvolt".
        w_poi = float(float(p["w_syn"] * p["f_poi"] / b2.mV) * b2.mvolt)
        return cls(
            dt=dt,
            v_0=float(p["v_0"]),
            v_rst=float(p["v_rst"]),
            v_th=float(p["v_th"]),
            t_mbr=float(p["t_mbr"]),
            tau=float(p["tau"]),
            t_rfc=float(p["t_rfc"]),
            t_dly=float(p["t_dly"]),
            w_syn=float(p["w_syn"]),
            w_poi=w_poi,
        )

    @property
    def refractory_steps(self) -> int:
        return _timestep(self.t_rfc, self.dt)

    @property
    def delay_steps(self) -> int:
        return int(np.round(self.t_dly / self.dt))

    def integration(self) -> tuple[float, float, float, float]:
        """(decay_g, offset_v, g_to_v, decay_v): over one step, g' = decay_g * g and
        v' = offset_v + (g_to_v * g + decay_v * v). Same operations as Brian2's scalar code."""
        dt, tau, t_mbr, v_0 = (np.float64(x) for x in (self.dt, self.tau, self.t_mbr, self.v_0))
        lio_1 = (-dt) / tau
        lio_3 = tau / (t_mbr - tau)
        lio_4 = dt / t_mbr
        lio_5 = dt / tau
        lio_6 = (-dt) / t_mbr
        decay_g = np.exp(lio_1)
        offset_v = v_0 - (v_0 * np.exp(lio_6))
        g_to_v = ((lio_3 * ((-np.exp(lio_4)) + np.exp(lio_5))) * np.exp(lio_6)) * np.exp(lio_1)
        decay_v = np.exp(lio_6)
        return float(decay_g), float(offset_v), float(g_to_v), float(decay_v)
