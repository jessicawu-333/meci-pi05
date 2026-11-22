"""MECI for pi0.5 -- framework-agnostic (NumPy) metric-correction utilities.

Spoken mid-execution corrections are *metric end-effector nudges* that may name the effector:
    "left hand more left half an inch", "right gripper up 1 cm", "move forward 1 inch",
    "rotate gripper 10 degrees".
They are parsed into (effector e, direction, magnitude) -> a 6-D Cartesian displacement
delta_p = [dx, dy, dz, droll, dpitch, dyaw] in metres / radians in a declared frame, and then
mapped into openpi's *normalized* action space for the platform's action layout.

Everything here is pure NumPy so the same code serves the JAX model (meci_correction.py), the
PyTorch model (meci_correction_pytorch.py), the policy server and the data pipeline.

Action layouts and normalisation verified in Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479:
  * ALOHA  (pi05_aloha, src/openpi/training/config.py l.572-579; LeRobotAlohaDataConfig l.228-279;
            src/openpi/policies/aloha_policy.py):
      14-D = [left 6 joints, left gripper, right 6 joints, right gripper]   (aloha_policy.py l.160-161)
      joints are deltas relative to the state at the start of the chunk:
      DeltaActions(make_bool_mask(6, -1, 6, -1))                            (config.py l.263-267)
      pi-space joints = _joint_flip_mask() * aloha joints                   (aloha_policy.py l.104-106, l.190-202)
  * LIBERO (pi05_libero, config.py l.743-763; LeRobotLiberoDataConfig extra_delta_transform=False):
      7-D = [EE dpos(3), EE drot(3), gripper]  (libero_policy.py l.100 LiberoOutputs returns [..., :7]);
      per-step OSC_POSE deltas in [-1, 1] (LIBERO env_wrapper.py l.17 controller="OSC_POSE"); robosuite
      v1.4.0/v1.4.1 controllers/config/osc_pose.json l.5 output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
      (m, rad) per unit action -- this is the commanded set-point delta; the executed motion depends on the
      impedance gains and must be calibrated (open decision).
  * DROID  (pi05_droid, config.py l.629-642): 8-D = [7 joints, gripper] (droid_policy.py DroidOutputs);
      no DeltaActions in the inference config; pi05_full_droid_finetune / pi05_droid_finetune use
      DroidActionSpace.JOINT_POSITION + DeltaActions(make_bool_mask(7, -1)) (config.py l.403-408).
  * Normalisation: pi0.5 uses *quantile* normalisation (config.py l.187:
      use_quantile_norm = model_config.model_type != ModelType.PI0), transforms.py l.141-145 / l.175-181:
        x_n = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1 ;  dims beyond len(q01) pass through on unnormalize.
      pi0 (not pi0.5) uses z-score: x_n = (x - mean) / (std + 1e-6)  (transforms.py l.137-139).
      Both are affine per dimension, so an *additive* residual maps exactly:
        quantile: dn = 2 * dx / (q99 - q01 + 1e-6) ;  z-score: dn = dx / (std + 1e-6).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import math
import re

import numpy as np

# ----------------------------------------------------------------------------------------------
# 1. Units, directions, effectors
# ----------------------------------------------------------------------------------------------
INCH_TO_M = 0.0254
CM_TO_M = 0.01
MM_TO_M = 0.001
DEG_TO_RAD = math.pi / 180.0

# Declared spoken-language frame (robot base). Axis signs are PLACEHOLDERS to be calibrated per
# platform by commanding a known residual and observing the EE motion (see DESIGN.md, open decisions).
CORRECTION_FRAME = "robot_base"
DIRECTION_AXES: dict[str, tuple[int, int]] = {
    "forward": (0, +1),
    "back": (0, -1),
    "left": (1, +1),
    "right": (1, -1),
    "up": (2, +1),
    "down": (2, -1),
}
ROTATION_AXES: dict[str, tuple[int, int]] = {"roll": (3, +1), "pitch": (4, +1), "yaw": (5, +1)}
ROTATION_ALIASES = {"gripper": "yaw", "wrist": "yaw", "end effector": "yaw", "end-effector": "yaw"}

EFFECTORS = ("default", "left", "right")
EFFECTOR_TO_ID = {e: i for i, e in enumerate(EFFECTORS)}
EFFECTOR_ALIASES: tuple[tuple[str, str], ...] = (
    ("left hand", "left"),
    ("left arm", "left"),
    ("left gripper", "left"),
    ("right hand", "right"),
    ("right arm", "right"),
    ("right gripper", "right"),
)

_WORD_MAGNITUDES = {
    "a quarter": 0.25,
    "quarter": 0.25,
    "a half": 0.5,
    "half": 0.5,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "ten": 10.0,
}
_UNIT_ALIASES = {
    "in": "inch",
    "inch": "inch",
    "inches": "inch",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "deg": "deg",
    "degs": "deg",
    "degree": "deg",
    "degrees": "deg",
    "rad": "rad",
    "radian": "rad",
    "radians": "rad",
}
_UNIT_RE = (
    r"(?P<unit>inch(?:es)?|in\b|centimet(?:er|re)s?|cm\b|millimet(?:er|re)s?|mm\b|met(?:er|re)s?|m\b|"
    r"degrees?|degs?\b|radians?|rad\b)"
)


@dataclasses.dataclass
class MetricCommand:
    """A parsed metric nudge."""

    kind: str  # "translate" | "rotate"
    direction: str  # forward/back/left/right/up/down or roll/pitch/yaw
    magnitude: float  # in `unit`
    unit: str  # canonical unit
    axis: int  # index into the 6-D Cartesian delta
    sign: int
    effector: str = "default"  # "default" | "left" | "right"
    phrase: str = ""

    @property
    def effector_id(self) -> int:
        return EFFECTOR_TO_ID[self.effector]

    def delta_p(self) -> np.ndarray:
        """6-D Cartesian displacement [dx, dy, dz, droll, dpitch, dyaw] in metres / radians."""
        d = np.zeros(6, dtype=np.float32)
        d[self.axis] = self.sign * magnitude_to_si(self.magnitude, self.unit, self.kind)
        return d


def magnitude_to_si(magnitude: float, unit: str, kind: str) -> float:
    u = _UNIT_ALIASES.get(unit.lower().strip(), unit.lower().strip())
    if kind == "rotate":
        if u == "deg":
            return magnitude * DEG_TO_RAD
        if u == "rad":
            return magnitude
        raise ValueError(f"unsupported rotation unit {unit!r}")
    table = {"inch": INCH_TO_M, "cm": CM_TO_M, "mm": MM_TO_M, "m": 1.0}
    if u not in table:
        raise ValueError(f"unsupported translation unit {unit!r}")
    return magnitude * table[u]


def parse_metric_command(phrase: str) -> MetricCommand | None:
    """Rule-based parser (the ASR -> parser baseline) for templated metric nudges.

    The effector phrase ("left hand", "right gripper", ...) is matched and removed *first*, so in
    "left hand more left half an inch" the first "left" is the effector and the second the direction.
    Returns None if the phrase is not a metric nudge.
    """
    text = " ".join(phrase.lower().replace(",", " ").strip().split())
    if not text:
        return None
    effector = "default"
    for alias, canon in EFFECTOR_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", text):
            effector = canon
            text = " ".join(re.sub(rf"\b{re.escape(alias)}\b", " ", text, count=1).split())
            break

    # magnitude + unit
    mag = unit = None
    for word, val in sorted(_WORD_MAGNITUDES.items(), key=lambda kv: -len(kv[0])):
        m = re.search(rf"\b{re.escape(word)}\b\s+(?:an?\s+)?{_UNIT_RE}", text)
        if m:
            mag, unit = val, m.group("unit")
            break
    if mag is None:
        m = re.search(rf"(?P<mag>\d+(?:\.\d+)?)\s*{_UNIT_RE}", text)
        if m:
            mag, unit = float(m.group("mag")), m.group("unit")
    if mag is None:
        return None
    unit = _UNIT_ALIASES.get(unit.strip(), unit.strip())

    if unit in ("deg", "rad") or re.search(r"\brotate\b|\bturn\b|\btwist\b", text):
        kind = "rotate"
        direction = None
        for alias, canon in ROTATION_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", text):
                direction = canon
                break
        for name in ROTATION_AXES:
            if re.search(rf"\b{name}\b", text):
                direction = name
                break
        direction = direction or "yaw"
        axis, sign = ROTATION_AXES[direction]
        if re.search(r"\b(counter[- ]?clockwise|anticlockwise)\b", text):
            sign = +1
        elif re.search(r"\bclockwise\b", text):
            sign = -1
    else:
        kind = "translate"
        direction = None
        for name in ("forward", "backward", "back", "left", "right", "up", "down"):
            if re.search(rf"\b{name}\b", text):
                direction = "back" if name == "backward" else name
                break
        if direction is None:
            return None
        axis, sign = DIRECTION_AXES[direction]
    return MetricCommand(kind, direction, float(mag), unit, axis, sign, effector=effector, phrase=phrase)


# ----------------------------------------------------------------------------------------------
# 2. Platform action layouts (verified in openpi @ 215abfb, see module docstring)
# ----------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ArmSlice:
    start: int  # first action index of this arm
    num_joints: int  # 6 (ALOHA), 7 (DROID); for Cartesian layouts: 6 = [dpos, drot]
    gripper_index: int


@dataclasses.dataclass(frozen=True)
class PlatformLayout:
    name: str
    action_dim: int  # dims used by the robot (model pads to 32)
    arms: dict[str, ArmSlice]  # effector name -> slice ("default" always present)
    space: str  # "cartesian" (EE deltas) or "joint"
    # "per_step": each action is an increment (LIBERO OSC deltas, joint velocities): the residual is
    #             spread so that SUM_t r_t = delta.
    # "chunk_relative": actions are targets relative to the chunk-start state (openpi DeltaActions):
    #             the residual is a ramp so that r_t = cumulative share * delta (final step = delta).
    semantics: str
    # Physical units per unit of *raw* (un-normalized) action, per Cartesian dim (cartesian space only).
    # LIBERO: robosuite OSC_POSE output_max (m, rad) per unit input.
    cartesian_scale: tuple[float, ...] | None = None
    # Per-joint sign flip between the robot convention (where the Jacobian is defined) and openpi's
    # model space (ALOHA adapt_to_pi). Length action_dim; None = identity.
    joint_flip: tuple[float, ...] | None = None
    # Control period in seconds (only for velocity semantics: residual / dt).
    velocity_dt: float | None = None


ALOHA = PlatformLayout(
    name="aloha",
    action_dim=14,
    arms={
        "left": ArmSlice(0, 6, 6),
        "right": ArmSlice(7, 6, 13),
        # Unnamed correction on a bimanual robot: do not guess an arm (residual is zero, flagged).
    },
    space="joint",
    semantics="chunk_relative",
    joint_flip=(1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1),  # aloha_policy._joint_flip_mask()
)
LIBERO = PlatformLayout(
    name="libero",
    action_dim=7,
    arms={"default": ArmSlice(0, 6, 6)},
    space="cartesian",
    semantics="per_step",
    cartesian_scale=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
)
DROID_JOINT_POSITION = PlatformLayout(
    name="droid_joint_position",
    action_dim=8,
    arms={"default": ArmSlice(0, 7, 7)},
    space="joint",
    semantics="chunk_relative",
)
DROID_JOINT_VELOCITY = PlatformLayout(
    name="droid_joint_velocity",
    action_dim=8,
    arms={"default": ArmSlice(0, 7, 7)},
    space="joint",
    semantics="per_step",
    # openpi examples/droid/main.py: DROID_CONTROL_FREQUENCY = 15 (l.23), RobotEnv(action_space="joint_velocity")
    # (l.80). Whether DROID's joint_velocity actions are in rad/s or a scaled unit is NOT verified here, so the
    # caller must set velocity_dt (and any scale) explicitly, e.g. dataclasses.replace(..., velocity_dt=1 / 15).
    velocity_dt=None,
)
PLATFORMS = {p.name: p for p in (ALOHA, LIBERO, DROID_JOINT_POSITION, DROID_JOINT_VELOCITY)}


def resolve_arm(layout: PlatformLayout, effector: str) -> ArmSlice | None:
    """Named-effector routing. Single-arm platforms ignore the name; bimanual + unnamed -> None."""
    if effector in layout.arms:
        return layout.arms[effector]
    if "default" in layout.arms and len(layout.arms) == 1:
        return layout.arms["default"]
    return None


# ----------------------------------------------------------------------------------------------
# 3. Cartesian -> joint (first order), for joint-space platforms (ALOHA, DROID)
# ----------------------------------------------------------------------------------------------
def dls_joint_delta(jacobian: np.ndarray, delta_p: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    """Damped-least-squares inverse kinematics step: dq = J^T (J J^T + lambda^2 I)^-1 dp.

    `jacobian` is the (6, n) geometric Jacobian of the named arm at the current joint state, in the
    same frame as `delta_p`. openpi does not ship arm kinematics; the caller supplies J (e.g. from the
    ViperX-300s URDF for ALOHA, Franka for DROID). First-order: exact for small nudges only.
    """
    jac = np.asarray(jacobian, dtype=np.float64)
    dp = np.asarray(delta_p, dtype=np.float64).reshape(-1)
    jjt = jac @ jac.T
    return (jac.T @ np.linalg.solve(jjt + (damping**2) * np.eye(jjt.shape[0]), dp)).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# 4. Physical residual over the hold window, routed to the named effector
# ----------------------------------------------------------------------------------------------
def hold_fade_strengths(hold_steps: int = 16, fade_steps: int = 8) -> np.ndarray:
    """Per-control-step strengths of a correction: 1 during the hold, linear fade to 0."""
    s = np.ones(hold_steps, dtype=np.float32)
    if fade_steps > 0:
        f = min(fade_steps, hold_steps)
        s[-f:] = np.arange(f, 0, -1, dtype=np.float32) / float(f)
    return s


def raw_action_delta(
    layout: PlatformLayout,
    delta_p: np.ndarray,
    effector: str,
    jacobian_fn: Callable[[str], np.ndarray] | None = None,
) -> tuple[np.ndarray, bool]:
    """Total raw-action-space displacement (length layout.action_dim) for the named effector.

    Returns (delta_a, routed). routed=False when the effector cannot be resolved (bimanual + unnamed),
    in which case delta_a is all zeros: the residual is withheld and the learned path decides.
    Gripper dims are never touched by metric nudges.
    """
    out = np.zeros(layout.action_dim, dtype=np.float32)
    arm = resolve_arm(layout, effector)
    if arm is None:
        return out, False
    dp = np.asarray(delta_p, dtype=np.float32).reshape(6)
    if layout.space == "cartesian":
        scale = np.asarray(layout.cartesian_scale, dtype=np.float32)
        out[arm.start : arm.start + 6] = dp / scale
    else:
        if jacobian_fn is None:
            raise ValueError(f"{layout.name}: joint-space platform needs a jacobian_fn(effector) -> (6, n_joints)")
        key = effector if effector in layout.arms else "default"
        dq = dls_joint_delta(jacobian_fn(key), dp)
        if dq.shape[0] != arm.num_joints:
            raise ValueError(f"Jacobian has {dq.shape[0]} joints, layout expects {arm.num_joints}")
        out[arm.start : arm.start + arm.num_joints] = dq  # velocity layouts: divided by dt in chunk_residual
        if layout.joint_flip is not None:
            out = out * np.asarray(layout.joint_flip, dtype=np.float32)
    return out, True


def chunk_residual(
    layout: PlatformLayout,
    delta_a: np.ndarray,
    window_strengths: np.ndarray,
    chunk_start: int,
    action_horizon: int,
) -> np.ndarray:
    """Raw-space residual for one action chunk of length `action_horizon` that starts `chunk_start`
    control steps after the utterance. Over the whole hold window:
      per_step:        SUM over the window of r_t            == delta_a
      chunk_relative:  r at the last window step of a chunk  == remaining share of delta_a
                       (each new chunk is re-based on the state, which already contains the motion done)
    Returns (action_horizon, action_dim).
    """
    w = np.asarray(window_strengths, dtype=np.float64)
    total = max(w.sum(), 1e-12)
    n_win = len(w)
    steps = np.arange(chunk_start, chunk_start + action_horizon)
    share = np.where(steps < n_win, w[np.clip(steps, 0, n_win - 1)], 0.0) / total  # per-step share
    if layout.semantics == "per_step":
        coeff = share
        if layout.velocity_dt is not None:
            coeff = coeff / layout.velocity_dt
    elif layout.semantics == "chunk_relative":
        coeff = np.cumsum(share)  # ramp, relative to the chunk-start state
    else:
        raise ValueError(layout.semantics)
    return (coeff[:, None] * np.asarray(delta_a, dtype=np.float64)[None, :]).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# 5. Map into openpi's normalized action space (pi0.5: quantile; pi0: z-score)
# ----------------------------------------------------------------------------------------------
def normalize_residual(
    residual: np.ndarray,
    *,
    model_action_dim: int = 32,
    q01: np.ndarray | None = None,
    q99: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    use_quantiles: bool = True,
) -> np.ndarray:
    """Additive raw residual -> additive residual in openpi's normalized, zero-padded model space.

    Exact because openpi's Normalize/Unnormalize are per-dimension affine maps
    (transforms.py l.137-145, l.170-181). Dims beyond the stats length pass through unchanged on
    Unnormalize, so they map 1:1 (they are zero for metric nudges anyway).
    """
    r = np.asarray(residual, dtype=np.float64)
    out = np.zeros((*r.shape[:-1], model_action_dim), dtype=np.float64)
    out[..., : r.shape[-1]] = r
    if use_quantiles:
        q01 = np.asarray(q01, dtype=np.float64)
        q99 = np.asarray(q99, dtype=np.float64)
        d = min(q01.shape[-1], model_action_dim)
        out[..., :d] = out[..., :d] * 2.0 / (q99[:d] - q01[:d] + 1e-6)
    else:
        std = np.asarray(std, dtype=np.float64)
        d = min(std.shape[-1], model_action_dim)
        out[..., :d] = out[..., :d] / (std[:d] + 1e-6)
    return out.astype(np.float32)


def unnormalize_quantile(x_n: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Copy of openpi transforms.Unnormalize._unnormalize_quantile (transforms.py l.175-181) for tests."""
    q01 = np.asarray(q01)
    q99 = np.asarray(q99)
    dim = q01.shape[-1]
    if dim < x_n.shape[-1]:
        return np.concatenate([(x_n[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x_n[..., dim:]], axis=-1)
    return (x_n + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def build_correction_inputs(
    phrase_or_cmd: str | MetricCommand,
    layout: PlatformLayout,
    *,
    action_horizon: int,
    chunk_start: int = 0,
    window_strengths: np.ndarray | None = None,
    norm_stats_actions=None,
    model_action_dim: int = 32,
    jacobian_fn: Callable[[str], np.ndarray] | None = None,
) -> dict | None:
    """Policy-side helper: phrase -> the `correction_*` keys consumed by the patched openpi model.

    `norm_stats_actions` is openpi's NormStats for the "actions" key (has .q01/.q99 for pi0.5).
    Audio features (`correction_audio`, `correction_audio_mask`) are added separately by the frozen
    Whisper encoder.
    """
    cmd = parse_metric_command(phrase_or_cmd) if isinstance(phrase_or_cmd, str) else phrase_or_cmd
    if cmd is None:
        return None
    w = hold_fade_strengths() if window_strengths is None else np.asarray(window_strengths, np.float32)
    dp = cmd.delta_p()
    delta_a, routed = raw_action_delta(layout, dp, cmd.effector, jacobian_fn)
    r = chunk_residual(layout, delta_a, w, chunk_start, action_horizon)
    rn = normalize_residual(
        r, model_action_dim=model_action_dim, q01=norm_stats_actions.q01, q99=norm_stats_actions.q99
    )
    step = min(chunk_start, len(w) - 1)
    strength = float(w[step]) if chunk_start < len(w) else 0.0
    return {
        "correction_active": np.bool_(strength > 0.0),
        "correction_strength": np.float32(strength),
        "correction_delta": dp.astype(np.float32),
        "correction_effector": np.int32(cmd.effector_id),
        "correction_residual": rn if routed else np.zeros_like(rn),
    }


def make_correction_jacobian_fn(jacobians: dict[str, np.ndarray]) -> Callable[[str], np.ndarray]:
    """Convenience: fixed per-arm Jacobians (e.g., computed once at the utterance from the state)."""
    return lambda effector: jacobians[effector]


__all__: Sequence[str] = [n for n in dir() if not n.startswith("_")]
