"""Sim-speed helpers: physics timing tags are rewritten, source world untouched."""
import re

import pytest

from tb3_slam.utils import (
    apply_sim_speed,
    physics_for_speed,
    SIM_SPEED_ENV,
    sim_speed_from_env,
    SIM_STEP_MAX,
    speed_up_world,
)

SDF = """<sdf><world name="default"><physics type="ode">
  <real_time_update_rate>1000.0</real_time_update_rate>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
</physics></world></sdf>"""


def _tag(text: str, tag: str) -> float:
    return float(re.search(rf'<{tag}>([^<]*)</{tag}>', text).group(1))


@pytest.mark.parametrize('speed, step, rtf', [
    (1.0, 0.001, 1.0),
    (2.0, 0.002, 2.0),
    (4.0, 0.004, 4.0),
    (8.0, SIM_STEP_MAX, 8.0),   # step is capped, rate carries the rest
])
def test_physics_for_speed(speed, step, rtf):
    got_step, rate, got_rtf = physics_for_speed(speed)
    assert got_step == pytest.approx(step)
    assert got_rtf == pytest.approx(rtf)
    assert got_step * rate == pytest.approx(rtf)   # rate consistent with RTF


def test_unlimited_speed_is_rtf_zero():
    step, rate, rtf = physics_for_speed(0)
    assert (step, rate, rtf) == (SIM_STEP_MAX, 0.0, 0.0)


def test_negative_speed_rejected():
    with pytest.raises(ValueError):
        physics_for_speed(-1)


def test_apply_sim_speed_rewrites_all_tags():
    out = apply_sim_speed(SDF, 4.0)
    assert _tag(out, 'max_step_size') == pytest.approx(0.004)
    assert _tag(out, 'real_time_factor') == pytest.approx(4.0)
    assert _tag(out, 'real_time_update_rate') == pytest.approx(1000.0)
    assert out.count('<physics') == 1


def test_apply_sim_speed_requires_physics_tags():
    with pytest.raises(ValueError):
        apply_sim_speed('<sdf><world/></sdf>', 2.0)


def test_speed_up_world_is_identity_at_realtime(tmp_path):
    world = tmp_path / 'w.sdf'
    world.write_text(SDF)
    assert speed_up_world(str(world), 1.0, str(tmp_path / 'out.sdf')) == str(world)


def test_speed_up_world_writes_copy_and_keeps_source(tmp_path):
    world = tmp_path / 'w.sdf'
    world.write_text(SDF)
    out = tmp_path / 'out.sdf'
    assert speed_up_world(str(world), 4.0, str(out)) == str(out)
    assert _tag(out.read_text(), 'real_time_factor') == pytest.approx(4.0)
    assert world.read_text() == SDF


def test_sim_speed_from_env(monkeypatch):
    monkeypatch.delenv(SIM_SPEED_ENV, raising=False)
    assert sim_speed_from_env() == 1.0
    monkeypatch.setenv(SIM_SPEED_ENV, '4')
    assert sim_speed_from_env() == 4.0
