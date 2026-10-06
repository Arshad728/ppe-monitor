"""Restricted zones: the point-in-polygon test, where a person stands, and the zones file."""

import numpy as np
import pytest

from ppe_monitor.config import ConfigError
from ppe_monitor.rules.zones import (CameraZones, Zone, feet_point, load_zones, point_in_polygon, polygon_problem,
                                     save_zones)
from ppe_monitor.vision.pose import Pose

SQUARE = ((0.2, 0.4), (0.6, 0.4), (0.6, 0.9), (0.2, 0.9))
# an L shape: the notch at the top right is outside
ELL = ((0.1, 0.1), (0.4, 0.1), (0.4, 0.5), (0.8, 0.5), (0.8, 0.9), (0.1, 0.9))


def test_the_books_example():
    assert point_in_polygon((0.35, 0.6), SQUARE)


@pytest.mark.parametrize("point, inside", [((0.3, 0.5), True), ((0.1, 0.5), False), ((0.7, 0.5), False),
                                           ((0.3, 0.3), False), ((0.3, 0.95), False)])
def test_square(point, inside):
    assert point_in_polygon(point, SQUARE) is inside


@pytest.mark.parametrize("point, inside", [((0.2, 0.3), True), ((0.6, 0.3), False), ((0.6, 0.7), True),
                                           ((0.3, 0.7), True), ((0.9, 0.7), False)])
def test_concave_polygon(point, inside):
    assert point_in_polygon(point, ELL) is inside


def test_a_ray_through_a_corner_is_counted_once():
    diamond = ((0.5, 0.1), (0.9, 0.5), (0.5, 0.9), (0.1, 0.5))
    assert point_in_polygon((0.3, 0.5), diamond)          # the ray passes exactly through the corner (0.9, 0.5)
    assert not point_in_polygon((0.05, 0.5), diamond)


def test_polygon_checks():
    assert polygon_problem(SQUARE) is None
    assert polygon_problem(ELL) is None
    assert "3 corners" in polygon_problem(((0.1, 0.1), (0.2, 0.2)))
    assert "pixels" in polygon_problem(((10, 20), (300, 20), (300, 200)))
    assert "cross" in polygon_problem(((0.1, 0.1), (0.5, 0.5), (0.5, 0.1), (0.1, 0.5)))   # a bow tie
    assert "area" in polygon_problem(((0.1, 0.1), (0.5, 0.5), (0.9, 0.9)))


def test_feet_are_the_bottom_centre_of_the_box():
    assert feet_point((0.2, 0.3, 0.4, 0.8)) == pytest.approx((0.3, 0.8))


def test_feet_cut_off_by_the_frame_edge_are_unknown():
    assert feet_point((0.2, 0.5, 0.4, 1.0)) is None


def test_visible_ankles_beat_the_box():
    xy, conf = np.zeros((17, 2)), np.zeros(17)
    xy[15], xy[16], conf[15], conf[16] = (0.30, 0.95), (0.34, 0.97), 0.9, 0.9
    pose = Pose((0.2, 0.3, 0.6, 1.0), xy, conf, aspect=16 / 9)   # box reaches the bottom edge, ankles visible
    assert feet_point(pose.box, pose) == pytest.approx((0.32, 0.96))
    conf[:] = 0
    assert feet_point(pose.box, pose) is None


def test_zones_file_round_trip(tmp_path):
    zones = {"cam1": CameraZones("cam1", [Zone("pit", "Excavation pit", SQUARE), Zone("bay", "Bay", ELL)],
                                 "configs/zones/cam1.jpg", (1280, 720))}
    path = tmp_path / "zones.yaml"
    save_zones(zones, path)
    assert load_zones(path) == zones
    assert "polygon: [[0.2, 0.4]" in path.read_text()


def test_no_zones_file_means_no_zones(tmp_path):
    assert load_zones(tmp_path / "missing.yaml") == {}


@pytest.mark.parametrize("text, message", [
    ("cameras:\n  cam1:\n    zones:\n      - {id: pit, polygon: [[10, 20], [300, 20], [300, 200]]}\n", "pixels"),
    ("cameras:\n  cam1:\n    zones:\n      - {id: Pit!, polygon: [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]}\n", "lowercase"),
    ("cameras:\n  cam1:\n    zones:\n      - {id: a, polygon: [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]}\n"
     "      - {id: a, polygon: [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]}\n", "twice"),
])
def test_bad_zones_are_refused_with_a_clear_message(tmp_path, text, message):
    path = tmp_path / "zones.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=message):
        load_zones(path)


def test_the_zone_editor_turns_clicks_into_a_zone():
    from ppe_monitor.rules.zone_editor import ZoneEditor
    ed = ZoneEditor("gate", 1280, 720)
    for x, y in ((128, 360), (640, 360), (640, 648), (999, 999)):   # the last click is off the frame
        ed.add(x, y)
    ed.undo()
    with pytest.raises(ValueError, match="lowercase"):
        ed.finish("Pit Area")
    zone = ed.finish("pit", "Excavation pit")
    assert zone.polygon == ((0.1, 0.5), (0.5, 0.5), (0.5, 0.9)) and ed.corners == [] and ed.changed
    for x, y in ((100, 100), (300, 300), (300, 100), (100, 300)):   # a bow tie
        ed.add(x, y)
    with pytest.raises(ValueError, match="cross"):
        ed.finish("tie")
    ed.corners.clear()
    with pytest.raises(ValueError, match="already"):
        ed.add(1, 1), ed.add(50, 1), ed.add(50, 50)
        ed.finish("pit")
    assert ed.delete("pit") and ed.result("configs/zones/gate.jpg").zones == []
    assert ed.render(np.zeros((720, 1280, 3), np.uint8)).shape == (720, 1280, 3)
