import numpy as np

from motion.kalman import TargetTrack, TrackConfig


def test_converges_to_static_target():
    trk = TargetTrack()
    truth = np.array([0.3, 0.1, 0.35])
    rng = np.random.default_rng(0)
    for _ in range(100):
        trk.predict(0.01)
        trk.update_point(truth + rng.normal(0, 0.03, 3))
    assert np.linalg.norm(trk.position - truth) < 0.02
    assert trk.confidence > 0.7


def test_estimates_velocity_of_moving_target():
    trk = TargetTrack()
    rng = np.random.default_rng(1)
    vel = np.array([0.2, 0.0, -0.1])
    pos = np.array([0.0, 0.1, 0.4])
    for _ in range(200):
        pos = pos + vel * 0.01
        trk.predict(0.01)
        trk.update_point(pos + rng.normal(0, 0.02, 3))
    assert np.linalg.norm(trk.velocity - vel) < 0.07
    lead = trk.predicted_position(0.1)
    assert np.linalg.norm(lead - (pos + vel * 0.1)) < 0.03


def test_coasts_then_drops_on_measurement_gap():
    trk = TargetTrack(TrackConfig(max_coast=0.5))
    for _ in range(50):
        trk.predict(0.01)
        trk.update_point(np.array([0.2, 0.0, 0.3]))
    assert trk.active
    c0 = trk.confidence
    for _ in range(30):  # 0.3 s gap - still coasting, less confident
        trk.predict(0.01)
    assert trk.active and trk.confidence < c0
    for _ in range(40):  # now past max_coast
        trk.predict(0.01)
    assert not trk.active
    assert trk.confidence == 0.0


def test_bearing_update_places_target_on_ray():
    trk = TargetTrack(TrackConfig(bearing_range=0.8))
    origin = np.array([0.0, 0.0, 0.2])
    direction = np.array([1.0, 1.0, 0.0])
    for _ in range(60):
        trk.predict(0.01)
        trk.update_bearing(origin, direction)
    to_target = trk.position - origin
    assert np.dot(to_target / np.linalg.norm(to_target),
                  direction / np.linalg.norm(direction)) > 0.98
