import json
import os
import random
import tempfile
import unittest
from unittest import mock

import omnivoice_voice_pool as vp


def _write_pool(presets) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"presets": presets}, f)
    return path


class LoadVoicePoolTests(unittest.TestCase):
    def test_loads_presets_object_form(self):
        path = _write_pool(
            [
                {"id": "warm_01", "name": "Maya", "weight": 2, "tags": ["young", "warm"]},
                {"id": "warm_02"},
            ]
        )
        self.addCleanup(os.remove, path)
        presets = vp.load_voice_pool(path)
        self.assertEqual([p.id for p in presets], ["warm_01", "warm_02"])
        self.assertEqual(presets[0].name, "Maya")
        self.assertEqual(presets[0].weight, 2.0)
        self.assertEqual(presets[0].tags, ("young", "warm"))

    def test_loads_bare_list_form(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump([{"id": "a"}, {"id": "b"}], f)
        self.addCleanup(os.remove, path)
        self.assertEqual(len(vp.load_voice_pool(path)), 2)

    def test_missing_file_raises(self):
        with self.assertRaises(vp.VoicePoolError):
            vp.load_voice_pool("/nonexistent/pool.json")

    def test_empty_pool_raises(self):
        path = _write_pool([])
        self.addCleanup(os.remove, path)
        with self.assertRaises(vp.VoicePoolError):
            vp.load_voice_pool(path)

    def test_preset_without_id_raises(self):
        path = _write_pool([{"name": "no id"}])
        self.addCleanup(os.remove, path)
        with self.assertRaises(vp.VoicePoolError):
            vp.load_voice_pool(path)


class SelectPresetTests(unittest.TestCase):
    def setUp(self):
        self.presets = [
            vp.VoicePreset(id="a", weight=1.0),
            vp.VoicePreset(id="b", weight=1.0),
            vp.VoicePreset(id="c", weight=1.0),
        ]

    def test_round_robin_is_deterministic(self):
        rng = random.Random(0)
        ids = [
            vp.select_preset(self.presets, rotation="round_robin", rng=rng, rr_index=i).id
            for i in range(5)
        ]
        self.assertEqual(ids, ["a", "b", "c", "a", "b"])

    def test_random_is_seeded_and_in_pool(self):
        rng = random.Random(42)
        p = vp.select_preset(self.presets, rotation="random", rng=rng)
        self.assertIn(p.id, {"a", "b", "c"})

    def test_weighted_respects_zero_weight(self):
        presets = [
            vp.VoicePreset(id="never", weight=0.0),
            vp.VoicePreset(id="always", weight=5.0),
        ]
        rng = random.Random(1)
        picks = {
            vp.select_preset(presets, rotation="weighted", rng=rng).id for _ in range(50)
        }
        self.assertEqual(picks, {"always"})

    def test_empty_pool_raises(self):
        with self.assertRaises(vp.VoicePoolError):
            vp.select_preset([], rotation="random", rng=random.Random())


class VoicePoolSelectorTests(unittest.TestCase):
    def test_disabled_pool_yields_none(self):
        cfg = vp.VoicePoolConfig(enabled=False, path="x", rotation="random")
        self.assertIsNone(vp.VoicePoolSelector(cfg).select())

    def test_bad_path_records_error_and_yields_none(self):
        cfg = vp.VoicePoolConfig(enabled=True, path="/nope.json", rotation="random")
        sel = vp.VoicePoolSelector(cfg)
        self.assertFalse(sel.available())
        self.assertIsNotNone(sel.load_error)
        self.assertIsNone(sel.select())

    def test_round_robin_advances_across_calls(self):
        path = _write_pool([{"id": "a"}, {"id": "b"}])
        self.addCleanup(os.remove, path)
        cfg = vp.VoicePoolConfig(enabled=True, path=path, rotation="round_robin")
        sel = vp.VoicePoolSelector(cfg)
        self.assertEqual([sel.select().id, sel.select().id, sel.select().id], ["a", "b", "a"])

    def test_stable_within_session_means_one_select_per_session(self):
        # The selector returns one preset per call; the agent calls select() once
        # per session, so the preset is fixed for that session by construction.
        path = _write_pool([{"id": "only"}])
        self.addCleanup(os.remove, path)
        cfg = vp.VoicePoolConfig(enabled=True, path=path, rotation="random")
        sel = vp.VoicePoolSelector(cfg)
        self.assertEqual(sel.select().id, "only")


class VoicePoolConfigEnvTests(unittest.TestCase):
    def test_from_env_defaults_and_rotation_validation(self):
        with mock.patch.dict(os.environ, {"OMNIVOICE_VOICE_ROTATION": "bogus"}, clear=False):
            self.assertEqual(vp.VoicePoolConfig.from_env().rotation, "random")
        with mock.patch.dict(
            os.environ,
            {
                "OMNIVOICE_VOICE_POOL_ENABLED": "true",
                "OMNIVOICE_VOICE_POOL_PATH": "/p.json",
                "OMNIVOICE_VOICE_ROTATION": "round_robin",
            },
            clear=False,
        ):
            c = vp.VoicePoolConfig.from_env()
        self.assertTrue(c.enabled)
        self.assertEqual(c.path, "/p.json")
        self.assertEqual(c.rotation, "round_robin")


if __name__ == "__main__":
    unittest.main()
