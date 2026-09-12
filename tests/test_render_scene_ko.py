import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'render_scene_ko',
    Path(__file__).resolve().parents[1] / 'tools/render_scene_ko.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class RenderSceneTest(unittest.TestCase):
    def test_known_labels_are_natural_and_deduplicated(self):
        text, unknown = renderer.render(
            'Workstation, robot arm, laptop, notebook, laptop.')
        self.assertEqual(unknown, [])
        self.assertEqual(
            text, '여기에는 작업대, 로봇 팔, 노트북과 공책이 보여요.')

    def test_unknown_label_fails_closed(self):
        text, unknown = renderer.render('Office chair, mystery device.')
        self.assertEqual(text, '여기에는 사무용 의자가 보여요.')
        self.assertEqual(unknown, ['mystery device'])

    def test_particles_follow_final_consonant(self):
        self.assertEqual(renderer.render('USB charger, power adapter.')[0],
                         '여기에는 USB 충전기와 전원 어댑터가 보여요.')
        self.assertEqual(renderer.render('Laptop, notebook.')[0],
                         '여기에는 노트북과 공책이 보여요.')


if __name__ == '__main__':
    unittest.main()
