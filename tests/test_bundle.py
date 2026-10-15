"""Cloud transfers must contain only explicitly selected source files."""
import importlib.util
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('build_cloud_bundle', Path(__file__).resolve().parents[1] / 'tools/build_cloud_bundle.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BundleTests(unittest.TestCase):
    def test_private_material_excluded_and_bundle_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('README.md', 'src/semflow/core.py', 'context/state.md', '.env', 'AGENTS.md',
                         'egocentric_human_data_manipulation_recap.md', 'data/clip.mp4'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('test')
            a, b = root/'a.tar.gz', root/'b.tar.gz'
            module.build_bundle(root, a)
            module.build_bundle(root, b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with tarfile.open(a) as archive:
                self.assertEqual(set(archive.getnames()), {'README.md', 'src/semflow/core.py', 'SOURCE_MANIFEST.json'})

    def test_source_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'private.txt').write_text('secret')
            (root/'src').mkdir()
            (root/'src/leak.py').symlink_to(root/'private.txt')
            with self.assertRaises(ValueError):
                module.build_bundle(root, root/'bundle.tar.gz')


if __name__ == '__main__':
    unittest.main()
