import os
import sys
import unittest


def run(tests_dir: str, project_root: str):
    loader = unittest.TestLoader()
    tests = loader.discover(start_dir=tests_dir, pattern='[!_]*.py', top_level_dir=project_root)
    flat_tests = [test for suite_ in tests for test in suite_]

    sorted_tests = sorted(flat_tests, key=lambda test: test.__class__.__name__)
    suite = unittest.TestSuite(sorted_tests)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    _tests_dir: str = os.path.dirname(__file__)
    _project_root: str = os.path.realpath(os.path.join(_tests_dir, os.pardir))
    src_dir: str = os.path.realpath(os.path.join(_project_root, 'src'))

    sys.path.insert(0, _project_root)
    sys.path.insert(0, src_dir)
    run(_tests_dir, _project_root)
