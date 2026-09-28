"""Rendering a trained run keeps its backend unless explicitly overridden."""
import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from arguments import ModelParams, PipelineParams, get_combined_args


class RasterizerConfigTests(unittest.TestCase):
    def load_pipeline(self, extra_args):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "cfg_args").write_text(
                str(argparse.Namespace(rasterizer_backend="rade"))
            )
            parser = argparse.ArgumentParser()
            ModelParams(parser, sentinel=True)
            pipeline = PipelineParams(parser)
            with patch.object(sys, "argv", ["render", "-m", directory, *extra_args]):
                return pipeline.extract(get_combined_args(parser))

    def test_saved_backend_survives_parser_default(self):
        self.assertEqual(self.load_pipeline([]).rasterizer_backend, "rade")

    def test_equals_style_explicit_override_wins_over_saved_backend(self):
        pipeline = self.load_pipeline(["--rasterizer_backend=diff_gauss"])
        self.assertEqual(pipeline.rasterizer_backend, "diff_gauss")


if __name__ == "__main__":
    unittest.main()
