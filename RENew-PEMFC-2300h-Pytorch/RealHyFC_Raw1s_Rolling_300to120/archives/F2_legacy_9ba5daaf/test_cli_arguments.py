"""Argument regression tests; no data, weights, inference, or training are loaded."""
import argparse
import ast
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def load_argument_parser():
    source = Path(__file__).with_name('RealHyFC_F2_Main.py')
    tree = ast.parse(source.read_text(encoding='utf-8'))
    names = {'SUPPORTED_SEEDS', 'PARTITIONS', 'DEFAULT_EVALUATION_PARTITION'}
    nodes = [node for node in tree.body if (
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
    ) or (isinstance(node, ast.FunctionDef) and node.name == 'parse_arguments')]
    namespace = {'argparse': argparse, 'sys': sys}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['parse_arguments']


parse_arguments = load_argument_parser()


class ArgumentTests(unittest.TestCase):
    def parse_kernel(self, arguments):
        with patch.dict(sys.modules, {'ipykernel': object()}), patch.object(
            sys, 'argv', ['ipykernel_launcher.py', *arguments]
        ):
            return parse_arguments()

    def test_reported_kernel_connection_argument(self):
        args = self.parse_kernel([r'--f=C:\Users\1\jupyter\runtime\kernel-test.json'])
        self.assertEqual((args.mode, args.seed, args.partition), ('evaluate', '42', 'VALIDATION'))

    def test_separate_kernel_connection_argument(self):
        for flag in ('--f', '-f'):
            with self.subTest(flag=flag):
                self.assertEqual(self.parse_kernel([flag, 'kernel.json', '--seed', '103']).seed, '103')

    def test_kernel_option_preserves_model_arguments(self):
        args = self.parse_kernel(['--mode', 'train', '--f=kernel.json', '--seed', '193'])
        self.assertEqual((args.mode, args.seed), ('train', '193'))

    def test_unknown_kernel_argument_still_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.parse_kernel(['--f=kernel.json', '--sead', '103'])
        self.assertEqual(error.exception.code, 2)

    def test_explicit_arguments_take_priority(self):
        with patch.dict(sys.modules, {'ipykernel': object()}), patch.object(
            sys, 'argv', ['ipykernel_launcher.py', '--f=kernel.json']
        ):
            self.assertEqual(parse_arguments(['--seed', 'all']).seed, 'all')

    def test_terminal_arguments_unchanged(self):
        with patch.object(sys, 'argv', ['RealHyFC_F2_Main.py', '--seed', '212']):
            self.assertEqual(parse_arguments().seed, '212')

    def test_connection_option_not_ignored_outside_kernel(self):
        with patch.dict(sys.modules), patch.object(sys, 'argv', ['RealHyFC_F2_Main.py', '--f=kernel.json']):
            sys.modules.pop('ipykernel', None)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments()

    def test_train_all_remains_forbidden(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.parse_kernel(['--f=kernel.json', '--mode', 'train', '--seed', 'all'])
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
