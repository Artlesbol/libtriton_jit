# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

"""Check compiler-entry dispatch without importing Torch or a device backend."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    tree = ast.parse(Path(sys.argv[1]).read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_compile_backend_ast")
    calls = []

    def record(label):
        def compile_fn(src, *, target, options):
            calls.append((label, src, target, options))
            return label
        return compile_fn

    triton = SimpleNamespace(compile=record("alias"),
                             compiler=SimpleNamespace(compile=record("hook")))
    scope = {"triton": triton}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<helper>", "exec"), scope)
    compile_ast = scope["_compile_backend_ast"]
    src, target, options = object(), object(), {"num_warps": 4}
    assert compile_ast(src, target, options, "NPU") == "hook"
    assert calls[-1] == ("hook", src, target, options)
    assert calls[-1][3] is options
    triton.compiler.compile = record("late_hook")
    assert compile_ast(src, target, options, "NPU") == "late_hook"
    assert calls[-1] == ("late_hook", src, target, options)
    del triton.compiler
    for backend in ("CUDA", "IX", "MUSA", "MTGPU", "MACA", "GCU", "MLU"):
        assert compile_ast(src, target, options, backend) == "alias"
        assert calls[-1] == ("alias", src, target, options)

    def fail(*args, **kwargs):
        raise RuntimeError("compile failure")

    triton.compiler = SimpleNamespace(compile=fail)
    try:
        compile_ast(src, target, options, "NPU")
    except RuntimeError as error:
        assert str(error) == "compile failure"
    else:
        raise AssertionError("compiler exception was hidden")
    print("10 compiler-entry checks passed")


if __name__ == "__main__":
    main()
