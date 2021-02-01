# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: <filename>
#  Purpose: <purpose>
#   Author: <author>
#    Email: <email>
#
# Copyright (C) <copyright>
# --------------------------------------------------------------------
"""


:copyright:
    <copyright>
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

# build.py

from typing import Any, Dict

from setuptools_cpp import (CMakeExtension, ExtensionBuilder,
                            Pybind11Extension)

ext_modules = [
    # An extension with a custom <project_root>/src/ext2/CMakeLists.txt:
    CMakeExtension(f"nlloc.ext2", sourcedir="nlloc"),
]


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": dict(build_ext=ExtensionBuilder),
            "zip_safe": False,
        }
    )
