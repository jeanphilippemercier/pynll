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

import subprocess
from distutils.command.build_ext import build_ext
from distutils.core import setup, Extension
from distutils.command.install import install as DistutilsInstall
from distutils.errors import DistutilsSetupError
from distutils import log as distutils_logger
import os
from glob import glob


nlloc = Extension('nlloc',
                  include_dirs=['pynll/nlloc/'],
                  libraries=['pynll/nlloc/'],
                  library_dirs=['pynll/nlloc/'],
                  sources=['pynll/nlloc/Nlloc1.c'])


class SpecializedBuildExt(build_ext, object):
    """
    Specialized builder for testlib library

    """

    special_extension = nlloc.name

    def build_extension(self, ext):

        bin_dirs = os.environ['PATH'].split(':')

        for bin_dir in bin_dirs:
            if 'virtualenv' in bin_dir:
                break

        os.environ['BINDIR'] = bin_dir

        if ext.name!=self.special_extension:
            # Handle unspecial extensions with the parent class' method
            super(SpecializedBuildExt, self).build_extension(ext)
        else:
            # Handle special extension
            sources = ext.sources
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(
                    "in 'ext_modules' option (extension '%s'), "
                    "'sources' must be present and must be "
                    "a list of source filenames" % ext.name)
            sources = list(sources)

            if len(sources)>1:
                sources_path = os.path.commonpath(sources)
            else:
                sources_path = os.path.dirname(sources[0])
            sources_path = os.path.realpath(sources_path)
            if not sources_path.endswith(os.path.sep):
                sources_path+= os.path.sep

            distutils_logger.info(f'source path: {sources_path}')

            if not os.path.exists(sources_path) or not os.path.isdir(sources_path):
                raise DistutilsSetupError(
                    "in 'extensions' option (extension '%s'), "
                    "the supplied 'sources' base dir "
                    "must exist" % ext.name)

            output_lib = 'nlloc'
            output_dir = 'nlloc'

            distutils_logger.info('Will execute the following command in '
                                  'with subprocess.Popen: \n{0}'.format(
                'make'.format(output_lib,
                              os.path.join(output_dir, output_lib))))

            make_process = subprocess.Popen('make',
                                            cwd=sources_path,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            shell=True)
            stdout, stderr = make_process.communicate()
            distutils_logger.debug(stdout)

            # cmd = f'for FILE in Vel2Grid hypoe2hyp oct2grid NLLoc ' \
            #       f'LocSum Grid2Time Time2Angles Grid2GMT PhsAssoc' \
            #       f'Time2EQ fpfit2hyp GridCascadingDecimate Loc2ddct ' \
            #       f'NLDiffLoc Vel2Grid3D interface2fmm scat2latlon; do ' \
            #       f'mv $FILE {bin_dir}/$FILE; done'
            #
            # mv_to_path_process = subprocess.Popen(cmd,
            #                                       cwd=sources_path,
            #                                       stdout=subprocess.PIPE,
            #                                       stderr=subprocess.PIPE,
            #                                       shell=True)

            # stdout, stderr = mv_to_path_process.communicate()
            distutils_logger.debug(stdout)
            if stderr:
                raise DistutilsSetupError('An ERROR occured while running the '
                                          'Makefile for the {0} library. '
                                          'Error status: {1}'.format(output_lib, stderr))
            super(SpecializedBuildExt, self).build_extension(ext)


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    print(os.getcwd())
    setup(name="pynll",
          version="1.0.0",
          description="Python interface for the NLLoc C library function",
          author="Jean-Philippe Mercier",
          author_email="jpmercier01@gmail.com",
          ext_modules=[nlloc],
          cmdclass={'build_ext': SpecializedBuildExt})
