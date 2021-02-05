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


cdef extern from 'NLLoc_main_func.h':
# cdef extern from '../nlloc/*':

    cdef int NLLoc_func(char* ctrl_file)

# NLLoc_fun(&file_str)


def run_nlloc(ctrl_file):

    return NLLoc_func(ctrl_file)




