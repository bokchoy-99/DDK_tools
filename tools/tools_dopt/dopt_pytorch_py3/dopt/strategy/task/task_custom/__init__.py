#!/usr/bin/env python
# encoding: utf-8
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved
"""npu opt tool chain init"""
from os.path import dirname, join, isdir, abspath, basename, realpath
from glob import glob


class TaskImport:
    """
    CLASS TaskImport
    """

    @staticmethod
    def file_import():
        """ file import """
        pwd = dirname(__file__)
        name = __name__
        import_num = 0
        for py_file in glob(join(pwd, '*.py')):
            if not py_file.endswith('__.py'):
                name_module = name[:-8] + basename(py_file)[:-3]
                __import__(name_module, globals(), locals())
                import_num = import_num + 1

        if import_num == 0:
            for so_file in glob(join(pwd, '*.so')):
                if not so_file.endswith('__.so'):
                    name_module = name[:-8] + basename(so_file)[:-3]
                    __import__(name_module, globals(), locals())

