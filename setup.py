from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='ncnn_cpp',
      ext_modules=[cpp_extension.CppExtension('ncnn_cpp', ['NCNN.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
