from setuptools import setup
from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension


setup(
    name="_relu_musa",
    ext_modules=[
        MUSAExtension(
            name="_relu_musa",
            sources=["binding.cpp", "kernel.mu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "mcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
