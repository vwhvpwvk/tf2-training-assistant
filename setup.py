import setuptools

setuptools.setup(
        name="tf2-training-assistant", # library name will go here
        version="1.0.0", # version info
        author="awesome_author",
        description="awesome_description",
        packages=setuptools.find_packages(),
        install_requires=[
            "keras-efficientnet-v2 @ git+https://github.com/vwhvpwvk/keras_efficientnet_v2.git@dev-tf2.16" # whatever library and version 
		# to specify version, use 1.2.*
		# 1.*.*
		# etc.
            ]
)

