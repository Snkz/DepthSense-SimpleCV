from distutils.core import setup, Extension

module = Extension('DepthSense',
        include_dirs = ['/usr/local/include', '/opt/softkinetic/DepthSenseSDK/include'],
        libraries = ['DepthSensePlugins', 'DepthSense', 'python2.7', 'turbojpeg'],
        library_dirs = ['/usr/local/lib', '/opt/softkinetic/DepthSenseSDK/lib'],
        #extra_compile_args = ['-std=g++11'],
        sources = ['src/depthsense.cxx', 'src/initdepthsense.cxx', 'src/imageproccessing.cxx'])

setup (name = 'DepthSense',
        version = '1.0',
        description = 'Python Wrapper for the DepthSense SDK',
        author = 'Abdi Dahir',
        author_email = 'abdi.dahir@outlook.com',
        url = 'http://github.com/snkz/DepthSense-SimpleCV',
        long_description = '''The Python DepthSense SDK wrapper allows basic
        interaction with the DepthSense camera, compitable with SimpleCV.''',
        ext_modules = [module])
