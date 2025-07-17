"""
Version information for Intelligent AutoML Framework
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "The world's most intelligent automated machine learning framework"
__url__ = "https://github.com/AhmedMansour1070/intelligent-automl"
__license__ = "MIT"
__copyright__ = "2024, Ahmed Mansour"

# Version info tuple
VERSION = tuple(map(int, __version__.split('.')))

# Framework metadata
FRAMEWORK_NAME = "Intelligent AutoML"
SUPPORTED_PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]
MINIMUM_PYTHON_VERSION = "3.8"

def get_version():
    """Get the version string."""
    return __version__

def get_version_info():
    """Get detailed version information."""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'url': __url__,
        'license': __license__,
        'python_versions': SUPPORTED_PYTHON_VERSIONS,
    }