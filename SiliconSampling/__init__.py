"""SiliconSampling Package

This package provides utilities for generating synthetic populations based on
configurable identity banks. It includes sampling functions and data management
for creating diverse demographic profiles for research and simulation purposes.

Main Components:
- sampler: Core sampling utilities and identity bank loading
- data: Identity bank configurations and demographic distributions  
- PersonalitySamplingAgent: Big Five personality profile generation and testing
"""

from .sampler import load_identity_bank, sample_identities

__version__ = "1.0.0"
__all__ = ["load_identity_bank", "sample_identities"]