# SiliconSampling

## Current Status

⚠️ **Important Notice**: This module is currently in a temporary development stage.

## Current Implementation

The `sampler.py` file is currently using **purely random sampled data with no valid sources** from the `identity_bank.json` file. This is a temporary implementation for the current development stage and should not be considered production-ready.

## Future Development

- **Synthetic Data**: We have synthetic data based on real US Census data (stored in parquet files) that is planned to be integrated
- **Data Sources**: The synthetic data based on US Census will provide more realistic and representative sampling
- **Implementation**: The current `sampler.py` is only temporarily used for the current stage and will be replaced or significantly updated

## Files

- `sampler.py` - Temporary random data sampler (current implementation)
- `data/` - Directory containing reference data and notebooks
  - `acs_pums_2023.parquet` - **Real US Census data (ACS PUMS 2023) - NOT YET IN USE**
  - `synth_demo.parquet` - **Synthetic demographic data based on US Census - NOT YET IN USE**
  - `identity_bank.json` - **Currently used random sampled data with no valid sources**
  - `code_reference.json` - Code reference data
  - `data.ipynb` - Data exploration notebook

## Next Steps

### Immediate Development Goals

1. **Integrate US Census-based synthetic data**: Replace random sampling with realistic demographic data from the parquet files
2. **Build Persona Agent**: Combine synthetic demographic data with sampled personality data to create comprehensive digital personas
3. **Multi-Agent Integration**: Implement the complete multi-agent architecture as described in the main project README:
   - **Demographic Sampling Agent**: Use `acs_pums_2023.parquet` and `synth_demo.parquet` for realistic population modeling
   - **Personality Sampling Agent**: Integrate Big Five personality trait generation (see `PersonalitySamplingAgent/`)
   - **Persona Generation Agent**: Synthesize demographics + personality into coherent synthetic personas

---

**Note**: Do not rely on the current sampling implementation for production use or accurate demographic representation.