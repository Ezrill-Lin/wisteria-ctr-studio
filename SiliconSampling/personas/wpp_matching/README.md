# WPP Matching Strategy

## Overview
This folder implements **Strategy 1: WPP Demographic Matching** for assigning OCEAN personality traits to US Census personas using Vietnamese WPP survey data.

## Single Script Workflow

**`assign_wpp_ocean.py`** - Complete WPP matching pipeline in one script:
1. Load labeled census data
2. Apply WPP demographic mappings (age groups, employment, income)
3. Generate 1M synthetic population via stratified sampling
4. Load WPP survey data and extract OCEAN traits
5. Match census personas to WPP respondents
6. Assign OCEAN traits (fallback to random if no exact match)
7. Save final dataset

## Prerequisites

Make sure you've run the common data preparation scripts:
```bash
cd ..
python download_census_data.py  # Downloads raw census
python label.py                 # Labels census codes
```

## Usage

```bash
cd wpp_matching
python assign_wpp_ocean.py
```

## Key Files

### Input
- `../acs_pums_2023_labeled.parquet` - Labeled census data (from parent folder)
- `wpp_merged_result_VT.xlsx` - Vietnamese WPP survey data (1,055 respondents)

### Output
- `census_with_personality_1m.parquet` ✅ **FINAL: 1M personas with WPP OCEAN levels**
- `census_with_personality_preview.csv` - Preview of first 1,000 personas

## Output Format

`census_with_personality_1m.parquet` contains:
- **Demographics**: age, gender, state, race, education, occupation, employment, income
- **WPP Mappings**: wpp_age_group, wpp_gender, wpp_employment, wpp_income_range, matching_key
- **OCEAN Traits** (text levels): openness, conscientiousness, extraversion, agreeableness, neuroticism

### OCEAN Format
Text-based 7-point scale from WPP survey responses.

## Matching Statistics
- Exact match rate: ~0.5% (granular demographic criteria: age + gender + employment + income)
- Fallback rate: ~99.5% (random sampling from WPP pool when no exact match)
- Source: Vietnamese survey population (1,055 respondents)

## Notes
- This strategy uses real survey responses with authentic trait correlations
- Geographic mismatch: US census paired with Vietnamese respondents
- OCEAN values are text levels, not numeric scores (unlike Random and IPIP strategies)
