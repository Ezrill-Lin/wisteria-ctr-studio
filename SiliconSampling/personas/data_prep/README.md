# Silicon Sampling: Three-Strategy Personality Assignment Framework

## Overview

This framework implements and compares **three different strategies** for assigning OCEAN personality traits to US Census personas, with the goal of identifying which approach produces the most human-like synthetic personas for CTR prediction research.

## Directory Structure

```
SiliconSampling/data/
├── big5data.csv                           # IPIP-NEO Big Five dataset (test set)
├── codebook.txt                           # IPIP dataset documentation
├── code_reference.json                    # Census code mappings
├── acs_pums_2023.parquet                 # Raw US Census data
├── acs_pums_2023_labeled.parquet         # Labeled census with demographics
├── download_census_data.py               # Download raw census data from API
├── label.py                              # Label census codes to human-readable format
├── synthesize_population.py              # Generic stratified sampling (privacy-preserving)
│
├── wpp_matching/                         # Strategy 1: WPP Demographic Matching
│   ├── assign_wpp_ocean.py                  # Complete WPP pipeline (single script)
│   ├── census_with_personality_1m.parquet   # 1M personas with WPP OCEAN levels ✅
│   ├── wpp_merged_result_VT.xlsx           # WPP survey data (Vietnamese)
│   └── README.md                            # WPP strategy documentation
│
├── random_matching/                      # Strategy 2: Random OCEAN Assignment
│   ├── assign_random_ocean.py               # Random assignment implementation
│   └── census_random_ocean_1m.parquet       # 1M personas with random scores (0-10) ✅
│
└── ipip_matching/                        # Strategy 3: IPIP Demographic Matching
    ├── match_ipip_demographics.py           # IPIP matching implementation
    ├── census_ipip_ocean_1m.parquet         # 1M personas with IPIP-derived scores ✅
    └── ipip_matching_report.txt             # Matching statistics
```

---

## Three Strategies

### Strategy 1: WPP Demographic Matching ✅ COMPLETE
**Status:** Implemented in `wpp_matching/assign_wpp_ocean.py`

**Methodology:**
- Single script handles complete workflow:
  1. Apply WPP demographic mappings (age groups, employment, income)
  2. Generate 1M synthetic population via stratified sampling
  3. Load WPP survey and extract OCEAN traits
  4. Match census personas to WPP respondents
  5. Assign OCEAN (fallback to random if no exact match)
- Matching criteria: Age group, Gender, Employment status, Income bracket

**Data Source:** 
- WPP survey: 1,055 Vietnamese respondents
- 13 personality dimensions (5 OCEAN + 6 psychological + 2 cultural)

**Output:**
- `census_with_personality_1m.parquet` (1M personas, 19 columns)
- OCEAN values: Text labels (e.g., "Agree", "Neutral")

**Pros:**
- Uses real survey responses with authentic trait correlations
- Preserves complete personality profiles from single individuals

**Cons:**
- Geographic mismatch (US census vs Vietnamese respondents)
- Low exact match rate (0.5% demographic matches, 99.5% fallback)
- Text-based OCEAN levels (not numeric scores)

---

### Strategy 2: Random OCEAN Assignment ⚡ NEW
**Status:** Implemented in `random_matching/assign_random_ocean.py`

**Methodology:**
- Assign **random numeric scores** (0-10) to each OCEAN trait
- Uniform distribution across the full 0-10 range
- No correlation between traits or demographics
- Serves as baseline/control condition

**Data Source:** 
- None (purely random generation)

**Output:**
- `census_random_ocean_1m.parquet` (1M personas)
- OCEAN values: Continuous numeric (0-10)

**Pros:**
- Simple baseline for comparison
- No demographic bias or data availability issues
- Numerically consistent (0-10 scale)

**Cons:**
- Lacks realistic trait correlations
- No demographic-personality patterns
- Likely to produce unrealistic personas

**Usage:**
```bash
cd SiliconSampling/data/random_matching
python assign_random_ocean.py
```

---

### Strategy 3: IPIP Demographic Matching ⚡ NEW
**Status:** Implemented in `ipip_matching/match_ipip_demographics.py`

**Methodology:**
- Match US Census demographics to IPIP-NEO Big Five dataset
- Calculate OCEAN **scores** (0-10 numeric) from IPIP-50 questionnaire items
- Matching criteria: Age group + Gender
- Properly handle reverse-coded items
- Fallback: Random sample from full IPIP pool

**Data Source:**
- IPIP-NEO dataset: ~1 million respondents (international)
- 50-item Big Five questionnaire (10 items per trait)
- Demographics: race, age, gender, country, native language

**OCEAN Score Calculation:**
1. Extract 10 items per trait (e.g., E1-E10 for Extraversion)
2. Reverse-code items marked with "_R" suffix: `score = 6 - original`
3. Calculate mean across 10 items (ignoring missed responses)
4. Scale from 1-5 to 0-10: `(score - 1) × 2.5`

**Output:**
- `census_ipip_ocean_1m.parquet` (1M personas)
- OCEAN values: Continuous numeric (0-10), derived from validated questionnaire
- `ipip_matching_report.txt`: Matching statistics

**Pros:**
- Large sample size (~1M respondents)
- International dataset (better demographic diversity)
- Numeric scores derived from validated IPIP-50 instrument
- Real questionnaire responses with natural trait correlations

**Cons:**
- Still some demographic mismatches (different race categorizations)
- Missing data in some IPIP responses

**Usage:**
```bash
cd SiliconSampling/data/ipip_matching
python match_ipip_demographics.py
```

---

## Validation Experiment

### Objective
Compare which strategy produces the most **human-like personas** when used in LLM-based CTR prediction.

### Test Set
- **Dataset:** `big5data.csv` (IPIP-NEO dataset with ~1M responses)
- **Items:** 50 Big Five questionnaire items
- **Ground truth:** Real human responses to personality questions

### Validation Approach

1. **Generate Personas** from each strategy:
   - Strategy 1 (WPP): 1M personas with text-based OCEAN levels
   - Strategy 2 (Random): 1M personas with random 0-10 scores
   - Strategy 3 (IPIP): 1M personas with IPIP-derived 0-10 scores

2. **LLM Response Testing:**
   - For each strategy, create persona prompts with demographic + OCEAN info
   - Ask LLM to answer IPIP-50 questions as the persona
   - Collect LLM responses (1-5 scale: Disagree to Agree)

3. **Statistical Comparison** (LLM vs Ground Truth):
   - **Mean absolute error (MAE):** How close are average responses?
   - **Distribution similarity:** KL divergence, Jensen-Shannon divergence
   - **Correlation analysis:** Do trait relationships match?
   - **Variance comparison:** Is response variability realistic?
   - **Item-level accuracy:** Which questions are hardest to simulate?

4. **Metrics:**
   - Overall humanlikeness score
   - Per-trait accuracy (O, C, E, A, N)
   - Demographic subgroup performance
   - Response pattern realism

### Expected Outcomes

**Hypothesis:**
- **Strategy 2 (Random)** will perform worst (no realistic correlations)
- **Strategy 1 (WPP)** may struggle due to text-based levels and Vietnamese bias
- **Strategy 3 (IPIP)** should perform best (numeric scores, validated instrument, demographic diversity)

### Implementation Status
- [ ] Create `test_personas/validate_personas.py` script
- [ ] Implement LLM prompting for each strategy
- [ ] Collect LLM responses to IPIP-50 questions
- [ ] Calculate statistical comparison metrics
- [ ] Generate `comparison_report.md` with visualizations

---

## Running the Full Pipeline

### Step 0: Download and Label Census Data (if not done)

```bash
cd SiliconSampling/data

# Download raw census data from API
python download_census_data.py

# Label census codes to human-readable format
python label.py
```

**Output:**
- `acs_pums_2023.parquet` - Raw census data
- `acs_pums_2023_labeled.parquet` - Labeled census data

### Step 1: Generate Three Datasets

```bash
cd SiliconSampling/data

# Strategy 1: WPP
cd wpp_matching
python assign_wpp_ocean.py
# Output: census_with_personality_1m.parquet ✅

# Strategy 2: Random
cd ../random_matching
python assign_random_ocean.py
# Output: census_random_ocean_1m.parquet ✅

# Strategy 3: IPIP
cd ../ipip_matching
python match_ipip_demographics.py
# Output: census_ipip_ocean_1m.parquet ✅
```

### Step 2: Validate Personas (Future)

```bash
cd ../test_personas
python validate_personas.py --strategy all --sample-size 10000
# Output: comparison_report.md with statistical results
```

### Step 3: Select Best Strategy

Based on validation results, choose the strategy that produces the most human-like LLM responses for downstream CTR prediction tasks.

---

## Key Differences Between Strategies

| Aspect | WPP Matching | Random | IPIP Matching |
|--------|--------------|--------|---------------|
| **OCEAN Format** | Text labels (7-point) | Numeric (0-10) | Numeric (0-10) |
| **Data Source** | Vietnamese survey | Random generation | International IPIP |
| **Sample Size** | 1,055 | N/A | ~1M |
| **Matching Basis** | Age+Gender+Employment+Income | None | Age+Gender |
| **Trait Correlations** | Real (from single person) | None | Real (from questionnaire) |
| **Geographic Bias** | Vietnamese | None | International |
| **Exact Match Rate** | 0.5% | N/A | Expected ~20-40% |
| **Validation Source** | Survey items | None | IPIP-50 items |

---

## Research Questions

1. **Does demographic matching improve persona realism?**
   - Compare Strategy 2 (random) vs Strategy 1/3 (matched)

2. **Does numeric scoring enable better LLM simulation?**
   - Compare Strategy 1 (text) vs Strategy 3 (numeric)

3. **Does dataset size/diversity matter?**
   - Compare Strategy 1 (1K Vietnamese) vs Strategy 3 (1M international)

4. **What is the optimal matching granularity?**
   - Analyze exact match vs fallback performance in Strategy 1 and 3

---

## Next Steps

1. ✅ Implement random OCEAN assignment (Strategy 2)
2. ✅ Implement IPIP demographic matching (Strategy 3)
3. ⏳ Create validation framework in `test_personas/`
4. ⏳ Run LLM response testing on all three strategies
5. ⏳ Perform statistical comparison
6. ⏳ Select winning strategy for CTR prediction research

---

## Contact & Contribution

- **Framework Design:** Three-strategy comparison for persona generation
- **Goal:** Identify most human-like personality assignment method
- **Application:** LLM-based CTR prediction with realistic synthetic personas

**Last Updated:** November 28, 2025
