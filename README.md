# Wisteria CTR Studio

## Overview

Wisteria CTR Studio is a **privacy-preserving CTR prediction system** that uses **SiliconSampling** synthetic personas to simulate real human responses to advertisements. The system predicts click-through rates using LLM-powered behavioral modeling, providing detailed insights and recommendations without requiring real user data.

### Key Innovation: Persona-Based CTR Prediction

Traditional CTR prediction relies on real user data, raising privacy concerns and regulatory issues. Wisteria CTR Studio creates **psychologically realistic synthetic personas** with complete demographic and personality profiles, enabling:

- **🔒 Privacy-Compliant**: No real user data - GDPR/CCPA compliant
- **🧠 Psychologically Grounded**: Based on Big Five personality framework with realistic trait manifestations
- **📊 Individual Reasoning**: Each persona provides click decision + 1-3 sentence explanation
- **💡 Actionable Insights**: LLM-generated final analysis with specific improvement recommendations
- **💰 Cost-Effective**: Support for OpenAI, DeepSeek, and Gemini with flexible provider selection

---

## Architecture

### System Workflow

```
📥 Input: Advertisement Content
    ↓
🧠 SiliconSampling: Load Synthetic Personas (v1 or v2)
    ├── Demographics: Age, gender, occupation, education, location
    ├── OCEAN Scores: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
    ├── V1: Persona description (2-3 paragraphs)
    └── V2: Enhanced with behavioral_tendencies + self_schema
    ↓
🤖 CTR Prediction: LLM-Based Behavioral Simulation
    ├── For each persona: 
    │   ├── System message: Embody the persona
    │   ├── User prompt: Evaluate the ad
    │   └── LLM response: Click decision (yes/no) + reasoning
    ↓
📊 Aggregation: Calculate CTR from individual responses
    ↓
💡 Analysis: LLM generates final insights
    ├── Overall assessment
    ├── What's working well
    ├── What's not working
    ├── Improvement recommendations
    └── Target audience insights
    ↓
📤 Output: CTR + Analysis + Individual Responses
```

---

## Core Components

### 1. SiliconSampling Package

**Synthetic persona generation engine** with two versions and three personality assignment strategies.

#### Persona Versions
- **V1**: Basic personas with simple 2-3 paragraph descriptions
- **V2**: Enhanced personas with:
  - `persona_description` (first-person narrative)
  - `behavioral_tendencies` (5 decision patterns)
  - `self_schema` (6-8 core belief statements)

#### Personality Assignment Strategies
- **Random**: Baseline control with random OCEAN scores (0-10)
- **WPP**: Real survey data from Vietnamese WPP study (~1,055 respondents)
- **IPIP**: IPIP-NEO Big Five dataset demographic matching

**Location:** `SiliconSampling/`

**Key Files:**
- `personas/generate_personas.py` - V1 persona generation
- `personas_v2/generate_personas_v2.py` - V2 enhanced persona generation
- `agent.py` - Persona response collection for validation
- `validate_all.py` - Validation against real US population data

### 2. CTRPrediction Package

**LLM-powered CTR prediction engine** with multi-provider support and detailed reasoning.

**Location:** `CTRPrediction/`

**Key Files:**
- `ctr_predictor.py` - Main prediction engine
  - `CTRPredictor`: Core prediction class
  - `CTRPredictionResult`: Result dataclass
  - `PersonaResponse`: Individual persona response
- `openai_client.py` - OpenAI integration (GPT-4, GPT-4o-mini)
- `deepseek_client.py` - DeepSeek integration (cost-effective)
- `gemini_client.py` - Google Gemini integration (fast)
- `base_client.py` - Abstract base class for extensibility

### 3. Demo Script

**Command-line interface** for quick CTR predictions.

**Location:** `demo.py`

**Usage:**
```bash
# Basic prediction with default settings (v2 personas, random strategy, openai)
python demo.py --ad "Special 0% APR credit card offer for travel rewards"

# Specify persona version and strategy
python demo.py --ad "Shop our new summer collection!" --persona-version v2 --persona-strategy wpp --population-size 200

# Use different LLM provider
python demo.py --ad "Try our new fitness app" --provider deepseek --population-size 50

# Save results to JSON
python demo.py --ad "Premium headphones on sale" --output results.json
```

### 4. REST API

**FastAPI web service** for programmatic CTR predictions.

**Location:** `api.py`

**Usage:**
```bash
# Start the server
python api.py

# Or with uvicorn
uvicorn api:app --host 0.0.0.0 --port 8080 --reload

# API docs: http://localhost:8080/docs
```

**Example API Request:**
```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ad_content": "Special 0% APR credit card offer",
    "population_size": 100,
    "ad_platform": "facebook",
    "persona_version": "v2",
    "persona_strategy": "random",
    "provider": "openai"
  }'
```

---

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **API Keys** (set as environment variables):
   ```bash
   # PowerShell (Windows)
   $env:OPENAI_API_KEY="your-key-here"
   $env:DEEPSEEK_API_KEY="your-key-here"  # Optional
   $env:GEMINI_API_KEY="your-key-here"    # Optional
   
   # Bash (Linux/Mac)
   export OPENAI_API_KEY="your-key-here"
   export DEEPSEEK_API_KEY="your-key-here"  # Optional
   export GEMINI_API_KEY="your-key-here"    # Optional
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/Ezrill-Lin/Wisteria-CTR-Studio.git
cd Wisteria-CTR-Studio

# Install dependencies
pip install -r requirements.txt
```

### Generate Personas (First Time Only)

Before running CTR predictions, you need to generate synthetic personas:

```bash
# Generate V2 personas with random strategy (recommended)
cd SiliconSampling/personas_v2
python generate_personas_v2.py --strategy random --sample-size 2000

# Generate V2 personas with WPP strategy
python generate_personas_v2.py --strategy wpp --sample-size 2000

# Or generate V1 personas (simpler)
cd ../personas
python generate_personas.py --strategy random --sample-size 2000
```

### Run Your First Prediction

```bash
# Return to project root
cd ../..

# Run a prediction
python demo.py --ad "Limited time offer: 50% off premium subscription!" --population-size 100
```

---

## Usage Examples

### Command-Line Demo

```bash
# V2 personas with random strategy (recommended for balanced results)
python demo.py \
  --ad "Discover eco-friendly products for sustainable living" \
  --persona-version v2 \
  --persona-strategy random \
  --population-size 200 \
  --ad-platform facebook

# Use DeepSeek for cost-effective prediction
python demo.py \
  --ad "Premium noise-canceling headphones - 40% off" \
  --provider deepseek \
  --population-size 100 \
  --output results.json

# Test different platforms
python demo.py \
  --ad "Learn Python in 30 days - online course" \
  --ad-platform youtube \
  --population-size 150
```

### Python API

```python
from CTRPrediction import CTRPredictor

# Initialize predictor
predictor = CTRPredictor(
    persona_version='v2',
    persona_strategy='random',
    provider='openai',
    model='gpt-4o-mini'
)

# Run prediction
result = predictor.predict(
    ad_content="Special offer: Buy 2 get 1 free!",
    population_size=100,
    ad_platform='facebook'
)

# Access results
print(f"Predicted CTR: {result.ctr:.2%}")
print(f"Total Clicks: {result.total_clicks}/{result.total_personas}")
print(f"\nFinal Analysis:\n{result.final_analysis}")

# Access individual persona responses
for response in result.persona_responses[:5]:  # First 5
    print(f"\nPersona {response.persona_id}: {'CLICK' if response.will_click else 'NO CLICK'}")
    print(f"Reasoning: {response.reasoning}")
```

### REST API

```python
import requests

response = requests.post(
    "http://localhost:8080/predict",
    json={
        "ad_content": "Try our new fitness app - 7 day free trial",
        "population_size": 100,
        "ad_platform": "instagram",
        "persona_version": "v2",
        "persona_strategy": "random",
        "provider": "openai",
        "include_persona_details": True
    }
)

result = response.json()
print(f"CTR: {result['ctr']:.2%}")
print(f"Analysis: {result['final_analysis']}")
```

---

## Configuration Options

### Persona Version
- **v1**: Simple persona descriptions (faster to generate)
- **v2**: Enhanced with behavioral tendencies and self-schema (more realistic)

### Persona Strategy
- **random**: Random OCEAN scores - baseline/control
- **wpp**: WPP survey matching - realistic trait correlations
- **ipip**: IPIP demographic matching - US population patterns

### LLM Providers
- **openai**: Best quality, higher cost (GPT-4o-mini recommended)
- **deepseek**: Good quality, lower cost (deepseek-chat)
- **gemini**: Fast and efficient (gemini-2.0-flash-exp)

### Ad Platforms
- **facebook**: Social media news feed
- **tiktok**: Short-form video feed
- **amazon**: E-commerce product pages
- **instagram**: Photo/video feed and stories
- **youtube**: Video platform ads

---

## Project Structure

```
Wisteria-CTR-Studio/
├── SiliconSampling/              # Persona generation and validation
│   ├── personas/                 # V1 persona generation
│   │   ├── generate_personas.py
│   │   ├── random_matching/      # Random OCEAN personas
│   │   ├── wpp_matching/         # WPP survey personas
│   │   └── ipip_matching/        # IPIP matched personas
│   ├── personas_v2/              # V2 enhanced persona generation
│   │   ├── generate_personas_v2.py
│   │   ├── random_matching/
│   │   ├── wpp_matching/
│   │   └── prompts/              # LLM prompts for generation
│   ├── agent.py                  # Persona response collection
│   ├── validate_all.py           # Validation pipeline
│   └── test/                     # Ground truth and validation utilities
│
├── CTRPrediction/                # CTR prediction engine
│   ├── ctr_predictor.py          # Main prediction logic
│   ├── openai_client.py          # OpenAI integration
│   ├── deepseek_client.py        # DeepSeek integration
│   ├── gemini_client.py          # Gemini integration
│   └── base_client.py            # Abstract base class
│
├── demo.py                       # Command-line demo
├── api.py                        # FastAPI web service
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Output Example

```
================================================================================
CTR PREDICTION RESULTS
================================================================================

📊 Overall Metrics
   Predicted CTR: 24.50%
   Total Clicks: 49 / 200
   Provider: openai
   Model: gpt-4o-mini
   Platform: facebook

✅ Personas Who WOULD Click (49):
--------------------------------------------------------------------------------

   Persona 342191
   Demographics: Age: 69, Gender: female, Occupation: N/A
   Reasoning: The promise of 0% APR aligns with my interest in managing finances 
              wisely in retirement, and travel rewards appeal to my adventurous spirit.

   Persona 128934
   Demographics: Age: 34, Gender: male, Occupation: Software Engineer
   Reasoning: I'm interested in maximizing credit card rewards for my frequent 
              business travel, and 0% APR would help me manage cash flow better.

   ... and 47 more

❌ Personas Who WOULD NOT Click (151):
--------------------------------------------------------------------------------

   Persona 445621
   Demographics: Age: 22, Gender: female, Occupation: Student
   Reasoning: I'm not interested in credit cards right now as I'm focused on 
              paying off student loans, and I don't travel much.

   ... and 150 more


📝 FINAL ANALYSIS
================================================================================

# Overall Assessment

The predicted CTR of 24.5% is **above average** for Facebook credit card offers,
which typically see CTRs between 0.5% and 2%. However, this synthetic population
may not perfectly represent real-world conversion patterns.

# What's Working Well

1. **Strong value proposition**: The 0% APR offer resonates with financially-minded
   personas across age groups
2. **Travel rewards appeal**: Particularly effective for:
   - Professionals with travel needs
   - Retirees with adventurous personalities
   - High-earning individuals who value experiences

3. **Clear benefit statement**: The offer is straightforward and easy to understand

# What's Not Working

1. **Limited relevance** for:
   - Young adults with student debt concerns
   - Low-income personas focused on necessities
   - Individuals with low Openness scores who prefer familiar financial products

2. **Missing urgency**: No time limit or scarcity element
3. **Lacks social proof**: No testimonials or trust indicators

# Improvement Recommendations

1. **Add urgency**: "Limited time: 0% APR for 18 months - Apply by Dec 31"
2. **Segment messaging**: Create separate ads for:
   - Young professionals (emphasize rewards)
   - Retirees (emphasize no annual fee, flexibility)
3. **Include social proof**: "Join 500,000+ satisfied cardholders"
4. **Visual enhancement**: Add imagery of travel destinations or financial freedom
5. **A/B test headlines**: Test benefit-focused vs. urgency-focused copy

# Target Audience Insights

**High-response personas** tend to be:
- Age 30-65 (financially established)
- Higher education and income levels
- High Openness (open to new financial products)
- High Conscientiousness (value financial planning)
- Active travelers or aspire to travel

**Low-response personas** tend to be:
- Under 25 or over 70
- Lower income or high debt burden
- Low Openness (prefer familiar options)
- Non-travelers or homebodies

**Recommendation**: Focus ad targeting on 30-65 age group with middle to high income,
professional occupations, and interest in travel/experiences.

================================================================================
```

---

## Performance & Costs

### Processing Speed
- **OpenAI (gpt-4o-mini)**: ~2-3 seconds per persona
- **DeepSeek**: ~1-2 seconds per persona
- **Gemini**: ~1-2 seconds per persona

### Cost Estimates (for 100 personas)
- **OpenAI (gpt-4o-mini)**: ~$0.10-0.20
- **DeepSeek**: ~$0.02-0.05
- **Gemini**: ~$0.03-0.08

**Note**: Actual costs vary by model, prompt length, and response length.

---

## Validation

The SiliconSampling personas are validated against real US population data from the IPIP Big Five dataset. Validation metrics include:

- **Mean Absolute Error (MAE)**: Measures average deviation from ground truth
- **Correlation**: Pearson correlation between synthetic and real responses
- **Kolmogorov-Smirnov Test**: Statistical similarity of distributions

Run validation:
```bash
cd SiliconSampling
python validate_all.py
```

Results are saved to `SiliconSampling/results/v2/{strategy}/{provider}/`

---

## Contributing

Contributions are welcome! Areas for improvement:

1. **Image Ad Support**: Add OCR and vision model integration
2. **Video Ad Support**: Extract keyframes and transcripts
3. **Additional LLM Providers**: Anthropic Claude, Cohere, etc.
4. **Persona Strategies**: New personality assignment methods
5. **Multi-language Support**: Non-English ads and personas
6. **A/B Testing**: Compare multiple ad variations

---

## License

MIT License - see LICENSE file for details

---

## Citation

If you use this work in research, please cite:

```bibtex
@software{wisteria_ctr_studio,
  title={Wisteria CTR Studio: Privacy-Preserving CTR Prediction with Synthetic Personas},
  author={Ezrill Lin},
  year={2025},
  url={https://github.com/Ezrill-Lin/Wisteria-CTR-Studio}
}
```

---

## Contact

For questions or collaboration opportunities:
- GitHub: [@Ezrill-Lin](https://github.com/Ezrill-Lin)
- Repository: [Wisteria-CTR-Studio](https://github.com/Ezrill-Lin/Wisteria-CTR-Studio)

---

## Acknowledgments

- SiliconSampling methodology inspired by computational social science research
- Big Five personality framework from IPIP-NEO
- WPP survey data from Vietnamese personality research
- US Census data from PUMS 2023
