# PersonaGenerationAgent

Generate realistic persona narratives by combining synthetic demographics with personality profiles using async LLM calls.

## Features

- **Async architecture** - 9x faster than synchronous approaches
- **Big Five personality integration** - OCEAN trait-based profiles
- **Configurable concurrency** - Optimize for your API tier
- **Structured output** - JSON with demographics, personality, and narrative descriptions

## Installation

```bash
pip install pandas openai tqdm pyarrow
```

Set your OpenAI API key:
```bash
# Windows
$env:OPENAI_API_KEY="your-api-key"

# Linux/Mac
export OPENAI_API_KEY="your-api-key"
```

## Usage

### Command Line (Recommended)

```bash
python main.py --num-personas 100 --max-concurrent 30
```

**Options:**
- `--demographics`: Path to demographic data (default: `data/synthetic_demographics_1m.parquet`)
- `--personalities`: Path to personality profiles (default: `PersonalitySamplingAgent/Personality_profiles/profiles.json`)
- `--num-personas`: Number of personas to generate (default: 100)
- `--output`: Output JSON file (default: `generated_personas.json`)
- `--model`: OpenAI model (default: `gpt-4o-mini`)
- `--max-concurrent`: Concurrent API requests (default: 30)
- `--seed`: Random seed for reproducibility (default: 42)

### Python Module

```python
from PersonaGenerationAgent import PersonaGenerationAgent
import asyncio

async def generate():
    agent = PersonaGenerationAgent(model="gpt-4o-mini")
    demographics = agent.load_demographic_data("data/synthetic_demographics_1m.parquet")
    personalities = agent.load_personality_profiles("PersonalitySamplingAgent/Personality_profiles/profiles.json")
    
    personas = await agent.generate_personas(
        demographic_data=demographics,
        personality_profiles=personalities,
        num_personas=100,
        max_concurrent=30
    )
    return personas

personas = asyncio.run(generate())
```

## Output Format

```json
{
  "id": 1,
  "demographics": {
    "age": 34,
    "gender": "Male",
    "state": "California/CA",
    "educational_attainment": "Bachelor's degree",
    "occupation": "Software Developer"
  },
  "personality": {
    "scores": {
      "openness": 80,
      "conscientiousness": 60,
      "extraversion": 70,
      "agreeableness": 50,
      "neuroticism": 30
    }
  },
  "description": "You are a 34-year-old software developer living in California..."
}
```

## Performance

| Personas | Time  | Throughput |
|----------|-------|------------|
| 10       | ~4s   | 2.5/s      |
| 100      | ~42s  | 2.4/s      |
| 1,000    | ~37s  | 27/s       |

**Concurrency Guidelines** (based on OpenAI API tier):
- **Free tier** (3 RPM): `max_concurrent=2`
- **Tier 1** (500 RPM): `max_concurrent=8-10`
- **Tier 2** (5,000 RPM): `max_concurrent=20`
- **Tier 3+** (10,000 RPM): `max_concurrent=30-50`

## Cost Estimation

Using `gpt-4o-mini` (~250 tokens/persona):
- 100 personas: ~$0.025
- 1,000 personas: ~$0.25

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API key not found | Set `OPENAI_API_KEY` environment variable |
| Rate limit errors | Reduce `max_concurrent` to match API tier |
| Data file missing | Run `data.ipynb` to generate demographics |
