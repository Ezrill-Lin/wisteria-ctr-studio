"""
Utility functions for synthetic agent and validation

This module contains helper functions for:
- Loading personas and questions
- Creating prompts for LLM responses
- Saving and analyzing responses
- Data processing utilities
"""

import json
from pathlib import Path
import pandas as pd


def detect_api_provider(model: str) -> str:
    """Auto-detect API provider from model name prefix
    Args:
        model: Model name (e.g., 'gemini-2.5-flash', 'deepseek-chat', 'gpt-4o')
    Returns:
        API provider: 'gemini', 'deepseek', or 'openai'
    """
    model_lower = model.lower()
    
    if model_lower.startswith('gemini-'):
        return 'gemini'
    elif model_lower.startswith('deepseek'):
        return 'deepseek'
    else:
        # Default to OpenAI for gpt-*, o1-*, o3-*, o4-*, and unknown models
        return 'openai'



def load_personas(strategy, sample_size=None, base_dir="personas", version="v1"):
    """Load generated personas from JSONL file
    Args:
        strategy: Strategy name (random/ipip/wpp)
        sample_size: Number of personas to load (None = all)
        base_dir: Base directory for personas
        version: Persona version ('v1' or 'v2')
    Returns:
        List of persona dictionaries
    """
    # Determine file path based on version
    if version == "v2":
        base_dir = "personas_v2"
        persona_file = Path(base_dir) / f"{strategy}_matching" / f"personas_{strategy}_v2.jsonl"
    else:
        persona_file = Path(base_dir) / f"{strategy}_matching" / f"personas_{strategy}.jsonl"
    
    if not persona_file.exists():
        raise FileNotFoundError(
            f"Persona file not found: {persona_file}\n"
            f"Please run: cd {base_dir} && python generate_personas{('_v2' if version == 'v2' else '')}.py --strategy {strategy}"
        )
    
    print(f"Loading personas from {persona_file}...")
    personas = []
    
    with open(persona_file, 'r', encoding='utf-8') as f:
        for line in f:
            persona = json.loads(line.strip())
            personas.append(persona)
            if sample_size and len(personas) >= sample_size:
                break
    
    print(f"✓ Loaded {len(personas):,} {version} personas")
    return personas


def load_questions(questions_file):
    """Load questions from JSON file
    Args:
        questions_file: Path to questions JSON file
    Returns:
        List of question dictionaries
    """
    questions_path = Path(questions_file)
    
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"✓ Loaded {len(questions)} questions from {questions_file}")
    return questions


def create_prompt(persona, questions):
    """Create system message and user prompt for LLM to answer questions as a persona
    Args:
        persona: Persona dictionary with demographics, ocean_scores, persona_description (or None for no-persona mode)
                 V2 personas also include behavioral_tendencies and self_schema
        questions: List of questions
    Returns:
        Tuple of (system_message, user_prompt)
    """
    # Build system message based on persona mode
    if persona is None or 'persona_description' not in persona:
        # No-persona mode: generic system message
        system_message = "You are answering personality test questions. Answer honestly and naturally."
    else:
        # V1 personas: Simple structure with persona_description only
        if 'behavioral_tendencies' not in persona and 'self_schema' not in persona:
            system_message = f"""<persona>
{persona['persona_description']}
</persona>

<instruction>
Embody this character completely. Answer all questions as this person would, staying true to the persona described above.
</instruction>"""
        
        # V2 personas: Enhanced structure with behavioral_tendencies and self_schema
        else:
            system_message = f"""<persona>
{persona['persona_description']}
</persona>"""
            
            if 'behavioral_tendencies' in persona:
                system_message += "\n\n<behavioral_tendencies>\n"
                for key, value in persona['behavioral_tendencies'].items():
                    system_message += f"- {key.replace('_', ' ').title()}: {value}\n"
                system_message += "</behavioral_tendencies>"
            
            if 'self_schema' in persona:
                system_message += "\n\n<core_beliefs>\n"
                for belief in persona['self_schema']:
                    system_message += f"- {belief}\n"
                system_message += "</core_beliefs>"
            
            system_message += "\n\n<instruction>\nEmbody this character completely. Answer all questions as this person would, staying true to the persona, behaviors, and beliefs described above.\n</instruction>"
    
    # Build user prompt
    user_prompt = """Rate the following statements on a five point scale where 1=Disagree, 3=Neutral, 5=Agree:
"""
    for i, q in enumerate(questions, 1):
        user_prompt += f"{i}. {q['question']}\n"
    
    user_prompt += """\nProvide your response as a JSON array: [{"statement_id":1,"answer":3},{"statement_id":2,"answer":4},...]
Include only the array, no additional text."""
    
    return system_message, user_prompt


def save_responses_batch(responses, output_file, append=True):
    """Save batch of responses to JSONL file
    Args:
        responses: List of response dictionaries
        output_file: Path to output file
        append: If True, append to existing file; if False, overwrite
    """
    mode = 'a' if append else 'w'
    
    with open(output_file, mode, encoding='utf-8') as f:
        for response in responses:
            f.write(json.dumps(response, ensure_ascii=False) + '\n')


def load_responses(responses_file):
    """Load all responses from JSONL file
    Args:
        responses_file: Path to JSONL responses file
    Returns:
        List of response dictionaries
    """
    responses = []
    with open(responses_file, 'r', encoding='utf-8') as f:
        for line in f:
            responses.append(json.loads(line.strip()))
    
    return responses


def analyze_responses(responses_file):
    """Analyze responses and print summary statistics
    Args:
        responses_file: Path to JSONL responses file
    Returns:
        DataFrame with analysis
    """
    print(f"\nAnalyzing responses from {responses_file}...")
    
    responses = load_responses(responses_file)
    
    # Create analysis DataFrame
    analysis_data = []
    for resp in responses:
        row = {
            'persona_id': resp['persona_id'],
            'response_count': len(resp['responses']) if isinstance(resp['responses'], list) else 0,
            'status': 'SUCCESS' if isinstance(resp['responses'], list) and len(resp['responses']) > 0 else 'FAILED'
        }
        
        analysis_data.append(row)
    
    df = pd.DataFrame(analysis_data)
    
    print(f"\nSummary:")
    print(f"  Total responses: {len(df)}")
    print(f"  Successful responses: {(df['status'] == 'SUCCESS').sum()}")
    print(f"  Failed responses: {(df['status'] == 'FAILED').sum()}")
    print(f"  Average response count: {df['response_count'].mean():.1f}")
    
    return df


def clean_failed_responses(responses_file):
    """Remove failed responses from JSONL file
    Args:
        responses_file: Path to JSONL responses file
    Returns:
        Tuple of (valid_count, removed_count)
    """
    responses = load_responses(responses_file)
    
    # Filter out failed responses
    valid_responses = [
        resp for resp in responses 
        if isinstance(resp['responses'], list) and len(resp['responses']) > 0
    ]
    
    removed_count = len(responses) - len(valid_responses)
    
    if removed_count > 0:
        # Overwrite file with only valid responses
        with open(responses_file, 'w', encoding='utf-8') as f:
            for resp in valid_responses:
                f.write(json.dumps(resp, ensure_ascii=False) + '\n')
        
        print(f"✓ Cleaned {responses_file}")
        print(f"  Removed: {removed_count} failed responses")
        print(f"  Remaining: {len(valid_responses)} valid responses")
    else:
        print(f"✓ No failed responses to clean")
    
    return len(valid_responses), removed_count


def clean_duplicated_responses(responses_file):
    """Remove duplicated responses (keep first occurrence of each persona_id)
    Args:
        responses_file: Path to JSONL responses file
    Returns:
        Tuple of (unique_count, removed_count)
    """
    responses = load_responses(responses_file)
    
    # Track seen persona_ids and keep only first occurrence
    seen_ids = set()
    unique_responses = []
    
    for resp in responses:
        persona_id = resp.get('persona_id')
        if persona_id not in seen_ids:
            seen_ids.add(persona_id)
            unique_responses.append(resp)
    
    removed_count = len(responses) - len(unique_responses)
    
    if removed_count > 0:
        # Overwrite file with only unique responses
        with open(responses_file, 'w', encoding='utf-8') as f:
            for resp in unique_responses:
                f.write(json.dumps(resp, ensure_ascii=False) + '\n')
        
        print(f"✓ Cleaned duplicates from {responses_file}")
        print(f"  Removed: {removed_count} duplicate responses")
        print(f"  Remaining: {len(unique_responses)} unique responses")
    else:
        print(f"✓ No duplicate responses to clean")
    
    return len(unique_responses), removed_count


def load_ground_truth_data(filepath="test/us_ground_truth.csv"):
    """Load ground truth test dataset for validation
    Args:
        filepath: Path to us_ground_truth.csv (US respondents only)
    Returns:
        DataFrame with ground truth responses
    """
    df = pd.read_csv(filepath)
    print(f"✓ Loaded ground truth data: {len(df):,} records")
    print(f"  Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
    
    return df


def extract_test_columns(df):
    """Extract test item columns from DataFrame
    Args:
        df: DataFrame with test responses
    Returns:
        DataFrame with only test item columns (E1-E10, N1-N10, A1-A10, C1-C10, O1-O10)
    """
    ipip_cols = []
    for trait in ['E', 'N', 'A', 'C', 'O']:
        for i in range(1, 11):
            col = f"{trait}{i}"
            if col in df.columns:
                ipip_cols.append(col)
    
    return df[ipip_cols]
