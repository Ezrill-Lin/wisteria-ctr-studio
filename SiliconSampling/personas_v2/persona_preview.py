"""
Persona Preview Tool

Preview v2 personas from JSONL files with formatted JSON output.

Usage:
    # Preview 3 random personas from random strategy
    python persona_preview.py --strategy random --count 3
    
    # Preview specific persona by ID
    python persona_preview.py --strategy random --id 12345
    
    # Save preview to file
    python persona_preview.py --strategy ipip --count 5 --output preview.json
"""

import json
import random
import argparse
from pathlib import Path


def load_personas(strategy):
    """Load all personas from JSONL file"""
    personas_file = Path(f"{strategy}_matching") / f"personas_{strategy}_v2.jsonl"
    
    if not personas_file.exists():
        raise FileNotFoundError(f"Personas file not found: {personas_file}")
    
    personas = []
    with open(personas_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                persona = json.loads(line.strip())
                personas.append(persona)
            except json.JSONDecodeError:
                continue
    
    return personas


def preview_personas(strategy, count=None, persona_id=None, output_file=None):
    """
    Preview personas with formatted JSON output
    
    Args:
        strategy: Strategy name (random/ipip/wpp)
        count: Number of random personas to preview
        persona_id: Specific persona ID to preview
        output_file: Optional output file path
    """
    
    print(f"\n{'='*80}")
    print(f"V2 PERSONA PREVIEW: {strategy.upper()}")
    print(f"{'='*80}\n")
    
    # Load all personas
    all_personas = load_personas(strategy)
    print(f"Total personas available: {len(all_personas):,}\n")
    
    # Select personas to preview
    if persona_id is not None:
        # Find specific persona by ID
        selected = [p for p in all_personas if p.get('id') == persona_id]
        if not selected:
            print(f"❌ Persona with ID {persona_id} not found")
            return
        print(f"Previewing persona ID: {persona_id}\n")
    elif count:
        # Random sample
        selected = random.sample(all_personas, min(count, len(all_personas)))
        print(f"Previewing {len(selected)} random personas\n")
    else:
        # Default: 1 random persona
        selected = random.sample(all_personas, 1)
        print(f"Previewing 1 random persona\n")
    
    # Format output
    preview_data = {
        'strategy': strategy,
        'total_personas': len(all_personas),
        'preview_count': len(selected),
        'personas': selected
    }
    
    # Pretty print JSON
    formatted_json = json.dumps(preview_data, indent=2, ensure_ascii=False)
    
    # Output to console
    print(formatted_json)
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_json)
        print(f"\n✓ Preview saved to: {output_path}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Preview v2 personas with formatted JSON output")
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['random', 'ipip', 'wpp'],
                       help='Persona strategy to preview')
    parser.add_argument('--count', type=int, default=None,
                       help='Number of random personas to preview (default: 1)')
    parser.add_argument('--id', type=int, default=None,
                       help='Specific persona ID to preview')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.count is not None and args.id is not None:
        print("❌ Error: Cannot specify both --count and --id")
        return
    
    preview_personas(
        strategy=args.strategy,
        count=args.count,
        persona_id=args.id,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
