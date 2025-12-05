import json
from pathlib import Path

# Check all v2 persona strategies for duplicates
strategies = [
    ('random_matching', 'personas_random_v2.jsonl'),
    ('ipip_matching', 'personas_ipip_v2.jsonl'),
    ('wpp_matching', 'personas_wpp_v2.jsonl')
]

print(f"{'='*70}")
print(f"DUPLICATE CHECK - ALL V2 PERSONA STRATEGIES")
print(f"{'='*70}")

for strategy_dir, filename in strategies:
    personas_file = Path(strategy_dir) / filename
    
    if not personas_file.exists():
        print(f"\n{strategy_dir.upper().replace('_', ' ')}")
        print(f"  ⚠️  File not found: {filename}")
        continue
    
    personas = []
    with open(personas_file, 'r', encoding='utf-8') as f:
        for line in f:
            personas.append(json.loads(line))
    
    ids = [p['id'] for p in personas]
    total = len(ids)
    unique = len(set(ids))
    duplicate_count = total - unique
    
    print(f"\n{strategy_dir.upper().replace('_MATCHING', '').replace('_', ' ')}")
    print(f"  File:           {filename}")
    print(f"  Total personas: {total:,}")
    print(f"  Unique IDs:     {unique:,}")
    print(f"  Duplicates:     {duplicate_count:,}")
    
    if duplicate_count > 0:
        duplicates = [id for id in set(ids) if ids.count(id) > 1]
        print(f"  ❌ Found {len(duplicates)} duplicate IDs")
        print(f"  First 5: {duplicates[:5]}")
    else:
        print(f"  ✓ No duplicates found!")

print(f"\n{'='*70}")
