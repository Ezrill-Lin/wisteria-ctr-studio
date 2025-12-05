import json
from pathlib import Path

# Check v2 personas for duplicates
personas_file = Path('personas_v2/random_matching/personas_random_v2.jsonl')

personas = []
with open(personas_file, 'r', encoding='utf-8') as f:
    for line in f:
        personas.append(json.loads(line))

ids = [p['id'] for p in personas]
total = len(ids)
unique = len(set(ids))
duplicate_count = total - unique

print(f"{'='*60}")
print(f"DUPLICATE CHECK - V2 Random Personas")
print(f"{'='*60}")
print(f"Total personas:     {total:,}")
print(f"Unique IDs:         {unique:,}")
print(f"Duplicates:         {duplicate_count:,}")

if duplicate_count > 0:
    duplicates = [id for id in set(ids) if ids.count(id) > 1]
    print(f"\nDuplicate IDs found: {len(duplicates)}")
    print(f"First 10 duplicate IDs: {duplicates[:10]}")
    
    # Show details for first duplicate
    first_dup = duplicates[0]
    dup_personas = [p for p in personas if p['id'] == first_dup]
    print(f"\nExample: ID {first_dup} appears {len(dup_personas)} times")
else:
    print("\n✓ No duplicates found!")
print(f"{'='*60}")
