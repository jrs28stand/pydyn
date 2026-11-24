import json, sys
seen = set(json.load(open('/tmp/seen_kinds.json'))) if len(sys.argv) > 1 else set()
new = {}
for p in 'ABCD':
    for k, v in json.load(open(f'soak/state/{p}.json'))['kinds'].items():
        new[f'{p}:{k}'] = v
print({k: v for k, v in new.items() if k not in seen})
json.dump(list(new), open('/tmp/seen_kinds.json', 'w'))
