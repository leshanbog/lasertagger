import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = f.readlines()

with open(sys.argv[2], 'w', encoding='utf-8') as f:
    for el in data:
        if '|' not in el:
            f.write(el)
            continue

        el = el.replace(' ##', '')
        if '##' not in el:
            f.write(el)

