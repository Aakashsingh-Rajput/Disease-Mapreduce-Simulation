import sys

first_line = True  # To skip the header

for line in sys.stdin:
    if first_line:
        first_line = False
        continue
    fields = line.strip().split(',')
    if fields:
        print(f"{fields[0]}\t1")
