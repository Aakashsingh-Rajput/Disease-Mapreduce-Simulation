import subprocess

# Read from file and simulate piping
with open("disease_data.csv", "r") as infile:
    mapper = subprocess.Popen(['python', 'mapper.py'], stdin=infile, stdout=subprocess.PIPE)
    reducer = subprocess.Popen(['python', 'reducer.py'], stdin=mapper.stdout, stdout=subprocess.PIPE)
    mapper.stdout.close()
    output = reducer.communicate()[0]

print("=== Disease Frequency Output ===")
print(output.decode())
