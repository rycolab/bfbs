import sys

file = sys.argv[1]
scores_file = file[:file.index(".out")] + ".pops"

tosave = []
with open(file, 'r') as f:
	lines = f.read().split('\n')
	lines = [l for l in lines if "ID" in l ]

	while lines:
		l = lines.pop(0)
		if "Next sentence (ID: " in l:
			if not lines:
				break
			d = lines.pop(0)
			while lines and not "Next sentence (ID: " in d and not "INFO: Stats (ID:" in d:
				d = lines.pop(0)
			if not "INFO: Stats (ID:" in d:
				tosave.append('')
				continue
			newl = d[d.index("num_expansions=") + len("num_expansions="):]
			newl = newl[:newl.index(" ")]
			tosave.append(int(newl))


print(len(tosave))

with open(scores_file, 'w') as f:
	#f.write('\n'.join(tosave))
	f.write(str(tosave))
