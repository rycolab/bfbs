import sys

file = sys.argv[1]
out = sys.argv[2]
with open(file, 'r') as f:
	lines = [line.strip().replace("@@ ", "") for line in f]
	with open(out, 'w') as g:
		g.write('\n'.join(lines))
		g.write('\n')
	
