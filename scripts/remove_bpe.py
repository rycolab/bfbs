with open("data/wmt14.en-fr.newstest2014/newtest.bpe.fr", 'r') as f:
	lines = [line.strip().replace("@@ ", "") for line in f]
	with open("test.fr", 'w') as g:
		g.write('\n'.join(lines))
		g.write('\n')
	
