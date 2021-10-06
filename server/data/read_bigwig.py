#%%
import pyBigWig
bw = pyBigWig.open("./geno_data/ENCFF158GBQ.bigWig")
#%%
bw.chroms()

#%%
bw.header()
# {'version': 4, 'nLevels': 10, 'nBasesCovered': 3095690480, 'minVal': 0, 'maxVal': 21422, 'sumData': 150199516, 'sumSquared': 73639229877}

#%%
# length of a chromosome
bw.chroms('chr2')
# stats of a range
bw.stats("chr2", 0, 300, type="max", nBins=20)

bw.intervals("chr2", 0, 300)

bw.values("chr2", 0, 300)


#%%
