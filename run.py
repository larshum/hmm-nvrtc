import common as c
from hmm_nvrtc import HMM
import numpy as np
import os
import sys
import time

def flatten(xss):
    return [x for xs in xss for x in xs]

model_path = os.getenv("MODEL_PATH")
signals_path = os.getenv("SIGNALS_PATH")

if model_path:
    tables, signals = c.read_kmer_inputs_trellis(model_path, signals_path)
else:
    tables, signals = c.get_weather_inputs_trellis(signals_path)

tables['gammaInv'] = np.log(1.0 - np.exp(tables['gamma']))
tables['trans1'] = flatten(tables['trans1'].T)
tables['outputProb'] = flatten(tables['outputProb'].T)
tables['initialProb'] = flatten(tables['initialProb'])

hmm = HMM(tables)

t0 = time.time()
p = hmm.forward(signals)
t1 = time.time()

t2 = time.time()
out = hmm.viterbi(signals, 5)
t3 = time.time()

outc = "ACGT"
for i, seq in enumerate(out):
    print(f"Seq #{i+1} ({p[i]})")
    col = 0
    for s in seq:
        if s % 16 == 0:
            if col == 80:
                print("")
                col = 0
            col += 1
            print(f"{outc[s // 16 % 4]}", end="")
    print("")
print(t1-t0)
print(t3-t2)
