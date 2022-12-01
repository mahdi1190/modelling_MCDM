import numpy as np
import pandas as pd

df = pd.read_csv('input.csv', skip_blank_lines=False)

elf=[]
calories=0
for x in df["col"]:
    if pd.isnull(x):
        elf.append(calories)
        calories=0
    else:
        calories=calories+float(x)
listy = sorted(elf, reverse=True)[:3]
print(sum(listy))

