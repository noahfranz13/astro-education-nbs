import glob
from copy import deepcopy
import os
import pandas as pd

def read_isochrone_files(dirpath="isochrones"):
    isochrones = glob.glob(os.path.join(dirpath, "SDSSugriz/*.SDSSugriz"))

    cleaned_isochrones = []
    for isofile in isochrones:
        with open(isofile, 'r') as f:
            data = f.readlines()

        hdr = data[0:6]
        data = data[6:]

        hdr_vals = [val.strip() for val in hdr[3].split(' ') if val not in {'#', ''}]
        mix_len, Y, Z, Zeff, Fe_H, a_Fe = hdr_vals

        isochrones_in_this_file = []
        base_iso = dict(
            idx = [],
            M = [],
            LogTeff = [],
            LogG = [],
            LogL = [],
            u = [],
            g = [],
            r = [],
            i = [],
            z = [],
            age = [],
            mix_len = [],
            Y = [],
            Z = [],
            Zeff = [],
            Fe_H = [],
            a_Fe = [],
        )
        iso = deepcopy(base_iso)
        age = None

        for j in range(len(data)):
            if data[j] == '\n':
                continue

            if '#AGE' in data[j]:
                isochrones_in_this_file.append(pd.DataFrame(iso))

                age = float(data[j].split(" EEPS=")[0].split("AGE=")[1].strip())
                iso = deepcopy(base_iso)
                continue

            if '#' == data[j][0]:
                # this line is a header
                continue


            line = data[j]
            goodline = [val for val in line.strip().split(' ') if len(val) > 0]

            if len(goodline) != 10:
                print(goodline)
                continue 

            for val, key in zip(goodline, iso.keys()):
                # if key == 'age': continue
                iso[key].append(val)

            iso['age'].append(age)
            iso['mix_len'].append(mix_len)
            iso['Y'].append(Y)
            iso['Z'].append(Z)
            iso['Zeff'].append(Zeff)
            iso['Fe_H'].append(Fe_H)
            iso['a_Fe'].append(a_Fe)

        cleaned_isochrones += isochrones_in_this_file
    return cleaned_isochrones[1:]
