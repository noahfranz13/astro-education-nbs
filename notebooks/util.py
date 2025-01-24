from functools import partial
from typing import List
import zipfile
import glob
from copy import deepcopy
import os
import pickle 
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy import units as u

def read_raw_isochrone_files(dirpath="isochrones"):
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

def read_isochrone_files(fpath:str) -> List[pd.DataFrame]:
    """
    Read the pickled isochrone files

    Args:
        fpath (str) : The path to the isochrone pickle file

    Returns:
        A list of pandas dataframes with the isochrones
    """

    DATADIR = os.path.dirname(fpath)
    
    # first extract the zipfile
    with zipfile.ZipFile(fpath, 'r') as zip_ref:
        zip_ref.extractall(DATADIR)


    pkl_fpath = fpath.replace(".zip", "")
    with open(pkl_fpath, "rb") as f:
        isochrones = pickle.load(f)

    return isochrones

def plot_labels_cmd(labels):
    fig  = plt.figure()
    plt.xlabel('B-V')
    plt.ylabel('B [mag]')
    plt.xlim([-4,4])
    plt.ylim([8,20])

    for k, v in labels.items():
        if k == 'upper-left':
            plt.text(0.2, 0.8, v, transform=fig.transFigure)
        elif k == 'upper-right':
            plt.text(0.7, 0.8, v, transform=fig.transFigure)
        elif k == 'lower-left':
            plt.text(0.2, 0.2, v, transform=fig.transFigure)
        elif k == 'lower-right':
            plt.text(0.7, 0.2, v, transform=fig.transFigure)
    
    return fig

def plot_labels_physical(labels):
    fig  = plt.figure()
    plt.xlabel('B-V')
    plt.ylabel('B [mag]')
    plt.xlim([-4,4])
    plt.ylim([8,20])

    for k, v in labels.items():
        if k == 'upper-left':
            plt.text(0.2, 0.8, v, transform=fig.transFigure)
        elif k == 'upper-right':
            plt.text(0.65, 0.8, v, transform=fig.transFigure)
        elif k == 'lower-left':
            plt.text(0.2, 0.2, v, transform=fig.transFigure)
        elif k == 'lower-right':
            plt.text(0.65, 0.2, v, transform=fig.transFigure)


    return fig


def plot_labels_stars(labels):
    fig  = plt.figure()
    plt.xlabel('B-V')
    plt.ylabel('B [mag]')
    plt.xlim([-4,4])
    plt.ylim([8,20])

    for k, v in labels.items():
        if k == 'upper-left':
            plt.text(0.2, 0.8, v, transform=fig.transFigure)
        elif k == 'upper-right':
            plt.text(0.65, 0.8, v, transform=fig.transFigure)
        elif k == 'lower-left':
            plt.text(0.2, 0.2, v, transform=fig.transFigure)
        elif k == 'lower-right':
            plt.text(0.65, 0.2, v, transform=fig.transFigure)
    
    return fig

def show_cluster_data(ax=None, *args, **kwargs):
    """
    Display the cluster data. All arguments are passed to the cluster_filter
    """

    res = cluster_filter(*args, **kwargs)

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_xlabel('Color (SDSS g-r)')
    ax.set_ylabel('Magnitude (SDSS g)')
    
    # need to correct the Gaia magnitude and convert to SDSS to compare with isochrone
    # See Table 4.7 https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
    g_bp_minus_g_rp = res.phot_bp_mean_mag - res.phot_rp_mean_mag
    correction = lambda x, c1, c2, c3, c4 : c1 + c2*x + c3*x**2 + c4*x**3
    r = res.phot_g_mean_mag - correction(g_bp_minus_g_rp, -0.12879, 0.24662, -0.027464, -0.049465)
    g = res.phot_g_mean_mag - correction(g_bp_minus_g_rp, 0.13518, -0.46245, -0.25171, 0.021349)
    
    ax.plot(
        g-r,
        g, 
        marker='.',
        linestyle='none',
        color='k'
    )

    ax.set_ylim(9, 16)
    ax.set_xlim(-0.2, 1.6)
    
    ax.invert_yaxis()
    
    return ax

def write_label(x, y, label, ax, color='red', **kwargs):

    if x == "CHANGE ME" or y == "CHANGE ME":
        return #they still need to replace them

    if isinstance(x, str):
        x = float(x)

    if isinstance(y, str):
        y = float(y)

    ax.text(x, y, label, color=color, fontsize=18)
    # ax.plot([x,x+0.2], [y,y+0.3], linestyle='-', color=color)

def distance_mod(M, d):
    return M + 5*np.log10(d/10)
    
def show_isochrone(age=14.5, distance=10, Fe_H=0.06, Y=0.2682, ax=None, **kwargs):
    """
    Fe_H = 0.06 is from this paper https://arxiv.org/abs/1310.6297 
    """

    if age < 0.76 or age > 14.74:
        raise Exception("the age you input must be between 0.76 and 14.74 Gyr!")
    
    isochrones = read_isochrone_files(os.path.join(os.getcwd(), "isochrones.pkl.zip"))
    isochrones = pd.concat(isochrones)
    isochrones.Y = isochrones.Y.astype(float)
    isochrones.Fe_H = isochrones.Fe_H.astype(float)

    # filter based on the input
    # we will also take some default metallicity and helium abundance (~ solar Y)
    # just for simplicity
    isochrones = isochrones[
        np.isclose(isochrones.age, age, atol=0.24) *
        (isochrones.Fe_H == Fe_H) *
        (isochrones.Y == Y)
    ]

    if len(isochrones) == 0:
        raise Exception("No isochrone matching your input parameters! Please try again or ask an instructor!")

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_xlabel('Color (SDSS g-r)')
    ax.set_ylabel('Magnitude (SDSS g)')
    
    g = distance_mod(isochrones.g.astype(float), distance)
    r = distance_mod(isochrones.r.astype(float), distance)
    
    ax.plot(g-r, g, linewidth=3, **kwargs)
    ax.invert_yaxis()
    
    return ax

show_correct_isochrone = partial(show_isochrone, distance=800, age=6, color='red', label='D=800 pc\nAge=6 Gyr')
show_close_isochrone = partial(show_isochrone, distance=400, age=6, color='blue', label='D=400 pc\nAge=6 Gyr')
show_old_isochrone = partial(show_isochrone, distance=800, age=14, color='green', label='D=800 pc\nAge=14 Gyr')

def cluster_filter(
        cone_search_radius:float=2,
        minimum_radial_velocity:float=34-13.6, #km/s
        maximum_radial_velocity:float=34+13.6, #km/s
        minimum_parallax:float=1.15-0.12, 
        maximum_parallax:float=1.15+0.12,
        proper_motion_delta:float=0.7,
        proper_motion_ra:float=-11,
        proper_motion_dec:float=-2.9,
        overwrite=False,
        gaia_result_path="gaia_cone_search_results.csv"
    ) -> pd.DataFrame:

    cluster_coord = SkyCoord(132.85, 11.81, unit=u.deg)
    cluster_radius = 2 * u.deg
        
    if not os.path.exists(gaia_result_path) or overwrite:
        from astroquery.gaia import Gaia
        Gaia.ROW_LIMIT = 100_000_000 # set this limit very high so we don't miss things
        print("Querying Gaia, this may take a bit...")
        
        job = Gaia.cone_search(cluster_coord, radius=cluster_radius)
        res = job.get_results().to_pandas()
        print(res)
 
        res.to_csv(gaia_result_path)

    else:
        res = pd.read_csv(gaia_result_path)
    
    # cone search cut
    res_coord = SkyCoord(res.ra, res.dec, unit="deg")
    where_cluster = np.where(res_coord.separation(cluster_coord) < (cone_search_radius*u.deg))
    res = res.iloc[where_cluster]
    
    # radial velocity cut
    res = res[(res.radial_velocity > minimum_radial_velocity) * (res.radial_velocity < maximum_radial_velocity)]
    
    # parallax cut
    res = res[(res.parallax > minimum_parallax) * (res.parallax < maximum_parallax)]
    
    # proper motion cut
    dmu = proper_motion_delta # mas yr^-1; same units as gaia uses
    mu = np.array([proper_motion_ra, proper_motion_dec]) # mas yr^-1; same units as gaia uses
    
    proper_motion_distance = np.sqrt(np.sum((res[["pmra", "pmdec"]] - mu)**2, axis=1))
    res = res[proper_motion_distance <= dmu]
    
    if overwrite:
        res.to_csv("cleaned-M67-data.csv")

    return res
