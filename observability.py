import os
import numpy as np
import pandas as pd
from astropy import units as u
import matplotlib.pyplot as plt
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord as coord

#Function to caluclate angular diameter with color relationships
def angdiam_sbr_vk(V,K):
    c0 = 0.529
    c1 = 0.062
    
    theta = 10**(c0+c1*(V-K)-0.2*K)
    
    
    return(theta)
    #return(print("Angular diameter [mas]:", theta))

def angdiam_sbr_vh(V,H):
    c0 = 0.538
    c1 = 0.074
    theta = 10**(c0+c1*(V-H)-0.2*H)
    return(theta)

def filter_by_chara_limits(df):
    dec = df['Dec (deg)']
    idx_dec = []
    for i in range(len(dec)):
        if dec[i] > -20:
            idx_dec.append(i)
    new_df = df.loc[idx_dec].reset_index(drop = True)
    #new_df.drop('Unnamed: 0', axis = 1)
    return new_df

def search_for_ids(id_name, df):
    list_of_names = df['IDS'].tolist()
    ids = []

    for entry in list_of_names:
        parts = entry.split("|")
        matches = [p for p in parts if id_name in p]

        if not matches:
            ids.append(" ")
        else:
            # If multiple parts match in this one entry, dedupe them but keep order
            seen = set()
            unique_matches = []
            for m in matches:
                if m not in seen:
                    unique_matches.append(m)
                    seen.add(m)

            # Choose what to store for this row:
            # - if you want exactly ONE value per input row, pick the first match:
            ids.append(unique_matches[0])

            # If you instead want to store *all* matches for the row, you'd need a nested list,
            # which would not be a "list of strings" anymore:
            # ids.append(unique_matches)

    return ids

def normalize_string(input_str, map_dict):
    inst_name = map_dict.get(input_str)

    return inst_name

def get_targets(df, inst_name, verbose = False):
    inst_map = {
        "PAVO": "PAVO",
        "pavo": "PAVO",
        "P": "PAVO",
        "p": "PAVO",
        "Pavo": "PAVO",
        "MIRCX": "MIRCX",
        "MIRC-X": "MIRCX",
        "M": "MIRCX",
        "m": "MIRCX",
        "mircx": "MIRCX",
        "mirc-x": "MIRCX",
        "Mircx": "MIRCX",
        "Mirc-x": "MIRCX",
        "MYSTIC": "MYSTIC",
        "mystic": "MYSTIC",
        "My": "MYSTIC",
        "my": "MYSTIC",
        "Mystic": "MYSTIC",
        "Classic": "Classic",
        "classic": "Classic",
        "C": "Classic",
        "c": "Classic",
        "CLASSIC": "Classic",
        "SPICA": "SPICA",
        "spica": "SPICA",
        "Spica": "SPICA",
        "S": "SPICA",
        "s": "SPICA"
    }

    instname_check = normalize_string(inst_name, inst_map)

    if instname_check is None:
        print("Instrument name is incorrect. Please check spelling.")
    else:
    
        v = df['V']
        k = df['K']
        h = df['H']
        r = df['R']
        ang_diam = []
        for i in range(len(v)):
            if k[i] != np.nan:
                ad = angdiam_sbr_vk(v[i], k[i])
            else:
                ad = angdiam_sbr_vh(v[i], h[i])
        
            ang_diam.append(ad)
    
        df['Ang. diam [mas]'] = ang_diam

        if inst_name == 'PAVO':
            PAVO_idx = []
            for ii in range(len(v)):
                if r[ii] == np.nan:
                    if ang_diam[ii] >= 0.25 and v[ii] < 7.0:
                        PAVO_idx.append(ii)
                else:
                    if ang_diam[ii] >= 0.25 and r[ii] < 7.5:
                        PAVO_idx.append(ii)

            pavo_df = df.loc[PAVO_idx].reset_index(drop = True)
            if verbose:
                print("There are ", len(pavo_df), "out of ", len(df), " targets observable with PAVO.")
            return pavo_df

        if inst_name == 'MIRCX':
            MIRCX_idx = []
            for ii in range(len(v)):
                if ang_diam[ii] >= 0.35 and h[ii] <= 7.5:
                    MIRCX_idx.append(ii)
    
            mircx_df = df.loc[MIRCX_idx].reset_index(drop = True)
            if verbose:
                print("There are ", len(mircx_df), "out of ", len(df), " targets observable with MIRC-X.")
            return mircx_df

        if inst_name == 'MYSTIC':
            MYSTIC_idx = []
            for ii in range(len(v)):
                if ang_diam[ii] >= 0.35 and k[ii] <= 6.5:
                    MYSTIC_idx.append(ii)
    
            mystic_df = df.loc[MYSTIC_idx].reset_index(drop = True)
            if verbose:
                print("There are ", len(mystic_df), "out of ", len(df), " targets observable with MYSTIC.")
            return mystic_df    
    
        if inst_name == 'Classic':
            classic_idx = []
            for ii in range(len(v)):
                if ang_diam[ii] >= 0.50 and h[ii] <= 7.0:
                    classic_idx.append(ii)
    
            classic_df = df.loc[classic_idx].reset_index(drop = True)
            if verbose:
                print("There are ", len(classic_df), "out of ", len(df), " targets observable with Classic.")
            return classic_df
    
        if inst_name == 'SPICA':
            spica_idx = []
            for ii in range(len(v)):
                if r[ii] == np.nan:
                    if ang_diam[ii] >= 0.25 and v[ii] < 5.0:
                        spica_idx.append(ii)
                else:
                    if ang_diam[ii] >= 0.25 and r[ii] < 5.5:
                        spica_idx.append(ii)
    
            spica_df = df.loc[spica_idx].reset_index(drop = True)
            if verbose:
                print("There are ", len(spica_df), "out of ", len(df), " targets observable with SPICA.")
            return spica_df

def extract_vmags(star_id_list, simbad_df, verbose = False):
    v_oja = Vizier(catalog="II/182/ubv", columns=["*", "Vmag"], row_limit=1)
    v_tycho = Vizier(catalog="I/259/tyc2", columns=["*", "BTmag", "VTmag"], row_limit=1)
    
    vmags = []  # one per ID, use np.nan if missing
    
    count_o = 0
    count_tycho = 0
    count_sim = 0
    count_none = 0
    
    for i, obj_id in enumerate(star_id_list):
        vmag = np.nan
        if verbose:
            print("Star #:", i)
        # 1) Oja
        res_oja = v_oja.query_object(obj_id)
        if res_oja and len(res_oja) > 0 and len(res_oja[0]) > 0:
            if verbose:
                print("Found", obj_id, "in Oja")
            vmag = res_oja[0]["V"][0]
            count_o += 1
    
        else:
            # 2) Tycho-2
            res_tyc = v_tycho.query_object(obj_id)
            if res_tyc and len(res_tyc) > 0 and len(res_tyc[0]) > 0:
                if verbose:
                    print("Found", obj_id, "in TYCHO")
                vt = res_tyc[0]["VTmag"][0]
                bt = res_tyc[0]["BTmag"][0]
                vmag = vt - 0.090 * (bt - vt)   # your transform
                count_tycho += 1
    
            else:
                # 3) SIMBAD (from your prebuilt dataframe)
                vsim = simbad_df.loc[i, "FLUX_V"]
                if pd.notna(vsim):
                    if verbose:
                        print(obj_id, "found in Simbad")
                    vmag = vsim
                    count_sim += 1
                else:
                    if verbose:
                        print(obj_id, "not found")
                    count_none += 1
    
        vmags.append(vmag)
    if verbose:
        print("Found ", count_o , "out of ", len(star_id_list), " stars in the Oja 1987-1993 catalog.")
        print("Found ", count_tycho , "out of ", len(star_id_list), " stars in the Tycho2 catalog.")
        print("Found ", count_sim , "out of ", len(star_id_list), " stars from Simbad.")
        print("V magnitudes for ", count_o , "out of ", len(star_id_list), " stars could not be found.")
    return vmags

def get_coords(simbad_df):
    ra_hms = simbad_df['RA']
    dec_dms = simbad_df['DEC']
    ra_deg = []
    dec_deg = []
    for i in range(len(ra_hms)):
        x = coord(ra=ra_hms[i], dec = dec_dms[i], unit = (u.hourangle, u.deg)).to_string('decimal')
        new_x = x.split()
        ra_deg.append(float(new_x[0]))
        dec_deg.append(float(new_x[1]))

    return ra_hms, dec_dms, ra_deg, dec_deg

def extract_star_information(star_id_list, tic_flag = False, toi_flag = False, gaia_flag = False, hd_flag = False, 
                            gj_flag = False, main_flag = False, hip_flag = False, tmass_flag = False, verbose = False):
    if verbose:
        print("Querying SIMBAD for targets")
    Simbad.add_votable_fields('main_id','ids','coordinates', 'sptype', 'distance', 'parallax', 'flux(V)', 'flux(R)', 'flux(H)', 'flux(K)')
    simbad_df = Simbad.query_objects(star_id_list).to_pandas()
    
    if verbose:
        print("Finished SIMBAD query")
    rahms, decdms, radeg, decdeg = get_coords(simbad_df)
    sptype = simbad_df['SP_TYPE'].tolist()
    hmag = simbad_df['FLUX_H'].tolist()
    kmag = simbad_df['FLUX_K'].tolist()
    rmag = simbad_df['FLUX_R'].tolist()
    plx = simbad_df['PLX_VALUE'].tolist()
    dist = simbad_df['Distance_distance'].tolist()
    
    if verbose:
        print("Extracting V magnitudes")
        
    vmag = extract_vmags(star_id_list, simbad_df, verbose = verbose)
    
    if verbose:
        print("Finished extracting V magnitudes")
    
    data = list(zip(star_id_list, rahms, decdms, radeg, decdeg, sptype, plx, dist, vmag, rmag, hmag, kmag))
    columns = ['Input ID', 'RA', 'Dec', 'RA (deg)', 'Dec (deg)', 'SpType', 'Plx', 'Distance', 'V', 'R', 'H', 'K']
    target_df = pd.DataFrame(data, columns = columns)

    if tic_flag:
        tics = search_for_ids('TIC', simbad_df)
        target_df['TIC ID'] = tics
    if toi_flag:
        tois = search_for_ids('TOI', simbad_df)
        target_df['TOI'] = tois
    if gaia_flag:
        gaias = search_for_ids('Gaia DR3', simbad_df)
        target_df['Gaia DR3'] = gaias
    if hd_flag:
        hds = search_for_ids('HD', simbad_df)
        target_df['HD'] = hds
    if gj_flag:
        gjs = search_for_ids('GJ', simbad_df)
        target_df['GJ'] = gjs
    if main_flag:
        mains = simbad_df['MAIN_ID'].tolist()
        target_df['MAIN ID'] = mains
    if hip_flag:
        hips = search_for_ids('HIP', simbad_df)
        target_df['HIP'] = hips
    if tmass_flag:
        tmass = search_for_ids('2MASS', simbad_df)
        target_df['2MASS'] = tmass

    return target_df

def time_of_year(df):
    ra = df['RA']
    month = []
    for i in range(len(ra)):
        ra_h = int(ra[i].split()[0])

        if ra_h >= 0 and ra_h < 2:
            month.append('Sep.')
        elif ra_h >= 2 and ra_h < 4:
            month.append('Oct.')
        elif ra_h >= 4 and ra_h < 6:
            month.append('Nov.')
        elif ra_h >= 6 and ra_h < 8:
            month.append('Dec.')
        elif ra_h >= 8 and ra_h < 10:
            month.append('Jan.')
        elif ra_h >= 10 and ra_h < 12:
            month.append('Feb.')
        elif ra_h >= 12 and ra_h < 14:
            month.append('Mar.')
        elif ra_h >= 14 and ra_h < 16:
            month.append('Apr.')
        elif ra_h >= 16 and ra_h < 18:
            month.append('May')
        elif ra_h >= 18 and ra_h < 20:
            month.append('Jun.')
        elif ra_h >= 20 and ra_h < 22:
            month.append('Jul.')
        else :
            month.append('Aug.')
            
    df['Month best observable'] = month
    return df

def month_to_ra_range(month):
    month_dict = {
        "Sept.": "0-2",
        "Oct.": "2-4",
        "Nov.": "4-6",
        "Dec.": "6-8",
        "Jan.": "8-10",
        "Feb.": "10-12",
        "Mar.": "12-14", 
        "Apr.": "14-16",
        "May": "16-18",
        "Jun.": "18-20",
        "Jul.": "20-22",
        "Aug.": "22-24"
    }
    ra_range = month_dict.get(month)
    rasplit = ra_range.split("-")
    ra_b = int(rasplit[0])
    ra_e = int(rasplit[1])
    return ra_b, ra_e

def time_of_year_single(ra):
    for i in range(len(ra)):
        ra_h = int(ra.split()[0])

        if ra_h >= 0 and ra_h < 2:
            month = 'September'
        elif ra_h >= 2 and ra_h < 4:
            month = 'October'
        elif ra_h >= 4 and ra_h < 6:
            month = 'November'
        elif ra_h >= 6 and ra_h < 8:
            month = 'December'
        elif ra_h >= 8 and ra_h < 10:
            month = 'January'
        elif ra_h >= 10 and ra_h < 12:
            month = 'February' 
        elif ra_h >= 12 and ra_h < 14:
            month = 'March' 
        elif ra_h >= 14 and ra_h < 16:
            month = 'April'
        elif ra_h >= 16 and ra_h < 18:
            month = 'May'
        elif ra_h >= 18 and ra_h < 20:
            month = 'June'
        elif ra_h >= 20 and ra_h < 22:
            month = 'July'
        else :
            month = 'August'
    print('Target with RA of', ra, 'is best observable in the month of', month)

def obsplot(df, inst, savefig = None):
    starnames = df['MAIN ID']
    RA = df['RA']
    month = df['Month best observable']
    fig, ax = plt.subplots()
    for i in range(len(month)):
        rb, _ = month_to_ra_range(month[i])
        bars = ax.barh(starnames[i], width = 2, left = rb, label = month[i])
        
        ax.bar_label(bars, labels=[str(month[i])], label_type='center', color='white', fontsize = 10)
    
    labels = np.arange(0, 25, 1)
    label_locs = np.arange(0, 25, 1)
    ax.set_xlim(0,24)
    ax.set_xticks(label_locs)
    ax.set_xticklabels(labels)
    ax.set_axisbelow(True)                # grid behind bars
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.ylabel('Star ID')
    plt.xlabel('RA (hour)')
    plt.title(f'Observable {inst} targets')

    if savefig:
        f.savefig(savefig, bbox_inches='tight')
    return fig, ax