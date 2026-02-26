import numpy as np
import pandas as pd

def load_and_format_sharks_gals(file_path, cols = ['ra', 'dec', 'id_galaxy_sky', 'id_group_sky', 
                                              'type', 'zcos', 'zobs', 'mstars_bulge', 'mstars_disk', 
                                              'mgas_disk', 'mgas_bulge', 'mvir_hosthalo', 'mvir_subhalo', 
                                              'id_fof', 'sfr_disk', 'sfr_burst', 'total_ab_dust_u_VST', 
                                              'total_ab_dust_g_VST', 'total_ab_dust_r_VST', 'total_ab_dust_i_VST', 
                                              'total_ab_dust_Z_VISTA', 'total_ab_dust_Y_VISTA', 'total_ab_dust_J_VISTA', 
                                              'total_ab_dust_H_VISTA', 'total_ab_dust_K_VISTA'], bcg_on = 'Z_VISTA'):
    """
    Load and format the SHARKS galaxy catalog.
    """
    if bcg_on not in ['Z_VISTA', 'Stellar Mass']:
        raise ValueError("Invalid value for bcg_on. Must be 'Z_VISTA' or 'Stellar Mass'.")
    
    gals = pd.read_parquet(file_path, columns = cols)
    # Filer for valid stellar mass
    gals['stellar_mass'] = (gals['mstars_disk'] + gals['mstars_bulge'])/0.67
    gals['log_stellar_mass'] = np.log10((gals['mstars_disk'] + gals['mstars_bulge'])/0.67)
    gals['log_sfr_total'] = np.log10((gals['sfr_disk'] + gals['sfr_burst']) * 1e-9/0.67 + 1e-20)
    gals['log_sSFR'] = np.log10((gals['sfr_disk'] * 1e-9 + gals['sfr_burst'] * 1e-9) / (gals['mstars_disk'] + gals['mstars_bulge']) / 0.67 + 1e-20)
    mask = (gals['log_stellar_mass'] > 8) & (gals['total_ab_dust_Z_VISTA'] > -99)
    gals = gals[mask].reset_index(drop=True)


    group_col = "id_fof"  # absolute r-band magnitude (lower = brighter)
    stellar_mass_col = "stellar_mass"
    mass_col  = "mvir_hosthalo"
    host_id_col = "id_group_sky"  # e.g. "host_halo_id"

    mag_col = "total_ab_dust_Z_VISTA"

    # -----------------------------
    # 1) BCG assignment
    #   - id_fof == -1 : by default BCG
    #   - otherwise: brightest (min Mr) per FoF group
    # -----------------------------
    gals["is_bcg"] = False

    ungrouped = gals[group_col] == -1
    gals.loc[ungrouped, "is_bcg"] = True

    valid_grouped = (
        gals[group_col].notna()
        & (gals[group_col] != -1)
        & gals[mag_col].notna()
        & gals[stellar_mass_col].notna()
    )

    # Pick brightest in r-band: minimum absolute magnitude
    if bcg_on == 'Z_VISTA':
        bcg_idx = gals.loc[valid_grouped].groupby(group_col)[mag_col].idxmin()
        gals.loc[bcg_idx, "is_bcg"] = True
        

    if bcg_on == 'Stellar Mass':
        bcg_idx = gals.loc[valid_grouped].groupby(group_col)[stellar_mass_col].idxmax()
        gals.loc[bcg_idx, "is_bcg"] = True

    #print(gals["is_bcg"].value_counts())

    # -----------------------------
    # 2) Broadcast BCG properties
    #   - For grouped galaxies: copy from the group BCG
    #   - For id_fof == -1: keep self values (each is its own BCG)
    # -----------------------------
    for col in ["ra", "dec", "zobs", mag_col, stellar_mass_col]:
        gals[f"{col}_bcg"] = gals[col]  # default: self (covers id_fof == -1 nicely)

    # Build mapping ONLY for real groups (unique by construction)
    bcg_rows = gals.loc[gals["is_bcg"] & ~ungrouped, [group_col, "ra", "dec", "zobs", mag_col, stellar_mass_col]]

    ra_map   = bcg_rows.set_index(group_col)["ra"]
    dec_map  = bcg_rows.set_index(group_col)["dec"]
    zobs_map = bcg_rows.set_index(group_col)["zobs"]
    mag_map  = bcg_rows.set_index(group_col)[mag_col]
    stellar_mass_map = bcg_rows.set_index(group_col)[stellar_mass_col]


    grouped = ~ungrouped & gals[group_col].notna()
    gals.loc[grouped, "ra_bcg"]   = gals.loc[grouped, group_col].map(ra_map)
    gals.loc[grouped, "dec_bcg"]  = gals.loc[grouped, group_col].map(dec_map)
    gals.loc[grouped, "zobs_bcg"] = gals.loc[grouped, group_col].map(zobs_map)
    gals.loc[grouped, f"{mag_col}_bcg"] = gals.loc[grouped, group_col].map(mag_map)

    #gals.loc[grouped, f"log_stellar_mass_bcg"] = gals.loc[grouped, group_col].map(stellar_mass_map)
    gals["stellar_mass_bcg"] = gals["stellar_mass"]  # default self for ungrouped
    gals.loc[grouped, "stellar_mass_bcg"] = gals.loc[grouped, group_col].map(stellar_mass_map)

    # broadcast log stellar mass of BCG
    gals["log_stellar_mass_bcg"] = np.log10(gals["stellar_mass_bcg"] + 1e-30)

    # -----------------------------
    # 3) FoF halo mass = sum of UNIQUE host halos within each FoF
    #   - Only for id_fof != -1
    #   - For id_fof == -1: per-galaxy default (no aggregation across all -1 rows)
    # -----------------------------
    gals["fof_halo_mass"] = np.nan

    grouped = gals[group_col].notna() & (gals[group_col] != -1)
    ungrouped = gals[group_col] == -1

    if host_id_col is not None and host_id_col in gals.columns:
        # sum one host mass per unique host halo ID within each FoF group
        fof_mass = (
            gals.loc[grouped]
                .dropna(subset=[group_col, host_id_col, mass_col])
                .drop_duplicates(subset=[group_col, host_id_col])
                .groupby(group_col)[mass_col]
                .sum()
        )
    else:
        raise ValueError(f"Host ID column '{host_id_col}' not found in the DataFrame. Cannot compute FoF halo mass without unique host identification.")

    gals.loc[grouped, "fof_halo_mass"] = gals.loc[grouped, group_col].map(fof_mass)

    # For id_fof == -1, treat each galaxy as its own "FoF": use its own host mass
    gals.loc[ungrouped, "fof_halo_mass"] = gals.loc[ungrouped, mass_col]

    gals["log_fof_halo_mass"] = np.log10(gals["fof_halo_mass"])

    gals['is_red'] = gals['log_sSFR'] < -11

    # Find Luminosity for each galaxy
    M_sun = 4.51
    gals['L'] = 10**(-0.4 * (gals['total_ab_dust_Z_VISTA']))

    # Now for each group find the total luminosity of the group, and then take the log of that
    gals['group_L'] = gals.groupby('id_fof')['L'].transform('sum')
    # if id fof is -1, then it's not in a group, so we can just set the group luminosity to be the same as the galaxy luminosity
    gals.loc[gals['id_fof'] == -1, 'group_L'] = gals.loc[gals['id_fof'] == -1, 'L']

    gals['log_group_L'] = np.log10(gals['group_L'])

    gals['n_group_fof'] = gals.groupby('id_fof')['id_fof'].transform('count')

    gals['group_stellar_mass'] = gals.groupby('id_fof')['stellar_mass'].transform('sum')
    gals.loc[gals['id_fof'] == -1, 'group_stellar_mass'] = gals.loc[gals['id_fof'] == -1, 'stellar_mass']


    gals['log_group_stellar_mass'] = np.log10(gals['group_stellar_mass'])
    # if id fof is -1, then it's not in a group, so we can just set the number of galaxies in the group to be 1
    gals.loc[gals['id_fof'] == -1, 'n_group_fof'] = 1

    # get group properties as well groupby group id fof
    groups = gals.groupby('id_fof')[['ra_bcg', 'dec_bcg', 'zobs_bcg', f'{mag_col}_bcg', 'fof_halo_mass', 'log_fof_halo_mass', 'group_L', 'log_group_L', 'n_group_fof', 'group_stellar_mass', 'log_group_stellar_mass']].first().reset_index()

    return gals, groups


def load_and_format_sharks_groups(file_path, cols = None):
    
    if cols is None:
        groups = pd.read_parquet(file_path)
    else:
        groups = pd.read_parquet(file_path, columns = cols)

    return groups
