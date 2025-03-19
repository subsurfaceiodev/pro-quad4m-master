#!/usr/bin/env python

import json
import jsonschema
import numpy as np
import copy
import argparse
from sys import exit as sys_exit
from sys import argv as sys_argv
from os import path as os_path
from os import mkdir as os_mkdir
import matplotlib as mpl; mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors as clr
from matplotlib import cm as mcm
import matplotlib.tri as mtri
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
from shapely.geometry import Point, MultiPoint
from sys import stdout as sys_stdout
from shutil import copyfile as shutil_copyfile


# #############
# CUSTOM FUNCT
# #############
from sys import path as sys_path
sys_path.append(os_path.join(*[os_path.dirname(sys_argv[0]), 'lib']))
from def_QUAD4M import lprint
from def_QUAD4M import strata_plot
from def_QUAD4M import add_tolerance
from def_QUAD4M import set_unions
from def_QUAD4M import set_intersections
from def_QUAD4M import find_nodes
from def_QUAD4M import find_intervals
from def_QUAD4M import all_intervals
from def_QUAD4M import adjust_intervals
from def_QUAD4M import adjust_line
from def_QUAD4M import adjust_all_intervals
from def_QUAD4M import generate_composite_ab
from def_QUAD4M import retrieve_poi_s
from def_QUAD4M import retrieve_pois_s
from def_QUAD4M import retrieve_pois_ndx
from def_QUAD4M import val_uncentanty
from def_QUAD4M import add_element
from def_QUAD4M import add_element_triang
from def_QUAD4M import find_np
from def_QUAD4M import mesh_ds
from def_QUAD4M import redefine_elements
from def_QUAD4M import add_fake_elements
from def_QUAD4M import element_lines_numpy
from def_QUAD4M import plot_border_inner
from def_QUAD4M import plot_border
from def_QUAD4M import search_id
from def_QUAD4M import add_markers
from def_QUAD4M import finalize_plot
from def_QUAD4M import plot_all_borders
from def_QUAD4M import pre_plot
from def_QUAD4M import post_plot
from def_QUAD4M import par_plot
from def_QUAD4M import write_soi
from def_QUAD4M import create_soi
from def_QUAD4M import is_valid_file
from def_QUAD4M import is_valid_dir
from def_QUAD4M import common_def
from def_QUAD4M import resp_spectra_acc


# #############
# PARAMETERS
# #############

#~ ##############################

common_def_dict = common_def()

p = argparse.ArgumentParser(description='description: Generate files "MDL.Q4I" and "SG.DAT" (along with acceleration time-histories "WFX.ACC" and "WFY.ACC") designed as input to execute QUAD4M finite-elements computer program. Produce some graphical-outputs in SVG format (Scalable Vector Graphics) which can be useful to check if QUAD4M input-files are correctly generated. File "borders.txt", that is due for QUAD4M post-processor execution, is also generated.')

p.add_argument("json", action="store", type=is_valid_file, help=common_def_dict['json'])
p.add_argument("xacc", action="store", type=is_valid_file, help="single-column horizontal acceleration time-history")
p.add_argument("yacc", action="store", type=is_valid_file, help="single-column vertical acceleration time-history")
#~ p.add_argument("-d", "--directory", action="store", dest="out_folder", default='var', type=is_valid_dir, help=common_def_dict['out_folder'])
p.add_argument("-v", "--version", action="version", version=common_def_dict["version"], help=common_def_dict["version_h"])

opts = p.parse_args()

#~ ##############################

infile = opts.json
inwaveX = opts.xacc
inwaveY = opts.yacc
#~ main_outfold = opts.out_folder
main_outfold = os_path.join(*[os_path.dirname(sys_argv[0]), 'var'])

#~ ##############################

schemafile = os_path.join(*[os_path.dirname(sys_argv[0]), 'lib', 'schema.json'])
elemfile = os_path.join(*[os_path.dirname(sys_argv[0]), 'lib', 'json_elements.json'])
tolerance_s = 5.0 # tolerance in metres to enlarge stratum in x- and y-directions

# JSON elements
with open(elemfile, 'r') as c_file: json_elem_dict = json.load(c_file)

# #############
# ERRORS CHECK
# #############

# JSON-input
with open(infile, 'r') as input_json:
    try: input_dict = json.load(input_json)
    except:
        print 'ERROR: file "' + infile + '" does not contain a valid JSON object (e.g. check it through any JSON-validator on the web)."'
        sys_exit(1)
# JSON-schema
with open(schemafile, 'r') as schema_json: dict_schema = json.load(schema_json)

#~ jsonschema.validate(instance=input_dict,schema=dict_schema)

# Lazily report all errors in the instance
c_errors = []
#~ try:
c_val = jsonschema.Draft4Validator(dict_schema)
for c_error in sorted(c_val.iter_errors(input_dict), key=str):
    if map(str,list(c_error.absolute_path)):
        c_error_absolute_path = \
        ' on instance [' + ']['.join(map(str,list(c_error.absolute_path))) + ']'
    else: c_error_absolute_path = ''
    if len(json.dumps(c_error.schema)) < 300:
        c_error_schema = ': ' + json.dumps(c_error.schema)
    else: c_error_schema = ''
    c_error_str = 'ERROR: ' + str(c_error.message) + c_error_absolute_path.replace('[','["').replace(']','"]') + ' (DETAIL: failed validating "' + c_error.validator + '" in schema ["' + '"]["'.join(c_error.schema_path) + '"]' + c_error_schema + ')'
    c_errors.append(c_error_str.replace('\r','').replace('\n','').replace(" u'"," '").replace("'",'"'))
#~ except jsonschema.ValidationError as c_exception:
    #~ print c_exception.message

try:
    nodal_points = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_nps"]]
    strata = int(input_dict[json_elem_dict["modp"]][json_elem_dict["modp_str"]])

    for stratum_ndx in map(lambda x:x+1, range(strata)):
        c_stratum = json_elem_dict["prem"] + "%03d" % stratum_ndx
        if not c_stratum in input_dict:
            first_stratum = json_elem_dict["prem"] + "%03d" % 1
            last_stratum = json_elem_dict["prem"] + "%03d" % strata
            c_error_str = 'ERROR: "' + c_stratum + '" is a required property (DETAIL: expected properties are "' + first_stratum + '"..."' + last_stratum + '", since ' + str(strata) + ' strata have been declared on instance ["' + json_elem_dict["modp"] + '"]["' + json_elem_dict["modp_str"] + '"])'
            c_errors.append(c_error_str.replace('\r','').replace('\n',''))
        else:
            st_nod = input_dict[c_stratum][json_elem_dict["prem_nod"]]
            if not int(st_nod[0]) == int(st_nod[-1]): st_nod.append(int(st_nod[0]))
            for k_nod in st_nod:
                sk_nod = str(k_nod)
                if sk_nod not in nodal_points:
                    c_error_str = 'ERROR: node "' + sk_nod + '" not found (DETAIL: value "' + sk_nod + '" given on instance ["' + c_stratum + '"]["' + json_elem_dict["prem_nod"] + '"] must exists as a key in ["' + c_stratum + '"]["' + json_elem_dict["modp_nps"] + '"])'
                    c_errors.append(c_error_str.replace('\r','').replace('\n',''))
            
            model_G_Gmax_XL_tf = input_dict[c_stratum][json_elem_dict["prem_mGX"]][json_elem_dict["prem_mGX_tf"]]
            if not input_dict[c_stratum][json_elem_dict["prem_xld"]][json_elem_dict["etc_va"]]:
                if model_G_Gmax_XL_tf:
                    # set placeholder (to be overwritten)
                    input_dict[c_stratum][json_elem_dict["prem_xld"]][json_elem_dict["etc_va"]] = 0.99
                else:
                    c_error_str = 'ERROR: in case "null" is given on instance ["' + c_stratum + '"]["' + json_elem_dict["prem_xld"] + '"]["' + json_elem_dict["etc_va"] + '"], "true" must be given on ["' + c_stratum + '"]["' + json_elem_dict["prem_mGX"] + '"]["' + json_elem_dict["prem_mGX_tf"] + '"])'
                    c_errors.append(c_error_str.replace('\r','').replace('\n',''))                

            if not model_G_Gmax_XL_tf:
                if input_dict[c_stratum][json_elem_dict["prem_soi"]][json_elem_dict["prem_soi_xl"]]:
                    XL0 = float(input_dict[c_stratum][json_elem_dict["prem_soi"]][json_elem_dict["prem_soi_xl"]][0][1])
                    XL = float(input_dict[c_stratum][json_elem_dict["prem_xld"]][json_elem_dict["etc_va"]] * 100.0)
                    if not XL0 == XL:
                        c_error_str = 'ERROR: damping values non-consistent (DETAIL: assuming low-deformation (i.e. low-shear-strain), the first damping value "' + str(float(XL0)) + '" given on instance ["' + c_stratum + '"]["' + json_elem_dict["prem_soi"] + '"]["' + json_elem_dict["prem_soi_xl"] + '"] must be equal to the value "' + str(float(XL)) + '" given on instance ["' + c_stratum + '"]["' + json_elem_dict["prem_xld"] + '"]["' + json_elem_dict["etc_va"] + '"])'
                        c_errors.append(c_error_str.replace('\r','').replace('\n',''))
                if input_dict[c_stratum][json_elem_dict["prem_soi"]][json_elem_dict["prem_soi_gg"]]:
                    G_Gmax0 = float(input_dict[c_stratum][json_elem_dict["prem_soi"]][json_elem_dict["prem_soi_gg"]][0][1])
                    if not G_Gmax0 == 1.0:
                        c_error_str = 'ERROR: shear modulus (G) values non-consistent (DETAIL: assuming low-deformation (i.e. low-shear-strain), the first G/Gmax value "' + str(float(G_Gmax0)) + '" given on instance ["' + c_stratum + '"]["' + json_elem_dict["prem_soi"] + '"]["' + json_elem_dict["prem_soi_gg"] + '"] must be equal to 1.0)'
                        c_errors.append(c_error_str.replace('\r','').replace('\n',''))

            min_value_check_fields = [json_elem_dict["prem_xld"], json_elem_dict["prem_gmx"], json_elem_dict["prem_dns"], json_elem_dict["prem_por"]]
            for check_field in min_value_check_fields:
                # log-normal
                if input_dict[c_stratum][check_field][json_elem_dict["etc_ln"]] \
                and input_dict[c_stratum][check_field][json_elem_dict["etc_uc"]] is not None \
                and (input_dict[c_stratum][check_field][json_elem_dict["etc_uc"]] < 1.0 or input_dict[c_stratum][check_field][json_elem_dict["etc_uc"]] > 5.0):
                    c_error_str = 'ERROR: in case of log-normal distribution is selected, "uncertanty" must be greater or equal to 1.0 but less than 5.0 (DETAIL: modify instance ["' + c_stratum + '"]["' + check_field + '"])'
                    c_errors.append(c_error_str.replace('\r','').replace('\n',''))
except: pass

if c_errors: 
    print 'NOTE: ' + str(len(c_errors)) + ' errors found on JSON-file "' + infile + '"'
    print '---'
    for c_error_str in c_errors:
        print c_error_str
        print '---'
    sys_exit(1)


# #############
# some out-files
# #############

spec_outfold = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_jbf"]]
outfold = os_path.join(*[main_outfold, spec_outfold])

if not os_path.isdir(outfold): os_mkdir(outfold)

jmodel_nam = os_path.join(*[outfold, 'used_model.json'])
waveX_nam = os_path.join(*[outfold, 'used_time-history.xacc'])
waveY_nam = os_path.join(*[outfold, 'used_time-history.yacc'])
strata_nam = os_path.join(*[outfold, 'strata_plot.svg'])
nmodel_nam = os_path.join(*[outfold, 'model_nodal_points.svg'])
st_me_nam = os_path.join(*[outfold, 'initial_grid_plot.svg'])
mesh_nam = os_path.join(*[outfold, 'mesh_plot.svg'])
par_DENS = os_path.join(*[outfold, 'plot_DENS.svg'])
par_PO = os_path.join(*[outfold, 'plot_PO.svg'])
par_GMX = os_path.join(*[outfold, 'plot_GMX.svg'])
par_G = os_path.join(*[outfold, 'plot_G.svg'])
par_XL = os_path.join(*[outfold, 'plot_XL.svg'])
wav_nam = os_path.join(*[outfold, 'input_waveforms.svg'])
sg_nam = os_path.join(*[outfold, 'SG_'])

shutil_copyfile(infile, jmodel_nam)
shutil_copyfile(inwaveX, waveX_nam)
shutil_copyfile(inwaveY, waveY_nam)


# #############
# CHECK PREV
# #############

if os_path.isfile(strata_nam):
    print 'WARNING: output file "' + strata_nam + '" already exists'
    ans_YN = raw_input("Overwrite all output files? [Y|N] [N]: ")
    if not ans_YN: sys_exit(1)
    elif ans_YN[0] in ['y', 'Y']: pass
    else: sys_exit(1)

# #############
# STRATA PLOT
# #############

print ""
lprint("results folder: " + outfold)
lprint("strata plot")

save_png = input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_png"]]
if save_png:
    png_outfold = os_path.join(*[outfold, 'PNG_images'])
    if not os_path.isdir(png_outfold + ''): os_mkdir(png_outfold)
else: png_outfold = None

line_thicknesses_scale = input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_sca"]][json_elem_dict["pltp_sca_lt"]]
fonts_scale = input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_sca"]][json_elem_dict["pltp_sca_fo"]]
symbols_scale = input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_sca"]][json_elem_dict["pltp_sca_sy"]]

norm = clr.Normalize(vmin=1.0, vmax=float(strata))
c_linewidth = 1.0 * line_thicknesses_scale
ax_linewidth = 0.5 * line_thicknesses_scale
ax_fontsize = 8.0 * fonts_scale
leg_fontsize = 7.0 * fonts_scale
ax_markersize = 1.0 * symbols_scale
legend_ncol = int(np.ceil(strata / (5.0/fonts_scale)))

if input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_bws"]]: cmap = mcm.get_cmap('Greys')
else: cmap = mcm.get_cmap('jet')

plt.clf()
plt.ylabel('[m]', fontsize=ax_fontsize)
plt.xlabel('[m]', fontsize=ax_fontsize)

plt_ax = plt.gca()
for stratum_ndx in map(lambda x:x+1, range(strata)):
    c_stratum = json_elem_dict["prem"] + "%03d" % stratum_ndx
    c_rgba = cmap(norm(float(stratum_ndx)))
    c_rgb = list(c_rgba)[:-1]
    c_sty = input_dict[c_stratum][json_elem_dict["prem_nod"]]
    c_st = [[float(nodal_points[str(a_st)][0]), float(nodal_points[str(a_st)][1])] for a_st in c_sty]
    c_na = input_dict[c_stratum][json_elem_dict["prem_nam"]]
    plt_ax = strata_plot(plt_ax, c_st, c_rgb, c_rgb, c_na, 0.5, c_linewidth)

plt.title('Strata', fontsize=ax_fontsize)
plt, plt_ax = finalize_plot(plt, plt_ax, ax_linewidth, ax_fontsize)
plt_ax.legend(fontsize=leg_fontsize, loc=4, ncol=legend_ncol)
plt_ax.tick_params(labelsize=ax_fontsize*0.85)
plt_ax.annotate('', xy=(0.02, 0.55), xycoords='figure fraction', xytext=(0.02, 0.45), arrowprops=dict(arrowstyle="-", color='w'))
plt_ax.annotate('', xy=(0.98, 0.55), xycoords='figure fraction', xytext=(0.98, 0.45), arrowprops=dict(arrowstyle="-", color='w'))
figure = plt.gcf()
figure.savefig(strata_nam, format='svg', bbox_inches='tight')
if save_png: figure.savefig((strata_nam[:-3] + 'png').replace(outfold, png_outfold), format='png', dpi=600, bbox_inches='tight')
#~ plt.show()

# #############
# NODES PLOT
# #############

for ii, (a_key, a_val) in enumerate(nodal_points.items()):
    tbbox = dict(facecolor='white', edgecolor='None', alpha=0.5)
    var_np = (float(a_val[0]), float(a_val[1]))
    pippo = plt_ax.annotate(str(a_key), (var_np[0]+2.,var_np[1]+7.), fontsize=ax_fontsize*5.0/8.0, bbox=tbbox, verticalalignment='center')
    pippo.get_bbox_patch().set_boxstyle("square, pad=0.1")
    if ii == 1: plt_ax.plot(var_np[0], var_np[1], 'o', linewidth=0.0, markersize=ax_markersize, color='r', label = 'node')
    else: plt_ax.plot(var_np[0], var_np[1], 'o', linewidth=0.0, markersize=ax_markersize, color='r', label = None)
plt_ax.legend(fontsize=leg_fontsize, loc=7, ncol=legend_ncol)
figure = plt.gcf()
figure.savefig(nmodel_nam, format='svg', bbox_inches='tight')
if save_png: figure.savefig((nmodel_nam[:-3] + 'png').replace(outfold, png_outfold), format='png', dpi=600, bbox_inches='tight')


# #################
# READING WAVEFORMS 
# #################

wave_hrx = int(input_dict[json_elem_dict["wave"]][json_elem_dict["wave_hrx"]])
waveX = np.loadtxt(fname=inwaveX, skiprows=wave_hrx)
hdrx_lines = []
c_fx = open(inwaveX, 'r')
for kk in range(wave_hrx): hdrx_lines.append(c_fx.readline())
c_fx.close()

wave_hry = int(input_dict[json_elem_dict["wave"]][json_elem_dict["wave_hry"]])
waveY = np.loadtxt(fname=inwaveY, skiprows=wave_hry)
hdry_lines = []
c_fy = open(inwaveY, 'r')
for kk in range(wave_hry): hdry_lines.append(c_fy.readline())
c_fy.close()

lenwave = min([len(waveX), len(waveX)])
wave_ftl = int(input_dict[json_elem_dict["wave"]][json_elem_dict["wave_ftl"]])
wave_ltl = int(input_dict[json_elem_dict["wave"]][json_elem_dict["wave_ltl"]])
wave_fts = int(input_dict[json_elem_dict["wave"]][json_elem_dict["wave_fts"]])
wave_lts = int(input_dict[json_elem_dict["wave"]][json_elem_dict["wave_lts"]])

if wave_ftl >= wave_ltl:
    print 'ERROR: first time-step to be considered must be lower than last! Please chose different values for "' + json_elem_dict["wave"] + '" JSON-input block. Consider, in particular, to change "' + json_elem_dict["wave_ftl"] + '" and "' + json_elem_dict["wave_ltl"] + '" input values.'
    sys_exit(1)
if wave_fts >= wave_lts:
    print 'ERROR: first time-step to be considered must be lower than last! Please chose different values for "' + json_elem_dict["wave"] + '" JSON-input block. Consider, in particular, to change "' + json_elem_dict["wave_fts"] + '" and "' + json_elem_dict["wave_lts"] + '" input values.'
    sys_exit(1)
if wave_ltl > lenwave:
    print 'ERROR: last time-step to be considered must be lower than length of whole waveforms! Please chose different values for "' + json_elem_dict["wave"] + '" JSON-input block. Consider, in particular, to change "' + json_elem_dict["wave_ltl"] + '" input value.'
    sys_exit(1)
if wave_lts > lenwave:
    print 'ERROR: last time-step to be considered must be lower than length of whole waveforms! Please chose different values for "' + json_elem_dict["wave"] + '" JSON-input block. Consider, in particular, to change "' + json_elem_dict["wave_lts"] + '" input value.'
    sys_exit(1)

# ##################################################
# ################ GENERATE MESH ################### 
# ##################################################

lprint("mesh generation")

# max element shape factor (H-elem/V-elem)
w_factor = float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_esf"]]\
[json_elem_dict["etc_hv"]])
# max element shape factor (V-elem/H-elem)
h_factor = float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_esf"]]\
[json_elem_dict["etc_vh"]])
# dist_threshold
dist_threshold = float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_ran"]]\
[json_elem_dict["mesh_ran_dt"]])
t_itera = int(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_ran"]]\
[json_elem_dict["mesh_ran_at"]])
y_l_def_coe = float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_ran"]]\
[json_elem_dict["mesh_ran_vc"]])
x_l_def_coe = float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_ran"]]\
[json_elem_dict["mesh_ran_hc"]])
elem_sf = float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_ran"]]\
[json_elem_dict["mesh_ran_sf"]])
rearr_nod = input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_ran"]]\
[json_elem_dict["mesh_ran_rn"]]
c_mesh_tre = input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_tre"]]
c_fm = float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_mfe"]])
c_mesh_uig = input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_uig"]]

lprint("max frequency (Kuhlemeyer et al., 1973) [Hz]: %4.1f" % c_fm)
lprint("max H/V shape factor: %4.1f" % w_factor)
lprint("max V/H shape factor: %4.1f" % h_factor)
lprint("use triangular elements: %s" % c_mesh_tre)
lprint("rearrange nodes: %s" % rearr_nod)

width_s = []
height_s = []
weight_s = {}
min_x = max_x = float(nodal_points["1"][0])
min_y = max_y = float(nodal_points["1"][1])
for stratum_ndx in map(lambda x:x+1, range(strata)):
    c_stratum = json_elem_dict["prem"] + "%03d" % stratum_ndx
    weight_s[stratum_ndx] = float(input_dict[c_stratum][json_elem_dict["prem_dns"]][json_elem_dict["etc_va"]])
    c_sty = input_dict[c_stratum][json_elem_dict["prem_nod"]]
    c_st = [[float(nodal_points[str(a_st)][0]), float(nodal_points[str(a_st)][1])] for a_st in c_sty]
    c_x = list(zip(*c_st)[0])
    c_y = list(zip(*c_st)[1])
    c_G0_v = float(input_dict[c_stratum][json_elem_dict["prem_gmx"]][json_elem_dict["etc_va"]])
    c_G0_u = input_dict[c_stratum][json_elem_dict["prem_gmx"]][json_elem_dict["etc_uc"]]
    if c_G0_u is None: c_G0 = c_G0_v
    else:
        if input_dict[c_stratum][json_elem_dict["prem_gmx"]][json_elem_dict["etc_ln"]]:
            c_G0 = c_G0_v/(float(c_G0_u)**2.0) # case log-normal
        else: c_G0 = c_G0_v - 2.0*float(c_G0_u) # case normal (Gaussian)
    c_G1_v = float(input_dict[c_stratum][json_elem_dict["prem_gmp"]][json_elem_dict["prem_gmp_pc"]])
    c_G1 = c_G0*(c_G1_v/100.)
    c_de = float(input_dict[c_stratum][json_elem_dict["prem_dns"]][json_elem_dict["etc_va"]])
    c_ro = c_de/1000./9.806
    c_Vs = (c_G1/c_ro)**0.5
    c_hmax = c_Vs/8./c_fm
    # aggravate factor for IDW on G0
    #~ if input_dict[json_elem_dict["modp"]][json_elem_dict["modp_gxw"]]:
        #~ c_hmax = c_hmax / 1.30
    # aggravate factor to adjust x- and y-coordinates of nodes
    if rearr_nod: c_hmax = c_hmax * elem_sf
    c_wmax = c_hmax * w_factor
    min_c_x = min(c_x)
    max_c_x = max(c_x)
    min_c_y = min(c_y)
    max_c_y = max(c_y)
    width_s.append([c_wmax, min_c_x, max_c_x, c_stratum])
    height_s.append([c_hmax, min_c_y, max_c_y, c_stratum])
    min_x = min([min_x, min_c_x])
    max_x = max([max_x, max_c_x])
    min_y = min([min_y, min_c_y])
    max_y = max([max_y, max_c_y])


def sort_first(val): return val[0]
height_s.sort(key = sort_first)
width_s.sort(key = sort_first)


max_y_thr = max_y + dist_threshold
height_s = add_tolerance(height_s, tolerance_s, min_y, max_y_thr)
width_s = add_tolerance(width_s, tolerance_s, min_x, max_x)
# add "dist_threshold" to the top of the grid
height_s[-1][2] = max_y_thr + dist_threshold


height_s_uig = copy.deepcopy(height_s)
width_s_uig = copy.deepcopy(width_s)
if c_mesh_uig:
    # if json-input option 'uniform_initial_grid' is True 
    # refill height_s_uig and width_s_uig with lowest height and width ([0][0])
    for k_uig in range(1,len(height_s_uig)): 
        height_s_uig[k_uig][0] = height_s_uig[0][0]
        width_s_uig[k_uig][0] = width_s_uig[0][0]


height_st = find_intervals(height_s_uig, min_y, max_y_thr, strata, width_s[0][0] * h_factor)
width_st = find_intervals(width_s_uig, min_x, max_x, strata, width_s[0][0])

y_lines = all_intervals(height_st)
x_lines = all_intervals(width_st)


# #############
# GRID PLOT
# #############

plt.cla()
plt.ylabel('[m]', fontsize=ax_fontsize)
plt.xlabel('[m]', fontsize=ax_fontsize)
plt_ax = plt.gca()
for stratum_ndx in map(lambda x:x+1, range(strata)):
    c_stratum = json_elem_dict["prem"] + "%03d" % stratum_ndx
    c_rgba = cmap(norm(float(stratum_ndx)))
    c_rgb = list(c_rgba)[:-1]
    c_sty = input_dict[c_stratum][json_elem_dict["prem_nod"]]
    c_st = [[float(nodal_points[str(a_st)][0]), float(nodal_points[str(a_st)][1])] for a_st in c_sty]
    c_na = input_dict[c_stratum][json_elem_dict["prem_nam"]]
    plt_ax = strata_plot(plt_ax, c_st, c_rgb, c_rgb, c_na, 0.5, c_linewidth)
plt, plt_ax = finalize_plot(plt, plt_ax, ax_linewidth, ax_fontsize)
plt_ax.legend(fontsize=leg_fontsize, loc=4, ncol=legend_ncol)
plt_ax.tick_params(labelsize=ax_fontsize*0.85)
plt_ax.annotate('', xy=(0.02, 0.55), xycoords='figure fraction', xytext=(0.02, 0.45), arrowprops=dict(arrowstyle="-", color='w'))
plt_ax.annotate('', xy=(0.98, 0.55), xycoords='figure fraction', xytext=(0.98, 0.45), arrowprops=dict(arrowstyle="-", color='w'))

plt.title('Initial-grid', fontsize=ax_fontsize)
for c_y in y_lines:
    plt.plot((x_lines[0], x_lines[-1]), (c_y, c_y), color='k', linewidth=ax_markersize*0.2, \
    dash_capstyle='round', dash_joinstyle='round', solid_capstyle='round', solid_joinstyle='round')
for c_x in x_lines:
    plt.plot((c_x, c_x), (y_lines[0], y_lines[-1]), color='k', linewidth=ax_markersize*0.2, \
    dash_capstyle='round', dash_joinstyle='round', solid_capstyle='round', solid_joinstyle='round')

figure = plt.gcf()
figure.savefig(st_me_nam, format='svg', bbox_inches='tight')
if save_png: figure.savefig((st_me_nam[:-3] + 'png').replace(outfold, png_outfold), format='png', dpi=600, bbox_inches='tight')


# #############
# GRID DEFORMATION
# #############

diff_y_lines = np.diff(y_lines)
diff_x_lines = np.diff(x_lines)

x_arr_lines = list(map(list, np.tile(x_lines, [len(y_lines), 1])))
y_arr_lines = list(map(list, np.tile(y_lines, [len(x_lines), 1])))

if rearr_nod:
    xy_max_tries = int(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_ran"]]\
    [json_elem_dict["mesh_ran_at"]])
else: xy_max_tries = 0

if rearr_nod:

    lprint("vertical deformation coefficient: %5.3f" % y_l_def_coe)
    lprint("horizontal deformation coefficient: %5.3f" % x_l_def_coe)
    lprint("distance threshold [m]: %4.1f" % dist_threshold)
    lprint("scaling factor: %4.2f" % elem_sf)

    exp_print = True
    for c_try in range(xy_max_tries):

        y_arr_lines, exp_print = \
        adjust_all_intervals(y_arr_lines, 'Y', x_arr_lines, \
        input_dict, strata, json_elem_dict, dist_threshold, exp_print, y_l_def_coe)

        x_arr_lines, exp_print = \
        adjust_all_intervals(x_arr_lines, 'X', y_arr_lines, \
        input_dict, strata, json_elem_dict, dist_threshold, exp_print, x_l_def_coe)

        c_str_rn = "========= rearrange nodes - iteration %3d =========" % (c_try+1)
        sys_stdout.write("\r{0}".format("%s" % (c_str_rn)))
        sys_stdout.flush()

    c_str_rn = "====== rearrange nodes - %3d iterations done ======" % (c_try+1)
    sys_stdout.write("\r{0}\n".format("%s" % (c_str_rn)))
    sys_stdout.flush()

y_poi_def = retrieve_pois_s('Y', y_arr_lines, x_arr_lines, input_dict, strata, json_elem_dict)
x_poi_def = retrieve_pois_s('X', x_arr_lines, y_arr_lines, input_dict, strata, json_elem_dict)

y_poi_ndxs = retrieve_pois_ndx(y_arr_lines, y_poi_def)
x_poi_ndxs = retrieve_pois_ndx(x_arr_lines, x_poi_def)


# #############
# INITIAL MESH
# #############

c_pos = 0
nodal_points = copy.deepcopy(y_arr_lines)
nodal_points_array = []
for a_ndx in range(len(y_arr_lines)):
    for b_ndx in range(len(x_arr_lines)):
        c_pos += 1
        c_nodal_point = [c_pos, x_arr_lines[b_ndx][a_ndx], y_arr_lines[a_ndx][b_ndx]]
        nodal_points[a_ndx][b_ndx] = c_nodal_point
        nodal_points_array.append(c_nodal_point)

element_lines = []
element_lines_H = []
element_lines_array = []
element_lines_plot = []
element_lines_write = []
xy_triang = []
xy_triang_ndxs = []
z_triang = []
ccw_continue = True
for a_ndx in range(len(y_arr_lines)-1):
    
    element_lines.append([])
    element_lines_H.append([])
    
    for b_ndx in range(len(x_arr_lines)-1):
        
        c_node_a = nodal_points[a_ndx][b_ndx]
        c_node_b = nodal_points[a_ndx+1][b_ndx]
        c_node_c = nodal_points[a_ndx+1][b_ndx+1]
        c_node_d = nodal_points[a_ndx][b_ndx+1]

        if not c_mesh_tre or a_ndx == 0 or a_ndx == len(y_arr_lines)-2:
            # quadrilateral elements only
            c_element_lines, c_element_lines_H, element_lines_array, \
            element_lines_plot, element_lines_write, ccw_continue = \
            add_element(c_node_a, c_node_b, c_node_c, c_node_d, strata, \
            json_elem_dict, input_dict, element_lines_array, element_lines_plot, \
            element_lines_write, 'quadrilateral', ccw_continue, None)

            element_lines[a_ndx].append(['q', c_element_lines])
            element_lines_H[a_ndx].append(['q', c_element_lines_H])

            xy_triang, xy_triang_ndxs, z_triang = \
            add_element_triang(xy_triang, xy_triang_ndxs, z_triang, \
            c_element_lines, element_lines_write[-1])

        else:
            # both quadrilateral and triangle elements
            if b_ndx in y_poi_ndxs[a_ndx] and b_ndx+1 in y_poi_ndxs[a_ndx+1] \
            or a_ndx in x_poi_ndxs[b_ndx] and a_ndx+1 in x_poi_ndxs[b_ndx+1]:
                
                c_node_a_t1 = c_node_a
                c_node_b_t1 = c_node_b
                c_node_c_t1 = c_node_c
                c_node_d_t1 = c_node_c

                c_element_lines_t1, c_element_lines_t1_H, element_lines_array, \
                element_lines_plot, element_lines_write, ccw_continue = \
                add_element(c_node_a_t1, c_node_b_t1, c_node_c_t1, c_node_d_t1, strata, \
                json_elem_dict, input_dict, element_lines_array, element_lines_plot, \
                element_lines_write, 'triangle', ccw_continue, None)

                c_node_a_t2 = c_node_a
                c_node_b_t2 = c_node_c
                c_node_c_t2 = c_node_d
                c_node_d_t2 = c_node_d

                c_element_lines_t2, c_element_lines_t2_H, element_lines_array, \
                element_lines_plot, element_lines_write, ccw_continue = \
                add_element(c_node_a_t2, c_node_b_t2, c_node_c_t2, c_node_d_t2, strata, \
                json_elem_dict, input_dict, element_lines_array, element_lines_plot, \
                element_lines_write, 'triangle', ccw_continue, None)

                element_lines[a_ndx].append(['tdx', c_element_lines_t1, c_element_lines_t2])
                element_lines_H[a_ndx].append(['tdx', c_element_lines_t1_H, c_element_lines_t2_H])

                xy_triang, xy_triang_ndxs, z_triang = \
                add_element_triang(xy_triang, xy_triang_ndxs, z_triang, \
                c_element_lines_t1, element_lines_write[-2])
                xy_triang, xy_triang_ndxs, z_triang = \
                add_element_triang(xy_triang, xy_triang_ndxs, z_triang, \
                c_element_lines_t2, element_lines_write[-1])

            elif b_ndx in y_poi_ndxs[a_ndx+1] and b_ndx+1 in y_poi_ndxs[a_ndx] \
            or a_ndx in x_poi_ndxs[b_ndx+1] and a_ndx+1 in x_poi_ndxs[b_ndx]:

                c_node_a_t1 = c_node_d
                c_node_b_t1 = c_node_a
                c_node_c_t1 = c_node_b
                c_node_d_t1 = c_node_b
                
                c_element_lines_t1, c_element_lines_t1_H, element_lines_array, \
                element_lines_plot, element_lines_write, ccw_continue = \
                add_element(c_node_a_t1, c_node_b_t1, c_node_c_t1, c_node_d_t1, strata, \
                json_elem_dict, input_dict, element_lines_array, element_lines_plot, \
                element_lines_write, 'triangle', ccw_continue, None)

                c_node_a_t2 = c_node_b
                c_node_b_t2 = c_node_c
                c_node_c_t2 = c_node_d
                c_node_d_t2 = c_node_d

                c_element_lines_t2, c_element_lines_t2_H, element_lines_array, \
                element_lines_plot, element_lines_write, ccw_continue = \
                add_element(c_node_a_t2, c_node_b_t2, c_node_c_t2, c_node_d_t2, strata, \
                json_elem_dict, input_dict, element_lines_array, element_lines_plot, \
                element_lines_write, 'triangle', ccw_continue, None)

                element_lines[a_ndx].append(['tsx', c_element_lines_t1, c_element_lines_t2])
                element_lines_H[a_ndx].append(['tsx', c_element_lines_t1_H, c_element_lines_t2_H])

                xy_triang, xy_triang_ndxs, z_triang = \
                add_element_triang(xy_triang, xy_triang_ndxs, z_triang, \
                c_element_lines_t1, element_lines_write[-2])
                xy_triang, xy_triang_ndxs, z_triang = \
                add_element_triang(xy_triang, xy_triang_ndxs, z_triang, \
                c_element_lines_t2, element_lines_write[-1])

            else:

                c_element_lines, c_element_lines_H, element_lines_array, \
                element_lines_plot, element_lines_write, ccw_continue = \
                add_element(c_node_a, c_node_b, c_node_c, c_node_d, strata, \
                json_elem_dict, input_dict, element_lines_array, element_lines_plot, \
                element_lines_write, 'quadrilateral', ccw_continue, None)

                element_lines[a_ndx].append(['q', c_element_lines])
                element_lines_H[a_ndx].append(['q', c_element_lines_H])

                xy_triang, xy_triang_ndxs, z_triang = \
                add_element_triang(xy_triang, xy_triang_ndxs, z_triang, \
                c_element_lines, element_lines_write[-1])

if not ccw_continue:
    print 'ERROR: some elements not in counter-clockwise order, aborting script. Please chose different values for "' + json_elem_dict["mesh"] + '" JSON-input block. Consider, in particular, to change "' + json_elem_dict["mesh_ran_at"] + '" and "' + json_elem_dict["mesh_ran_dc"] + '" input values.'
    sys_exit(1)


# #############
# model_G_Gmax_XL
# #############

# ~ depth_of_water_table_m = float(input_dict[json_elem_dict["modp"]][json_elem_dict["modp_dwt"]])
print_wt = True
len_H = len(element_lines)
len_H0 = len(element_lines[0])
no_soi = [0.0001, 0.0002, 0.0004, 0.0007, 0.0010, 0.0020, 0.0040, 0.0070, \
0.0100, 0.0200, 0.0400, 0.0700, 0.1000, 0.2000, 0.4000, 0.7000, 1.0000, 2.0000]
curvature_coefficient_a = 0.9190
excitation_frequency_Hz = 1.0 # float [0.01, 20]
number_of_cycles_N = 10 # int [1, 100]
# Ten cycles of loading at 1 Hz is chosen so that the loading conditions represent the characteristics of an earthquake.
triang_elms_wei = True # debug option: 'False' use just quadrilateral elements to calculate the mean effective stress
# Note: dimensions of elements are read in "diff_x_lines" and "diff_y_lines" variables, i.e.: on the initial-grid

for stratum_ndx in map(lambda x:x+1, range(strata)):
    c_stratum = json_elem_dict["prem"] + "%03d" % stratum_ndx
    model_G_Gmax_XL_tf = input_dict[c_stratum][json_elem_dict["prem_mGX"]][json_elem_dict["prem_mGX_tf"]]
    if model_G_Gmax_XL_tf:
        if print_wt:
            lprint("estimating: G/G0-gamma and D-gamma")
            print_wt = False
        depth_of_water_table_m = float(input_dict[c_stratum][json_elem_dict["prem_mGX"]][json_elem_dict["prem_mGX_dwt"]])
        st_K0 = float(input_dict[c_stratum][json_elem_dict["prem_mGX"]][json_elem_dict["prem_mGX_K0"]])
        st_ES = input_dict[c_stratum][json_elem_dict["prem_mGX"]][json_elem_dict["prem_mGX_ES"]]
        st_PI = float(input_dict[c_stratum][json_elem_dict["prem_mGX"]][json_elem_dict["prem_mGX_PI"]])
        st_OCR = float(input_dict[c_stratum][json_elem_dict["prem_mGX"]][json_elem_dict["prem_mGX_OR"]])
        lprint("## " + c_stratum + " ##")

        if not st_ES:

            # calculation of mean_effective_stress_atm
            lprint("depth of water table [m]: %4.1f" % depth_of_water_table_m)
            lprint("coef. of lateral earth pressure K0: %4.2f" % st_K0)
            c_are_sum = sigma_prime_vert_norm = 0.0
            if triang_elms_wei:

                for mm in range(len_H):
                    for jj in range(len_H0):
                        
                        if element_lines_H[mm][jj][0][0] == 'q': elm_coe_out = 1
                        else: elm_coe_out = 2 # 'tsx'|'tdx'
                        
                        for gg_out in range(1, elm_coe_out+1):
                            if element_lines_H[mm][jj][gg_out][1] == stratum_ndx:
                                c_lngt = diff_x_lines[mm]
                                c_hgt_sum = c_wei_sum = 0.0
                                evi_break = False
                                for tt in reversed(range(len_H0)):
                                    c_hgt = diff_y_lines[tt]
                                    
                                    if element_lines_H[mm][tt][0][0] == 'q': elm_coe_in = 1
                                    else: elm_coe_in = 2 # 'tsx'|'tdx'
                                    
                                    for gg_in in reversed(range(1, elm_coe_in+1)):
                                        c_soi = element_lines_H[mm][tt][gg_in][1]
                                        if c_soi > 0:
                                            c_wei = weight_s[c_soi] / 1000
                                            c_hgt_norm = c_hgt / float(elm_coe_in)
                                            if tt > jj or (tt == jj and gg_in > gg_out):
                                                c_hgt_sum += c_hgt_norm
                                                c_wei_sum += c_wei * c_hgt_norm
                                            elif tt == jj and gg_in == gg_out:
                                                c_hgt_sum += c_hgt_norm/2.
                                                c_are_sum += c_hgt_norm * c_lngt
                                                c_wei_sum += c_wei * c_hgt_norm/2.
                                                # if mm == 9: print '--', mm, tt, c_hgt_sum, c_wei_sum, c_are_sum, elm_coe_in
                                                evi_break = True
                                                break
                                    if evi_break: break
                                c_hgt_of_water_table_m = c_hgt_sum - depth_of_water_table_m
                                if c_hgt_of_water_table_m < 0.0: c_hgt_of_water_table_m = 0.0
                                sigma_prime_vert = c_wei_sum - 9.806 * c_hgt_of_water_table_m
                                sigma_prime_vert_norm += sigma_prime_vert * c_hgt_norm * c_lngt
                                # if mm == 9: print mm, sigma_prime_vert, c_hgt_norm, c_lngt, elm_coe_out

            else:

                for mm in range(len_H):
                    for jj in range(len_H0):
                        if element_lines_H[mm][jj][1][1] == stratum_ndx and element_lines_H[mm][jj][0][0] == 'q':
                            c_lngt = diff_x_lines[mm]
                            c_hgt_sum = c_wei_sum = 0.0
                            for tt in reversed(range(len_H0)):
                                if element_lines_H[mm][tt][0][0] == 'q':
                                    c_hgt = diff_y_lines[tt]
                                    c_soi = element_lines_H[mm][tt][1][1]
                                    if c_soi > 0:
                                        c_wei = weight_s[c_soi] / 1000
                                        if tt > jj:
                                            c_hgt_sum += c_hgt
                                            c_wei_sum += c_wei * c_hgt
                                        elif tt == jj:
                                            c_hgt_sum += c_hgt/2.
                                            c_are_sum += c_hgt * c_lngt
                                            c_wei_sum += c_wei * c_hgt/2.
                                            # if mm == 9: print '--', mm, tt, c_hgt_sum, c_wei_sum, c_are_sum
                                            break
                            c_hgt_of_water_table_m = c_hgt_sum - depth_of_water_table_m
                            if c_hgt_of_water_table_m < 0.0: c_hgt_of_water_table_m = 0.0
                            sigma_prime_vert = c_wei_sum - 9.806 * c_hgt_of_water_table_m
                            sigma_prime_vert_norm += sigma_prime_vert * c_hgt * c_lngt
                            # if mm == 9: print mm, sigma_prime_vert

            if c_are_sum:
                mean_sigma_prime_vert_kPa = sigma_prime_vert_norm/c_are_sum
                # lprint("mean sigma prime vert. [kPa]: %7.1f" % mean_sigma_prime_vert_kPa)
                mean_effective_stress_atm = (1. + 2.*st_K0) / 3. * mean_sigma_prime_vert_kPa * 0.00987
            else:
                mean_effective_stress_atm = 1.0
                lprint("WARNING: mean effective stress estimation aborted (fixed to 1.0 atm)")
        
        else: mean_effective_stress_atm = float(st_ES)
        
        lprint("mean effective stress [atm]: %6.2f" % mean_effective_stress_atm)
        lprint("PI/OCR: %.1f/%.1f" % (st_PI, st_OCR))
        # lprint("PI/OCR/K0: %.1f/%.1f/%.2f" % (st_PI, st_OCR, st_K0))
        # lprint("plasticity index PI: %4.1f" % st_PI)
        # lprint("over consolidation ratio OCR: %4.1f" % st_OCR)
        
        # Darendeli (2001)
        gamma_r = (mean_effective_stress_atm/1.)**0.3483 * (0.0352 + 0.0010 * st_PI * (st_OCR ** 0.3246))
        D0 = (mean_effective_stress_atm**-0.2889) * (0.8005 + 0.0129 * st_PI * (st_OCR ** -0.1069)) \
        * (1. + 0.2919 * np.log(excitation_frequency_Hz))
        c_1 = -1.1143 * (curvature_coefficient_a ** 2) + 1.8618 * curvature_coefficient_a + 0.2533
        c_2 =  0.0805 * (curvature_coefficient_a ** 2) - 0.0710 * curvature_coefficient_a - 0.0095
        c_3 = -0.0005 * (curvature_coefficient_a ** 2) + 0.0002 * curvature_coefficient_a + 0.0003
        b_D = 0.6329 - 0.0057 * np.log(float(number_of_cycles_N))
        input_dict_G_G0_g = []; input_dict_D_g = []; first_no_soi = True
        for gamma in no_soi:
            G_G0_v = 1. / (1. + (gamma/gamma_r)**curvature_coefficient_a)
            D_masing_a_1 = (100./np.pi) * (4. * ((gamma - gamma_r * np.log((gamma + gamma_r)/gamma_r)) \
            / ((gamma ** 2.) / (gamma + gamma_r))) - 2.)
            D_masing = c_1 * D_masing_a_1 + c_2 * (D_masing_a_1 ** 2.) + c_3 * (D_masing_a_1 ** 3.)
            D_v = b_D * (G_G0_v ** 0.1) * D_masing + D0
            if first_no_soi:
                G_G0_v = 1.0
                D0_sigma = np.exp(-5.) + np.exp(-0.25) * (D_v**0.5)
                first_no_soi = False
            input_dict_G_G0_g.append([gamma, G_G0_v])
            input_dict_D_g.append([gamma, D_v])
        
        D0_min = D0 / 5.; # fixing lower bound of distribution to one-fifth of D0
        if not input_dict[c_stratum][json_elem_dict["prem_mGX"]][json_elem_dict["prem_mGX_DU"]]: D0_sigma = 0.0
        input_dict[c_stratum][json_elem_dict["prem_xld"]][json_elem_dict["etc_va"]] = D0 / 100.
        input_dict[c_stratum][json_elem_dict["prem_xld"]][json_elem_dict["etc_uc"]] = D0_sigma / 100.
        input_dict[c_stratum][json_elem_dict["prem_xld"]][json_elem_dict["etc_mv"]] = D0_min / 100.
        lprint("D0/D0_sigma/D0_min [percent]: %.2f/%.2f/%.2f" % (D0, D0_sigma, D0_min))
        
        input_dict[c_stratum][json_elem_dict["prem_soi"]][json_elem_dict["prem_soi_gg"]] = input_dict_G_G0_g
        input_dict[c_stratum][json_elem_dict["prem_soi"]][json_elem_dict["prem_soi_xl"]] = input_dict_D_g
        
        element_lines_H_mask = np.full((len(element_lines_H), len(element_lines_H[0])), True, dtype=bool)
        element_lines_array, element_lines_plot, element_lines_write, xy_triang, xy_triang_ndxs, z_triang = \
        redefine_elements(element_lines_H, element_lines_H_mask, strata, json_elem_dict, input_dict)


# #############
# MESH DOWNSAMPLING
# #############

mesh_downsampling = input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_dow"]][json_elem_dict["mesh_dow_tf"]]
min_F_nodes = int(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_dow"]][json_elem_dict["mesh_dow_ml"]])
w_multiplier = float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_dow"]][json_elem_dict["mesh_dow_fm"]][json_elem_dict["etc_hv"]])
h_multiplier = float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_dow"]][json_elem_dict["mesh_dow_fm"]][json_elem_dict["etc_vh"]])
tri_extr = input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_dow"]][json_elem_dict["mesh_dow_te"]]

dict_width_s = {}
for w_s in width_s: dict_width_s[w_s[-1]] = w_s[0]*w_multiplier
dict_height_s = {}
for w_s in height_s: dict_height_s[w_s[-1]] = w_s[0]

lprint("mesh downsampling: %s" % mesh_downsampling)

if mesh_downsampling:
    
    lprint("min length of module to be reshaped: %3.0f" % min_F_nodes)
    lprint("H/V shape factor multiplier: %4.1f" % w_multiplier)
    lprint("V/H shape factor multiplier: %4.1f" % h_multiplier)
    
    element_lines_H, diff_x_lines, diff_y_lines = \
    add_fake_elements(element_lines_H, diff_x_lines, diff_y_lines)
    
    if isinstance(diff_x_lines, type(None)):
        f_elm_H = element_lines_H
        print 'ERROR: element belonging to last vertical polyline and described by nodes ' + str(f_elm_H[1][0][0][0]) + ', ' + str(f_elm_H[1][0][1][0]) + ', ' + str(f_elm_H[1][0][2][0]) + ' is triangular. This should never happen, since the condition "... or a_ndx == 0 or a_ndx == len(y_arr_lines)-2" has been added in the code. Please use "ipython" to debug this!'
        sys_exit(1)
    if isinstance(diff_y_lines, type(None)):
        f_elm_H = element_lines_H
        print 'ERROR: element belonging to last horizontal polyline and described by nodes ' + str(f_elm_H[1][0][0][0]) + ', ' + str(f_elm_H[1][0][1][0]) + ', ' + str(f_elm_H[1][0][2][0]) + ' is triangular. Try to increment "distance_threshold_m" within inner-block "rearrange_nodes".'
        sys_exit(1)
    
    element_lines_H = element_lines_numpy(element_lines_H)
    element_lines_H_mask = np.full(np.shape(element_lines_H), True, dtype=bool)
    
    ini_len = len(element_lines_array)
    w_fac_mul = w_factor * w_multiplier
    h_fac_mul = h_factor * h_multiplier
    for ref_soil in map(lambda x:x+1, range(strata)):
        for c_round in range(1,8):

            pre_len = len(element_lines_array)

            element_lines_H = element_lines_H.T
            element_lines_H_mask = element_lines_H_mask.T

            element_lines_H, element_lines_H_mask, element_lines_array, \
            element_lines_plot, element_lines_write, xy_triang, xy_triang_ndxs, z_triang = \
            mesh_ds(element_lines_H, element_lines_H_mask, ref_soil, min_F_nodes, json_elem_dict, \
            input_dict, w_fac_mul, h_fac_mul, dict_width_s, dict_height_s, diff_x_lines, diff_y_lines, tri_extr)
        
            element_lines_H = element_lines_H.T
            element_lines_H_mask = element_lines_H_mask.T
        
            element_lines_H, element_lines_H_mask, element_lines_array, \
            element_lines_plot, element_lines_write, xy_triang, xy_triang_ndxs, z_triang = \
            mesh_ds(element_lines_H, element_lines_H_mask, ref_soil, min_F_nodes, json_elem_dict, \
            input_dict, w_fac_mul, h_fac_mul, dict_width_s, dict_height_s, diff_x_lines, diff_y_lines, tri_extr)

            pos_len = len(element_lines_array)

            c_stratum = json_elem_dict["prem"] + "%03d" % ref_soil
            c_string = c_stratum + ": iteration " + str(c_round)
            sys_stdout.write("\r{0}".format(c_string + " - elements: %5d --> %5d  " % (pre_len, pos_len)))
            sys_stdout.flush()
            if pre_len == pos_len: break

    sys_stdout.write("\r{0}".format("%s" % ""))
    sys_stdout.flush()
    lprint("# of elements: %5d (before) --> %5d (after)" % (ini_len, pos_len))            


if len(element_lines_write) > 99999:
    print 'ERROR: due to limit of five numeric digits to be read by QUAD4M for element-IDs (it is "%5i"), maximum allowed number of elements is fixed to 99999 (' + str(len(element_lines_write)) + ' have been generated instead), aborting script.'
    sys_exit(1)


# #############
# MESH NODES
# #############

element_lines_t = map(list, zip(*element_lines))

element_lines_9 = copy.deepcopy(element_lines)
for cel in element_lines_9:
    for celi in cel:
        if celi[1][1]: celi[1][1] = 9
        if not celi[0] == 'q':
            if celi[2][1]: celi[2][1] = 9
element_lines_t_9 = copy.deepcopy(element_lines_t)
for cel in element_lines_t_9:
    for celi in cel:
        if celi[1][1]: celi[1][1] = 9
        if not celi[0] == 'q':
            if celi[2][1]: celi[2][1] = 9
c_s_9 = []
plt_ax, c_s_9 = plot_all_borders(plt_ax, element_lines_9, element_lines_t_9, c_s_9, l_in=c_linewidth*0.3)

e_l_9 = []
for bi_arr in c_s_9:
    for mo_arr in reversed(bi_arr):
        e_l_9.append(search_id(element_lines_H, mo_arr))
e_l_9_w = map(list, zip(*e_l_9))

# --------

element_lines_w = map(list, zip(*element_lines_write))
nodal_points_pre = []
for x_ndx in range(len(nodal_points_array)):
    if nodal_points_array[x_ndx][0] in element_lines_w[0] \
    or nodal_points_array[x_ndx][0] in element_lines_w[1] \
    or nodal_points_array[x_ndx][0] in element_lines_w[2] \
    or nodal_points_array[x_ndx][0] in element_lines_w[3]:
        nodal_points_pre.append([nodal_points_array[x_ndx][1], nodal_points_array[x_ndx][2]])

c_MP = MultiPoint(nodal_points_pre)

modp_c_X = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_acc"]][json_elem_dict["modp_acc__X"]][json_elem_dict["etc_i"]]
modp_c_Y = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_acc"]][json_elem_dict["modp_acc__Y"]][json_elem_dict["etc_i"]]
modp_cXY = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_acc"]][json_elem_dict["modp_acc_XY"]][json_elem_dict["etc_i"]]

c__X_d = find_np(modp_c_X, c_MP)
c__Y_d = find_np(modp_c_Y, c_MP)
c_XY_d = find_np(modp_cXY, c_MP)

modp_d_X = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_acc"]][json_elem_dict["modp_acc__X"]][json_elem_dict["etc_g"]]
modp_d_Y = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_acc"]][json_elem_dict["modp_acc__Y"]][json_elem_dict["etc_g"]]
modp_dXY = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_acc"]][json_elem_dict["modp_acc_XY"]][json_elem_dict["etc_g"]]
if modp_d_X:
    for modp_d in modp_d_X:
        idx_d = np.abs(np.array(e_l_9_w[1]) - modp_d).argmin()
        c__X_d.append([e_l_9_w[1][idx_d], e_l_9_w[2][idx_d]])
if modp_d_Y:
    for modp_d in modp_d_Y:
        idx_d = np.abs(np.array(e_l_9_w[1]) - modp_d).argmin()
        c__Y_d.append([e_l_9_w[1][idx_d], e_l_9_w[2][idx_d]])
if modp_dXY:
    for modp_d in modp_dXY:
        idx_d = np.abs(np.array(e_l_9_w[1]) - modp_d).argmin()
        c_XY_d.append([e_l_9_w[1][idx_d], e_l_9_w[2][idx_d]])

c__X_MP = MultiPoint(c__X_d)
c__Y_MP = MultiPoint(c__Y_d)
c_XY_MP = MultiPoint(c_XY_d)

modp_trb = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_trb"]]
modp_trb_vr_po = float(modp_trb[json_elem_dict["modp_trb_vr"]][json_elem_dict["etc_po"]])
modp_trb_vl_po = float(modp_trb[json_elem_dict["modp_trb_vl"]][json_elem_dict["etc_po"]])
modp_trb_hz_po = float(modp_trb[json_elem_dict["modp_trb_hz"]][json_elem_dict["etc_po"]])
modp_trb_vr_va = int(modp_trb[json_elem_dict["modp_trb_vr"]][json_elem_dict["etc_va"]])
modp_trb_vl_va = int(modp_trb[json_elem_dict["modp_trb_vl"]][json_elem_dict["etc_va"]])
modp_trb_hz_va = int(modp_trb[json_elem_dict["modp_trb_hz"]][json_elem_dict["etc_va"]])
no_bc_array_1 = []; no_bc_array_2 = []; no_bc_array_3 = []; no_bc_array_4 = []
no_out_array_1 = []; no_out_array_2 = []; no_out_array_3 = []
nodal_points_write = []
for x_ndx in range(len(nodal_points_array)):
    if nodal_points_array[x_ndx][0] in element_lines_w[0] \
    or nodal_points_array[x_ndx][0] in element_lines_w[1] \
    or nodal_points_array[x_ndx][0] in element_lines_w[2] \
    or nodal_points_array[x_ndx][0] in element_lines_w[3]:

        c_po = Point(nodal_points_array[x_ndx][1], nodal_points_array[x_ndx][2])

        no_bc = int(0)
        if nodal_points_array[x_ndx][1] == modp_trb_vr_po: no_bc = modp_trb_vr_va
        if nodal_points_array[x_ndx][1] == modp_trb_vl_po: no_bc = modp_trb_vl_va
        if nodal_points_array[x_ndx][2] == modp_trb_hz_po: no_bc = modp_trb_hz_va

        no_out = int(0)
        if c_po.intersects(c__X_MP): no_out = int(1) # if not c__X_MP.is_empty:
        if c_po.intersects(c__Y_MP): no_out = int(2) # if not c__Y_MP.is_empty:
        if c_po.intersects(c_XY_MP): no_out = int(3) # if not c_XY_MP.is_empty:
        
        if no_bc:
            exec("no_bc_array_" + str(no_bc) + ".append([nodal_points_array[x_ndx][0], " + \
            "nodal_points_array[x_ndx][1], nodal_points_array[x_ndx][2], no_bc])")
        if no_out:
            exec("no_out_array_" + str(no_out) + ".append([nodal_points_array[x_ndx][0], " + \
            "nodal_points_array[x_ndx][1], nodal_points_array[x_ndx][2], no_out])")

        nodal_points_write.append([nodal_points_array[x_ndx][0], \
        nodal_points_array[x_ndx][1], nodal_points_array[x_ndx][2], no_bc, no_out])

if len(nodal_points_write) > 99999:
    print 'ERROR: due to limit of five numeric digits to be read by QUAD4M for node-IDs (it is "%5i"), maximum allowed number of nodes is fixed to 99999 (' + str(len(element_lines_write)) + ' have been generated instead), aborting script.'
    sys_exit(1)


# #############
# MESH PLOT
# #############

# #############################
lprint("mesh plot")
plt, plt_ax, add_legend_dict = pre_plot(plt, input_dict, json_elem_dict, ax_fontsize, c_linewidth)
# #############################
mesh_cont = 0
for c_elem in element_lines_plot:
    c_stratum = json_elem_dict["prem"] + "%03d" % c_elem[1]
    c_rgba = cmap(norm(float(c_elem[1])))
    c_rgb = list(c_rgba)[:-1]
    if eval("add_legend_dict['add_legend_st" + "%03d" % c_elem[1] + "'] == True"):
        c_na = input_dict[c_stratum][json_elem_dict["prem_nam"]]
        exec("add_legend_dict['add_legend_st" + "%03d" % c_elem[1] + "'] = False")
    else: c_na = None
    if c_elem[2] == 't': c_elems = c_elem[0][:-1]
    elif c_elem[2] == 'q': c_elems = c_elem[0]
    plt_ax = strata_plot(plt_ax, c_elems, c_rgb, c_rgb, c_na, 0.50, c_linewidth*0.10)
    mesh_cont += 1
    if mesh_cont%100 == 0:
        sys_stdout.write("\r{0}".format("%6d/%6d" % (mesh_cont, len(element_lines_plot))))
        sys_stdout.flush()
sys_stdout.write("\r{0}".format("%s" % ""))
sys_stdout.flush()
lprint("%6d elements" % (mesh_cont))
plt_ax.plot(0, 0, color=[0.0, 0.0, 0.0], label='input', linewidth=c_linewidth*1.0)
plt_ax.plot(0, 0, color=[1.0, 1.0, 1.0], label='model', linewidth=c_linewidth*0.3)
l1 = plt_ax.legend(fontsize=leg_fontsize, loc=7, ncol=legend_ncol)
# #############################
plt.title('Mesh', fontsize=ax_fontsize)
# #############################
c_s = []
plt_ax, c_s = plot_all_borders(plt_ax, element_lines, element_lines_t, c_s, l_in=c_linewidth*0.3)
# #############################
plt, plt_ax = post_plot(plt, plt_ax, no_bc_array_1, no_bc_array_2, no_bc_array_3, \
no_bc_array_4, no_out_array_1, no_out_array_2, no_out_array_3, leg_fontsize, ax_markersize*2.0)
# #############################
plt, plt_ax = finalize_plot(plt, plt_ax, ax_linewidth, ax_fontsize)
plt_ax.tick_params(labelsize=ax_fontsize*0.85)
plt_ax.annotate('', xy=(0.02, 0.55), xycoords='figure fraction', xytext=(0.02, 0.45), arrowprops=dict(arrowstyle="-", color='w'))
plt_ax.annotate('', xy=(0.98, 0.55), xycoords='figure fraction', xytext=(0.98, 0.45), arrowprops=dict(arrowstyle="-", color='w'))
# #############################
plt_ax.add_artist(l1)
# #############################
figure = plt.gcf()
figure.savefig(mesh_nam, format='svg', bbox_inches='tight')
if save_png: figure.savefig((mesh_nam[:-3] + 'png').replace(outfold, png_outfold), format='png', dpi=600, bbox_inches='tight')
# #############################


# ##############
# PARAMETR PLOTS
# ##############

xy_triang = np.array(xy_triang)
xy_mtri = mtri.Triangulation(xy_triang[:,0], xy_triang[:,1], triangles=xy_triang_ndxs)
xy_triang_DENS = map(list, zip(*z_triang))[0]
xy_triang_DENS = map(lambda x: x/1000.0, xy_triang_DENS)
xy_triang_PO = map(list, zip(*z_triang))[1]
xy_triang_GMX = map(list, zip(*z_triang))[2]
xy_triang_GMX = map(lambda x: x/1000.0, xy_triang_GMX)
xy_triang_G = map(list, zip(*z_triang))[3]
xy_triang_G = map(lambda x: x/1000.0, xy_triang_G)
xy_triang_XL = map(list, zip(*z_triang))[4]

lprint_str = "DENS plot"
xy_triang_PAR = xy_triang_DENS
plt_title = 'Unit weight [kN/m^3]'
par_name = par_DENS
c_bar_ticks = [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
c_fmt_punct = '{:.0f}'
c_minVal, c_maxVal, logn = 18.0, 23.0, False
#~ c_minVal, c_maxVal = min(xy_triang_DENS), max(xy_triang_DENS)
par_plot(plt, plt_ax, input_dict, json_elem_dict, lprint_str, xy_mtri, xy_triang_PAR, \
plt_title, par_name, c_minVal, c_maxVal, c_bar_ticks, c_fmt_punct, cmap, logn, \
c_linewidth, leg_fontsize, ax_linewidth, ax_fontsize, c_s, save_png, outfold, png_outfold)

lprint_str = "PO plot"
xy_triang_PAR = xy_triang_PO
plt_title = "Poisson's ratio"
par_name = par_PO
c_bar_ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
c_fmt_punct = '{:.1f}'
c_minVal, c_maxVal, logn = 0.1, 0.5, False
#~ c_minVal, c_maxVal = min(xy_triang_PO), max(xy_triang_PO)
par_plot(plt, plt_ax, input_dict, json_elem_dict, lprint_str, xy_mtri, xy_triang_PAR, \
plt_title, par_name, c_minVal, c_maxVal, c_bar_ticks, c_fmt_punct, cmap, logn, \
c_linewidth, leg_fontsize, ax_linewidth, ax_fontsize, c_s, save_png, outfold, png_outfold)

lprint_str = "GMX plot"
xy_triang_PAR = xy_triang_GMX
plt_title = 'Shear modulus at small strains [kN/m^2]'
par_name = par_GMX
c_bar_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000]
c_fmt_punct = '{:.0g}'
c_minVal, c_maxVal, logn = 20, 7000, True
#~ c_minVal, c_maxVal = min(xy_triang_GMX), max(xy_triang_GMX)
par_plot(plt, plt_ax, input_dict, json_elem_dict, lprint_str, xy_mtri, xy_triang_PAR, \
plt_title, par_name, c_minVal, c_maxVal, c_bar_ticks, c_fmt_punct, cmap, logn, \
c_linewidth, leg_fontsize, ax_linewidth, ax_fontsize, c_s, save_png, outfold, png_outfold)

lprint_str = "G plot"
xy_triang_PAR = xy_triang_G
plt_title = 'Shear modulus for 1st iteration [kN/m^2]'
par_name = par_G
#~ c_minVal, c_maxVal = min(xy_triang_G), max(xy_triang_G)
par_plot(plt, plt_ax, input_dict, json_elem_dict, lprint_str, xy_mtri, xy_triang_PAR, \
plt_title, par_name, c_minVal, c_maxVal, c_bar_ticks, c_fmt_punct, cmap, logn, \
c_linewidth, leg_fontsize, ax_linewidth, ax_fontsize, c_s, save_png, outfold, png_outfold)

lprint_str = "XL plot"
xy_triang_PAR = xy_triang_XL
plt_title = 'Initial fraction of critical damping [decimal]'
par_name = par_XL
c_bar_ticks = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
c_fmt_punct = '{:.0g}'
c_minVal, c_maxVal, logn = 0.002, 0.1, True
#~ c_minVal, c_maxVal = min(xy_triang_XL), max(xy_triang_XL)
par_plot(plt, plt_ax, input_dict, json_elem_dict, lprint_str, xy_mtri, xy_triang_PAR, \
plt_title, par_name, c_minVal, c_maxVal, c_bar_ticks, c_fmt_punct, cmap, logn, \
c_linewidth, leg_fontsize, ax_linewidth, ax_fontsize, c_s, save_png, outfold, png_outfold)


# #############################
# SAVING QUAD4M INPUT FILES
# #############################

lprint("saving QUAD4M input files")

wave_tss = float(input_dict[json_elem_dict["wave"]][json_elem_dict["wave_tss"]])
wave_mpf = float(input_dict[json_elem_dict["wave"]][json_elem_dict["wave_mpf"]])
waveXg = waveX * wave_mpf
waveYg = waveY * wave_mpf
WF_format = '%8.5f'
#~ WFX
WFX_fname = os_path.join(*[outfold, 'WFX.ACC'])
WFX_fid = open(WFX_fname,'w')
for ii in range(wave_hrx): WFX_fid.write(hdrx_lines[ii])
WFX_fid.write('# -------- CONVERTED TO g (MULTIPLIER FACTOR ' + str(wave_mpf) + ') ----------\n')
np.savetxt(WFX_fid, waveXg, fmt=WF_format)
WFX_fid.write('\n')
WFX_fid.close()
#~ WFY
WFY_fname = os_path.join(*[outfold, 'WFY.ACC'])
WFY_fid = open(WFY_fname,'w')
for ii in range(wave_hry): WFY_fid.write(hdry_lines[ii])
WFY_fid.write('# -------- CONVERTED TO g (MULTIPLIER FACTOR ' + str(wave_mpf) + ') ----------\n')
np.savetxt(WFY_fid, waveYg, fmt=WF_format)
WFY_fid.write('\n')
WFY_fid.close()

modp_rvp = float(input_dict[json_elem_dict["modp"]][json_elem_dict["modp_rvp"]])
modp_rvs = float(input_dict[json_elem_dict["modp"]][json_elem_dict["modp_rvs"]])
modp_rho = float(input_dict[json_elem_dict["modp"]][json_elem_dict["modp_rho"]])
modp_nmb = int(input_dict[json_elem_dict["modp"]][json_elem_dict["modp_nmb"]])
wave_prp_pre = input_dict[json_elem_dict["wave"]][json_elem_dict["wave_prp"]]
if not wave_prp_pre:
    uti_g = 980.6 #[cm/s^2]
    SA_UE, SD_UE, Tp = resp_spectra_acc(wave_tss,waveXg,zz=0.05,st=0.04,ed=4.00,nu=80)
    SA_UE_ndx = np.argmax(SA_UE)
    wave_prp = np.around(Tp[SA_UE_ndx], decimals=3)
    lprint("'PRINPUT_...' set to " + str(wave_prp) + "s")
else: wave_prp = float(wave_prp_pre)

nodal_points_write_m = []
nodal_points_write_d = {}
for hh in range(len(nodal_points_write)):
    nodal_points_write_d[nodal_points_write[hh][0]] = [hh+1]
    nodal_points_write_m.append([hh+1] + nodal_points_write[hh][1:])

element_lines_write_m = []
for hh in range(len(element_lines_write)):
    element_lines_write_m.append([hh+1] + \
    nodal_points_write_d[element_lines_write[hh][0]] + \
    nodal_points_write_d[element_lines_write[hh][1]] + \
    nodal_points_write_d[element_lines_write[hh][2]] + \
    nodal_points_write_d[element_lines_write[hh][3]] + \
    element_lines_write[hh][4:])

row00 = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_jbt"]]
row01 = 'UNITS (E for English, S for SI):                               *** (A1)            ***'
row02 = 'S'
row03 = '       DRF       PRM    ROCKVP    ROCKVS   ROCKRHO    	   *** (5F10.0)        ***'
row04 = '{:10.4f}'.format(1) + '{:10.4f}'.format(0.65) + '{:10.1f}'.format(modp_rvp) + '{:10.1f}'.format(modp_rvs) + '{:10.1f}'.format(modp_rho)
row05 = ' NELM NDPT NSLP                                                *** (3I5)           ***'
row06 = '{:5.0f}'.format(len(element_lines_write_m)) + '{:5.0f}'.format(len(nodal_points_write_m)) + '{:5.0f}'.format(0)
row07 = 'KGMAX KGEQ N1EQ N2EQ N3EQ NUMB   KV KSAV                       *** (8I5)           ***'
row08 = '{:5.0f}'.format(lenwave) + '{:5.0f}'.format(wave_ltl) + '{:5.0f}'.format(wave_ftl) + '{:5.0f}'.format(wave_fts) + '{:5.0f}'.format(wave_lts) + '{:5.0f}'.format(modp_nmb) + '{:5.0f}'.format(2) + '{:5.0f}'.format(1)
row09 = '      DTEQ    EQMUL1    EQMUL2    UGMAX1    UGMAX2 HDRX HDRY NPLX NPLY   PRINPUT *** (5F10.0,4I5,F10.0) ***'
row10 = '{:10.5f}'.format(wave_tss) + '{:10.5f}'.format(1) + '{:10.5f}'.format(1) + '{:10s}'.format('') + '{:10s}'.format('') + '{:5.0f}'.format(wave_hrx+1) + '{:5.0f}'.format(wave_hry+1) + '{:5.0f}'.format(1) + '{:5.0f}'.format(1) + '{:10.5f}'.format(wave_prp)
row11 = 'EARTHQUAKE INPUT FILE NAME(S) & FORMAT(S) (* for FREE FORMAT)  *** (A)             ***'
row12 = 'WFX.ACC'
row13 = '*'
row14 = 'WFY.ACC'
row15 = '*'
row16 = ' SOUT AOUT KOUT                                                *** (3I5)           ***'
row17 = '    1    1    1'
row18 = 'STRESS OUTPUT FORMAT (M or C), FILE PREFIX, AND SUFFIX:        *** (A)             ***'
row19 = 'COMBINED'
row20 = 'SG'
row21 = 'Q4S'
row22 = 'ACCELERATION OUTPUT FORMAT (M or C), FILE PREFIX, AND SUFFIX:  *** (A)             ***'
row23 = 'COMBINED'
row24 = 'SG'
row25 = 'Q4A'
row26 = 'SEISMIC COEFFICIENT OUTPUT FORMAT:                             *** (A)             ***'
row27 = 'COMBINED'
row28 = 'SG'
row29 = 'Q4K'
row30 = 'SYSTEM STATE OUTPUT FILE:                                      *** (A)             ***'
row31 = 'SG.Q4R'
row32 = '    N  NP1  NP2  NP3  NP4 TYPE      DENS       PO        GMX         G        XL LSTR  *** (6I5,5F10.0,I5) ***'

Q4I_filename = os_path.join(*[outfold, 'MDL.Q4I'])
Q4I_format_ele = '%5u%5u%5u%5u%5u%5u%10.0f%10.4f%10.0f%10.0f%10.4f'
Q4I_format_poi = '%5u%10.2f%10.2f%5u%5u'
Q4I_fid = open(Q4I_filename,'w')
for ii in range(33):
    curr_line = eval("row" + "%02d" % ii)
    Q4I_fid.write(curr_line + '\n')
np.savetxt(Q4I_fid, list(element_lines_write_m), fmt=Q4I_format_ele)
rowxx = '    N      XORD      YORD   BC  OUT      X2IH      X1IH       XIH      X2IV      X1IV       XIV   *** (I5,2F10.0,2I5,6F10.0) ***'
Q4I_fid.write(rowxx + '\n')
np.savetxt(Q4I_fid, list(nodal_points_write_m), fmt=Q4I_format_poi)
Q4I_fid.write('\n')
Q4I_fid.close()


SG_filename = os_path.join(*[outfold, 'SG.DAT'])
SG_fid = open(SG_filename,'w')
SG_fid.write('{:5.0f}'.format(strata) + '\n')
for stratum_ndx in map(lambda x:x+1, range(strata)):
    c_stratum = json_elem_dict["prem"] + "%03d" % stratum_ndx
    c_na = input_dict[c_stratum][json_elem_dict["prem_nam"]]
    XL = float(input_dict[c_stratum][json_elem_dict["prem_xld"]][json_elem_dict["etc_va"]] * 100.0)

    soi_gg = input_dict[c_stratum][json_elem_dict["prem_soi"]][json_elem_dict["prem_soi_gg"]]
    if not soi_gg: soi_gg = create_soi(no_soi, 1.0)
    SG_fid.write('{:5.0f}'.format(len(soi_gg)))
    #~ SG_fid.write('{:>4s}'.format('#' + str(stratum_ndx)))
    #~ SG_fid.write('{:>20s}'.format('modulus for ' + c_na))
    SG_fid.write('\n')
    SG_fid = write_soi(soi_gg, SG_fid)

    soi_xl = input_dict[c_stratum][json_elem_dict["prem_soi"]][json_elem_dict["prem_soi_xl"]]
    if not soi_xl: soi_xl = create_soi(no_soi, XL)
    SG_fid.write('{:5.0f}'.format(len(soi_xl)))
    #~ SG_fid.write('{:>4s}'.format(''))
    #~ SG_fid.write('{:>20s}'.format('damping for ' + c_na))
    SG_fid.write('\n')
    SG_fid = write_soi(soi_xl, SG_fid)

    try:
        plt.clf()
        plt.figure(figsize=(2.5, 2))
        
        plt.ylabel('Normalized shear-modulus, $G/G_0$', fontsize=ax_fontsize)
        plt.xlabel('Shear-strain, $\gamma$ [%]', fontsize=ax_fontsize)
        plt_ax = plt.gca()
        c_plt_soi_gg = plt_ax.plot(zip(*soi_gg)[0], zip(*soi_gg)[1], color='black', marker='o', linewidth=c_linewidth, markersize=ax_markersize)
        plt.xscale('log')
        plt.title(c_na, fontsize=ax_fontsize, fontweight="bold")
        plt.grid(True, linestyle=':', dash_capstyle='round', dash_joinstyle='round', linewidth=ax_linewidth, color=[0.5, 0.5, 0.5])
        plt.grid(True, axis='x', which='minor', linestyle=':', dash_capstyle='round', dash_joinstyle='round', linewidth=ax_linewidth*0.7, color=[0.8, 0.8, 0.8])
        plt.box(True)
        plt.ylim([0,1.05])
        plt.xlim([0.0001,1])
        for axis in ['top','bottom','left','right']: plt_ax.spines[axis].set_linewidth(ax_linewidth)
        plt_ax.tick_params(width=ax_linewidth, labelsize=ax_fontsize, which='both')
        figure = plt.gcf()
        f_nam = sg_nam + 'NG_' + c_stratum + '.svg'
        figure.savefig(f_nam, format='svg', bbox_inches='tight')
        if save_png: figure.savefig((f_nam[:-3] + 'png').replace(outfold, png_outfold), format='png', dpi=600, bbox_inches='tight')

        plt.ylabel('Damping, $D$ [%]', fontsize=ax_fontsize)
        for handle in c_plt_soi_gg: handle.remove()
        plt_ax.plot(zip(*soi_xl)[0], zip(*soi_xl)[1], color='black', marker='o', linewidth=c_linewidth, markersize=ax_markersize)
        if max(zip(*soi_xl)[1]) < 30: plt.ylim([0,30])
        f_nam = sg_nam + 'D_' + c_stratum + '.svg'
        figure.savefig(f_nam, format='svg', bbox_inches='tight')
        if save_png: figure.savefig((f_nam[:-3] + 'png').replace(outfold, png_outfold), format='png', dpi=600, bbox_inches='tight')
    except:
        lprint("WARNING: degradation figures not generated or partially generated")

SG_fid.write('\n')
SG_fid.close()


# #############################
# borders.txt
# #############################

BO_filename = os_path.join(*[outfold, 'borders.txt'])
BO_fid = open(BO_filename,'w')
for cc_s in c_s:
    BO_fid.write('{:10.3f}'.format(cc_s[0][0]) + '{:10.3f}'.format(cc_s[0][1]) + '{:10.3f}'.format(cc_s[1][0]) + '{:10.3f}'.format(cc_s[1][1]) + '\n')
BO_fid.write('\n')
for cc_s in c_s_9:
    BO_fid.write('{:10.3f}'.format(cc_s[0][0]) + '{:10.3f}'.format(cc_s[0][1]) + '{:10.3f}'.format(cc_s[1][0]) + '{:10.3f}'.format(cc_s[1][1]) + '\n')
BO_fid.write('\n')
BO_fid.write('{:10.3f}'.format(-999.999) + '{:10.3f}'.format(-999.999) + '\n')
for cc_ndx in range(len(c__X_MP)):
    BO_fid.write('{:10.3f}'.format(c__X_MP[cc_ndx].x) + '{:10.3f}'.format(c__X_MP[cc_ndx].y) + '\n')
BO_fid.write('\n')
BO_fid.write('{:10.3f}'.format(-999.999) + '{:10.3f}'.format(-999.999) + '\n')
for cc_ndx in range(len(c__Y_MP)):
    BO_fid.write('{:10.3f}'.format(c__Y_MP[cc_ndx].x) + '{:10.3f}'.format(c__Y_MP[cc_ndx].y) + '\n')
BO_fid.write('\n')
BO_fid.write('{:10.3f}'.format(-999.999) + '{:10.3f}'.format(-999.999) + '\n')
for cc_ndx in range(len(c_XY_MP)):
    BO_fid.write('{:10.3f}'.format(c_XY_MP[cc_ndx].x) + '{:10.3f}'.format(c_XY_MP[cc_ndx].y) + '\n')
BO_fid.close()


# #############################
# INPUT WAVEFORMS PLOT
# #############################

timeArr = np.array(range(0,len(waveXg))) * wave_tss

plt.clf()
figure, plt_ax = plt.subplots(2, 1, sharex=True)

v001 = 1.20; v002 = 1.10; v003 = 1.05
wave_max = max(np.max(np.abs(waveYg)), np.max(np.abs(waveXg))) * v001

plt_ax[0].plot(timeArr, waveXg, linestyle='-', marker='', color='k', linewidth=c_linewidth, label='_horizontal')
plt_ax[1].plot(timeArr, waveYg, linestyle='-', marker='', color='k', linewidth=c_linewidth, label='_vertical')
therectF0 = patches.Rectangle(xy=(wave_fts*wave_tss,-wave_max*v003/v001), width=wave_lts*wave_tss - wave_fts*wave_tss, height=2*wave_max*v003/v001, fill=False, linestyle='-', edgecolor='g', alpha=0.75, label='first iterations', linewidth=c_linewidth)
therectF1 = patches.Rectangle(xy=(wave_fts*wave_tss,-wave_max*v003/v001), width=wave_lts*wave_tss - wave_fts*wave_tss, height=2*wave_max*v003/v001, fill=False, linestyle='-', edgecolor='g', alpha=0.75, label='first iterations', linewidth=c_linewidth)
therectL0 = patches.Rectangle(xy=(wave_ftl*wave_tss,-wave_max*v002/v001), width=wave_ltl*wave_tss - wave_ftl*wave_tss, height=2*wave_max*v002/v001, fill=False, linestyle='-', edgecolor='r', alpha=0.75, label='last iteration', linewidth=c_linewidth)
therectL1 = patches.Rectangle(xy=(wave_ftl*wave_tss,-wave_max*v002/v001), width=wave_ltl*wave_tss - wave_ftl*wave_tss, height=2*wave_max*v002/v001, fill=False, linestyle='-', edgecolor='r', alpha=0.75, label='last iteration', linewidth=c_linewidth)
plt_ax[0].add_patch(therectF0)
plt_ax[1].add_patch(therectF1)
plt_ax[0].add_patch(therectL0)
plt_ax[1].add_patch(therectL1)

plt_ax[0].set_ylim([-wave_max, wave_max])
plt_ax[1].set_ylim([-wave_max, wave_max])
plt_ax[1].set_xlabel('Time [s]', fontsize=ax_fontsize*1.5)
plt_ax[0].set_ylabel('horizontal [g]', fontsize=ax_fontsize*1.5)
plt_ax[1].set_ylabel('vertical [g]', fontsize=ax_fontsize*1.5)
plt_ax[0].set_title('Acceleration input time-histories', fontsize=ax_fontsize*1.5)
plt_ax[0].legend(fontsize=leg_fontsize*1.5)

figure.canvas.draw()
pos0 = plt_ax[0].get_position()
pos1 = plt_ax[1].get_position()
plt_ax[0].set_position([pos1.x0, pos1.y0 + 1.1*pos0.height, pos1.width, pos0.height])
plt_ax[1].set_position([pos1.x0, pos1.y0 + 0.1*pos0.height, pos1.width, pos0.height])

plt_ax[0].grid(True, linestyle=':', dash_capstyle='round', \
dash_joinstyle='round', linewidth=ax_linewidth, color=[0.5, 0.5, 0.5])
plt_ax[1].grid(True, linestyle=':', dash_capstyle='round', \
dash_joinstyle='round', linewidth=ax_linewidth, color=[0.5, 0.5, 0.5])
plt_ax[0].tick_params(width=ax_linewidth, labelsize=ax_fontsize*1.5*0.85)
plt_ax[1].tick_params(width=ax_linewidth, labelsize=ax_fontsize*1.5*0.85)
plt_ax[0].annotate('', xy=(0.02, 0.55), xycoords='figure fraction', xytext=(0.02, 0.45), arrowprops=dict(arrowstyle="-", color='w'))
plt_ax[0].annotate('', xy=(0.98, 0.55), xycoords='figure fraction', xytext=(0.98, 0.45), arrowprops=dict(arrowstyle="-", color='w'))

for axis in ['top','bottom','left','right']:
  plt_ax[0].spines[axis].set_linewidth(ax_linewidth)
  plt_ax[1].spines[axis].set_linewidth(ax_linewidth)

figure.savefig(wav_nam, format='svg', bbox_inches='tight')
if save_png: figure.savefig((wav_nam[:-3] + 'png').replace(outfold, png_outfold), format='png', dpi=600, bbox_inches='tight')

sys_exit(0)


# ######################
# G --> Vs
# ######################
#~ In [1]: ro = 21200./1000./9.806
#~ In [2]: G = 445610.0
#~ In [3]: (G/ro)**0.5
#~ Out[3]: 453.99960830338017


# ######################
# points inside polygon
# ######################
#~ In [59]: import matplotlib.path as mpltPath
#~ In [60]: polygon = input_dict['stratum006']['nodes']
#~ In [61]: path = mpltPath.Path(polygon)
#~ In [62]: points = [[200.0, 170.1], [200.0, 270.1]]
#~ In [63]: inside = path.contains_points(points)
#~ In [64]: inside
#~ Out[64]: array([ True, False], dtype=bool)
#~ In [65]: inside2[0]
#~ Out[65]: True
#~ In [66]: inside2[1]
#~ Out[66]: False


# ######################
# c_error(s)
# ######################
#~ c_error.absolute_path
#~ c_error.create_from
#~ c_error.absolute_schema_path
#~ c_error.instance
#~ c_error.args
#~ c_error.message
#~ c_error.cause
#~ c_error.parent
#~ c_error.context
#~ c_error.path
#~ c_error.relative_path
#~ c_error.validator_value
#~ c_error.relative_schema_path
#~ c_error.schema
#~ c_error.schema_path
#~ c_error.validator
