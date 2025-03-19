#!/usr/bin/env python

import json
import copy
import argparse
from re import finditer as re_finditer
from io import StringIO as io_StringIO
from sys import exit as sys_exit
from sys import argv as sys_argv
from sys import stdout as sys_stdout
from os import path as os_path
from os import mkdir as os_mkdir
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.tri as mtri
from matplotlib import pyplot as plt
from matplotlib import cm as mcm
from matplotlib import colors as clr
#~ from matplotlib import transforms as trns
import math
from scipy import interpolate
from matplotlib.ticker import AutoLocator
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore",category=RuntimeWarning)

# #############
# CUSTOM FUNCT
# #############
from sys import path as sys_path
sys_path.append(os_path.join(*[os_path.dirname(sys_argv[0]), 'lib']))
from def_QUAD4M import lprint
from def_QUAD4M import par_plot
from def_QUAD4M import pre_plot
from def_QUAD4M import resp_spectra_acc
from def_QUAD4M import Fourier_spectra_amplitude
from def_QUAD4M import KonnoOhmachi
from def_QUAD4M import SP_resample
from def_QUAD4M import plot_border_inner
from def_QUAD4M import finalize_plot
from def_QUAD4M import is_valid_file
from def_QUAD4M import is_valid_dir
from def_QUAD4M import common_def

#~ ##############################

common_def_dict = common_def()

p = argparse.ArgumentParser(description='description: Generate 1) a serie of graphical-outputs in SVG format (Scalable Vector Graphics) of the QUAD4M outputs, and 2) a file, whose extension is "sd", containing the QUAD4M acceleration output in terms of displacement response spectra.')

p.add_argument("json", action="store", type=is_valid_file, help=common_def_dict['json'])
p.add_argument("Q4O", action="store", type=is_valid_file, help='QUAD4M main output file (denoted by extension "Q4O" in QUAD4M user''s manual)')
p.add_argument("Q4A", action="store", type=is_valid_file, help='QUAD4M acceleration output file (denoted by extension "Q4A" in QUAD4M user''s manual)')
p.add_argument("borders", action="store", type=is_valid_file, help='file "borders.txt" produced by pre-processor')
#~ p.add_argument("-d", "--directory", action="store", dest="out_folder", default='var', type=is_valid_dir, help=common_def_dict['out_folder'])
p.add_argument("-v", "--version", action="version", version=common_def_dict["version"], help=common_def_dict["version_h"])

opts = p.parse_args()

#~ ##############################

infile = opts.json
infile_O = opts.Q4O
infile_A = opts.Q4A
infile_B = opts.borders
#~ main_outfold = opts.out_folder
main_outfold = os_path.join(*[os_path.dirname(sys_argv[0]), 'var'])

#~ ##############################

S_filename = infile_A + '.sd.txt'
if os_path.isfile(S_filename):
    print 'WARNING: output file "' + S_filename + '" already exists'
    ans_YN = raw_input("Overwrite all output files [Y|N] [N]: ")
    if not ans_YN: sys_exit(1)
    elif ans_YN[0] in ['y', 'Y']: pass
    else: sys_exit(1)
F_filename = infile_A + '.fft.txt'
K_filename = infile_A + '.fft.ko.txt'
N_filename = infile_A + '.output_nodes.txt'

#~ ##############################

elemfile = os_path.join(*[os_path.dirname(sys_argv[0]), 'lib', 'json_elements.json'])

#~ ##############################

# JSON-input
with open(infile, 'r') as input_json: input_dict = json.load(input_json)
with open(elemfile, 'r') as c_file: json_elem_dict = json.load(c_file)

#~ ##############################

max_width = float(input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_msw"]])
max_semiwidth = max_width / 2.0

#~ ##############################

spec_outfold = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_jbf"]]

d_var_O = os_path.split(os_path.split(infile_O)[0])[-1]
if spec_outfold != d_var_O:
    print 'ERROR: directory "' + spec_outfold + '" specified in "' + infile + '" (field "job_folder") does not correspond to directory "' + d_var_O + '" where file "' + infile_O + '" is located'
    print 'NOTE: to avoid this error, consider to pass "' + infile_O.replace(d_var_O, spec_outfold) + '" instead of "' + infile_O + '"'
    sys_exit(1)
d_var_A = os_path.split(os_path.split(infile_A)[0])[-1]
if spec_outfold != d_var_A:
    print 'ERROR: directory "' + spec_outfold + '" specified in "' + infile + '" (field "job_folder") does not correspond to directory "' + d_var_A + '" where file "' + infile_A + '" is located'
    print 'NOTE: to avoid this error, consider to pass "' + infile_A.replace(d_var_A, spec_outfold) + '" instead of "' + infile_A + '"'
    sys_exit(1)
d_var_B = os_path.split(os_path.split(infile_B)[0])[-1]
if spec_outfold != d_var_B:
    print 'ERROR: directory "' + spec_outfold + '" specified in "' + infile + '" (field "job_folder") does not correspond to directory "' + d_var_B + '" where file "' + infile_B + '" is located'
    print 'NOTE: to avoid this error, consider to pass "' + infile_B.replace(d_var_B, spec_outfold) + '" instead of "' + infile_B + '"'
    sys_exit(1)

outfold = os_path.join(*[main_outfold, spec_outfold])
if not os_path.isdir(outfold):
    a_error_str = 'ERROR: directory "' + outfold + '" does not exist'
    print a_error_str
    sys_exit(1)

#~ ##############################

f_O = open(infile_O, 'r')
data_O = f_O.read()
f_O.close()
f_A = open(infile_A, 'r')
data_A = f_A.read()
f_A.close()
f_B = open(infile_B, 'r')
data_B = f_B.read()
f_B.close()

#~ ##############################

if data_O[0] == '\r': newline_sep = '\r\n'
elif data_O[0] == '\n': newline_sep = '\n'
else:
    b_error_str = 'ERROR: cannot recognize newline separator! Please, contact the developer.'
    print b_error_str
    sys_exit(1)

#~ ##############################


def extract_from_O(data_O, c_line_id, empty_line, c_dlm, c_dtype, c_names):
    hdr_elems_ndxs = [m.start() for m in re_finditer(c_line_id, data_O)]
    tmp_ndxs = [m.start() for m in re_finditer(empty_line, data_O[hdr_elems_ndxs[0]:])]
    elems_ndxs = [hdr_elems_ndxs[0]+tmp_ndxs[0]+len(empty_line),hdr_elems_ndxs[0]+tmp_ndxs[1]]
    elems_O = data_O[elems_ndxs[0]:elems_ndxs[1]]
    s_elems_O = io_StringIO(u"" + elems_O)
    mat_elems_O = np.genfromtxt(fname=s_elems_O, dtype=c_dtype, delimiter=c_dlm, names=c_names)
    s_elems_O.close()
    return mat_elems_O

#~ ##############################

elems_line_id = '   ELM     NODE 1  NODE 2  NODE 3  NODE 4  MAT.TYPE  DENSITY    POISSON R.     GMX    SH. MODULUS  DAMP. RATIO     AREA'
empty_line = newline_sep + newline_sep
elems_dlm = (8,)*6+(12,)*6
elems_dtype = ','.join([','.join(["i8"] * 6), ','.join(["f8"] * 6)])
elems_names = ['N', 'NP1', 'NP2', 'NP3', 'NP4', 'TYPE', 'DENS', 'PO', 'GMX', 'G', 'XL', 'AREA']

mat_elems_O = extract_from_O(data_O=data_O, c_line_id=elems_line_id, \
empty_line=empty_line, c_dlm=elems_dlm, c_dtype=elems_dtype, c_names=elems_names)

#~ ##############################

node_line_id = '    NODE           XORD           YORD       TRIBUTARY LEN'
empty_line = newline_sep + newline_sep
node_dlm = (8,)*1+(15,)*2#+(15,)*1
node_dtype = ','.join([','.join(["i8"] * 1), ','.join(["f8"] * 2)])#, ','.join(["S5"] * 1)])
node_names = ['NODE', 'XORD', 'YORD']#, 'TRIBUTARY']

mat_node_O = extract_from_O(data_O=data_O, c_line_id=node_line_id, \
empty_line=empty_line, c_dlm=node_dlm, c_dtype=node_dtype, c_names=node_names)

#~ ##############################

pga_line_id = '          NODE     XORD     YORD         X-ACC       AT TIME           Y-ACC       AT TIME'
empty_line = newline_sep + newline_sep
pga_dlm = (14,)*1+(9,)*2+(14,)*4
pga_dtype = ','.join([','.join(["i8"] * 1), ','.join(["f8"] * 6)])
pga_names = ['NODE', 'XORD', 'YORD', 'XACC', 'XTIME', 'YACC', 'YTIME']

mat_pga_O = extract_from_O(data_O=data_O, c_line_id=pga_line_id, \
empty_line=empty_line, c_dlm=pga_dlm, c_dtype=pga_dtype, c_names=pga_names)

#~ ##############################

strains_line_id = '          ELM           SIG-X          SIG-Y         SIG-XY         EPS-MAX        AT TIME'
empty_line = newline_sep + newline_sep + newline_sep
strains_dlm = (14,)*1+(15,)*5
strains_dtype = ','.join([','.join(["i8"] * 1), ','.join(["f8"] * 5)])
strains_names = ['ELM', 'SIGX', 'SIGY', 'SIGXY', 'EPSMAX', 'TIME']

mat_strains_O = extract_from_O(data_O=data_O, c_line_id=strains_line_id, \
empty_line=empty_line, c_dlm=strains_dlm, c_dtype=strains_dtype, c_names=strains_names)

#~ ##############################

tol_nod = 0.2
s_B = io_StringIO(u"" + data_B)
mat_B = np.genfromtxt(fname=s_B, dtype='f8,f8,f8,f8', delimiter=(10, 10, 10, 10), names=('cs00', 'cs01', 'cs10', 'cs11'))
c_s_B = []
for ndx_B in range(len(mat_B)):
    if np.isnan(mat_B[ndx_B][0]): break
    c_s_B.append([[mat_B[ndx_B][0], mat_B[ndx_B][1]], [mat_B[ndx_B][2], mat_B[ndx_B][3]]])
c_s_B_surf = []
for ndx_B_surf in range(ndx_B+1,len(mat_B)):
    if np.isnan(mat_B[ndx_B_surf][0]): break
    c_s_B_surf.append([[mat_B[ndx_B_surf][0], mat_B[ndx_B_surf][1]], [mat_B[ndx_B_surf][2], mat_B[ndx_B_surf][3]]])
c__X = []
for ndx__X in range(ndx_B_surf+1,len(mat_B)):
    if np.isnan(mat_B[ndx__X][0]): break
    c__X.append([mat_B[ndx__X][0], mat_B[ndx__X][1]])
c__X = [x for x in c__X if not np.allclose(x, [-999.999, -999.999], rtol=.0, atol=tol_nod)]
c__Y = []
for ndx__Y in range(ndx__X+1,len(mat_B)):
    if np.isnan(mat_B[ndx__Y][0]): break
    c__Y.append([mat_B[ndx__Y][0], mat_B[ndx__Y][1]])
c__Y = [x for x in c__Y if not np.allclose(x, [-999.999, -999.999], rtol=.0, atol=tol_nod)]
c_XY = []
for ndx_XY in range(ndx__Y+1,len(mat_B)):
    if np.isnan(mat_B[ndx_XY][0]): break
    c_XY.append([mat_B[ndx_XY][0], mat_B[ndx_XY][1]])
c_XY = [x for x in c_XY if not np.allclose(x, [-999.999, -999.999], rtol=.0, atol=tol_nod)]

#~ ##############################

xy_triang_ndxs_O = []
mat_pars_O = []
for c_k in xrange(len(mat_elems_O)):
    c_elem = mat_elems_O[c_k]
    c_elem_s = mat_strains_O[c_k]
    c_elem_sl = list(c_elem_s.item())
    c_XACC_1, c_YACC_1 = mat_pga_O[c_elem[1]-1][3], mat_pga_O[c_elem[1]-1][5]
    c_XACC_2, c_YACC_2 = mat_pga_O[c_elem[2]-1][3], mat_pga_O[c_elem[2]-1][5]
    c_XACC_3, c_YACC_3 = mat_pga_O[c_elem[3]-1][3], mat_pga_O[c_elem[3]-1][5]
    c_XACC_4, c_YACC_4 = mat_pga_O[c_elem[4]-1][3], mat_pga_O[c_elem[4]-1][5]
    if c_elem[3] == c_elem[4]:
        xy_triang_ndxs_O.append([c_elem[1]-1, c_elem[2]-1, c_elem[3]-1])
        c_XACC = np.mean([c_XACC_1, c_XACC_2, c_XACC_3])
        c_YACC = np.mean([c_YACC_1, c_YACC_2, c_YACC_3])
        mat_pars_O.append(c_elem_sl + [c_XACC, c_YACC])
    else:
        xy_triang_ndxs_O.append([c_elem[1]-1, c_elem[2]-1, c_elem[4]-1])
        c_XACC = np.mean([c_XACC_1, c_XACC_2, c_XACC_4])
        c_YACC = np.mean([c_YACC_1, c_YACC_2, c_YACC_4])
        mat_pars_O.append(c_elem_sl + [c_XACC, c_YACC])
        xy_triang_ndxs_O.append([c_elem[2]-1, c_elem[3]-1, c_elem[4]-1])
        c_XACC = np.mean([c_XACC_2, c_XACC_3, c_XACC_4])
        c_YACC = np.mean([c_YACC_2, c_YACC_3, c_YACC_4])
        mat_pars_O.append(c_elem_sl + [c_XACC, c_YACC])

#~ ##############################

print " "
lprint('identifying output nodes')

surf_nodes = []
for c_h in c_s_B_surf:
    c_h_0_add = False
    for c_n in surf_nodes:
        if np.allclose(c_h[0], c_n, rtol=.0, atol=tol_nod):
            c_h_0_add = True
            break
    if not c_h_0_add: surf_nodes.append(c_h[0])
    c_h_1_add = False
    for c_n in surf_nodes:
        if np.allclose(c_h[1], c_n, rtol=.0, atol=tol_nod):
            c_h_1_add = True
            break
    if not c_h_1_add: surf_nodes.append(c_h[1])

#~ ##############################

c__X_all = c_XY + c__X
c__X_bll = []
for c_h in c__X_all:
    c_h_0_add = False
    for c_n in c__X_bll:
        if np.allclose(c_h, c_n, rtol=.0, atol=tol_nod):
            c_h_0_add = True
            break
    if not c_h_0_add: c__X_bll.append(c_h)

c__X_nodes_surf = []
for cc__X in c__X_bll:
    [c__X_nodes_surf.append(x) for x in surf_nodes if np.allclose([x[0], x[1]], [cc__X[0], cc__X[1]], rtol=.0, atol=tol_nod)]

c__X_nodes = []
for cc__X in c__X_nodes_surf:
    [c__X_nodes.append(x) for x in mat_node_O if np.allclose([x[1], x[2]], [cc__X[0], cc__X[1]], rtol=.0, atol=tol_nod)]

c__X_nodes_all = []
for cc__X in c__X_all:
    [c__X_nodes_all.append(x) for x in mat_node_O if np.allclose([x[1], x[2]], [cc__X[0], cc__X[1]], rtol=.0, atol=tol_nod)]

#~ ##############################

c__Y_all = c_XY + c__Y
c__Y_bll = []
for c_h in c__Y_all:
    c_h_0_add = False
    for c_n in c__Y_bll:
        if np.allclose(c_h, c_n, rtol=.0, atol=tol_nod):
            c_h_0_add = True
            break
    if not c_h_0_add: c__Y_bll.append(c_h)

c__Y_nodes_surf = []
for cc__Y in c__Y_bll:
    [c__Y_nodes_surf.append(x) for x in surf_nodes if np.allclose([x[0], x[1]], [cc__Y[0], cc__Y[1]], rtol=.0, atol=tol_nod)]

c__Y_nodes = []
for cc__Y in c__Y_nodes_surf:
    [c__Y_nodes.append(x) for x in mat_node_O if np.allclose([x[1], x[2]], [cc__Y[0], cc__Y[1]], rtol=.0, atol=tol_nod)]

c__Y_nodes_all = []
for cc__Y in c__Y_all:
    [c__Y_nodes_all.append(x) for x in mat_node_O if np.allclose([x[1], x[2]], [cc__Y[0], cc__Y[1]], rtol=.0, atol=tol_nod)]

#~ ##############################

c__X_nodes_numb = map(str, map(list, zip(*c__X_nodes_all))[0])
c__Y_nodes_numb = map(str, map(list, zip(*c__Y_nodes_all))[0])
c__X_nodes_numb = [s + 'X' for s in c__X_nodes_numb]
c__Y_nodes_numb = [s + 'Y' for s in c__Y_nodes_numb]
c__nodes_numb = c__X_nodes_numb + c__Y_nodes_numb
for c__k in range(len(c__nodes_numb)):
    c__node_numb = c__nodes_numb[c__k]
    if len(c__node_numb) != 6:
        c__nodes_numb[c__k] = '0'*(6-len(c__node_numb)) + c__nodes_numb[c__k]
c__nodes_numb = list(set(c__nodes_numb))
c__nodes_numb.sort()
for c__k in range(len(c__nodes_numb)):
    while c__nodes_numb[c__k][0] == '0':
        c__nodes_numb[c__k] = c__nodes_numb[c__k][1:]
c__nodes_numb = ['Node' + s for s in c__nodes_numb]
c__nodes_numb = ['Timesec'] + c__nodes_numb

#~ ##############################

s_acc_A = io_StringIO(u"" + data_A)

for i, hdr_acc_A in enumerate(s_acc_A):
    if i == 2: break
h_hdr_acc_A = io_StringIO(u"" + hdr_acc_A)
acc_names_hdr = tuple(np.genfromtxt(fname=h_hdr_acc_A, delimiter=(10), dtype='str'))
acc_names_hdr = [x.strip().replace('-','').replace(' ','') for x in acc_names_hdr]
acc_names = c__nodes_numb

for acc_ndx in range(1, len(acc_names_hdr)):
    acc_name_hdr = acc_names_hdr[acc_ndx]
    if '*' in acc_name_hdr: break
    acc_name = acc_names[acc_ndx]
    if acc_name != acc_name_hdr:
        print 'ERROR: "' + acc_name_hdr + '" in Q4A does not have correspondance in Q4O output-nodes'
        print '-- '
        print '      Q4A header: ' + ' '.join(acc_names_hdr[1:])
        print '-- '
        print 'Q4O output-nodes: ' + ' '.join(acc_names[1:])
        sys_exit(1)        

acc_dlm = (10,)*len(acc_names)
acc_dtype = ','.join(["f8"] * len(acc_names))

mat_acc_A = np.genfromtxt(fname=s_acc_A, dtype=acc_dtype, delimiter=acc_dlm, names=acc_names)

#~ ##############################

nu = 50
K_nu = 100
uti_g = 980.6 #[cm/s^2]
uti_gm = uti_g/100.0 #[m/s^2]
c_gm = float(input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_fft"]])
c_fm = max(12.,float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_mfe"]])*2.0)
dt = float(input_dict[json_elem_dict["wave"]][json_elem_dict["wave_tss"]])
out_npts = int(np.exp2(np.ceil(np.log2(len(mat_acc_A))))/2)
mat_F_A = copy.deepcopy(mat_acc_A[:out_npts]) #[g*s]
mat_K_A = copy.deepcopy(mat_acc_A[:K_nu]) #[g*s]
for t_name in acc_names[1:]:
    c_string = 'calculating/smoothing Fourier spectra at ' + t_name
    sys_stdout.write("\r{0}".format("%s" % lprint(c_string, return_str=True)))
    sys_stdout.flush()
    mat_F_A[t_name], freq_A = Fourier_spectra_amplitude(dt, mat_acc_A[t_name])
    K_SP_A = KonnoOhmachi(mat_F_A[t_name], freq_A, b=40.0)
    mat_K_A[t_name], freq_K = SP_resample(K_SP_A, freq_A, min_freq=c_gm, max_freq=c_fm)
mat_F_A[acc_names[0]] = freq_A
mat_K_A[acc_names[0]] = freq_K
sys_stdout.write("\r{0}".format("%s" % ""))
sys_stdout.flush()
lprint('Fourier spectra have been calculated')
mat_SA_A = copy.deepcopy(mat_acc_A[:nu]) #[cm/s^2]
mat_SD_A = copy.deepcopy(mat_acc_A[:nu]) #[cm]
for t_name in acc_names[1:]:
    c_string = 'calculating response spectra at ' + t_name
    sys_stdout.write("\r{0}".format("%s" % lprint(c_string, return_str=True)))
    sys_stdout.flush()
    mat_SA_A[t_name], mat_SD_A[t_name], T = \
    resp_spectra_acc(dt,mat_acc_A[t_name]*uti_g,nu=nu)
mat_SA_A[acc_names[0]] = T
mat_SD_A[acc_names[0]] = T
sys_stdout.write("\r{0}".format("%s" % ""))
sys_stdout.flush()
lprint('response spectra have been calculated')
mat_PSV_A = copy.deepcopy(mat_SD_A) #[cm/s]
for t_name in acc_names[1:]:
    mat_PSV_A[t_name][1:] = np.true_divide(mat_SD_A[t_name][1:],T[1:]) * 2 * np.pi
TH = [0.1, 0.1044, 0.1149, 0.1265, 0.1392, 0.1532, 0.1687, 0.1857, 0.2044, \
0.2249, 0.2476, 0.2725, 0.3, 0.3302, 0.3634, 0.4, 0.4403, 0.4846, 0.5334, \
0.5871, 0.6462, 0.7113, 0.7829, 0.8618, 0.9485, 1.0441, 1.1492, 1.2649, \
1.3923, 1.5325, 1.6868, 1.8566, 2.0436, 2.2494, 2.4759, 2.5]
mat_A_A = copy.deepcopy(mat_acc_A)
par_dict = {}
par_dict['PGA'] = {} #[g]
par_dict['PGV'] = {} #[cm/s]
par_dict['PGD'] = {} #[cm]
par_dict['HI'] = {}  #[cm]
par_dict['AI'] = {}  #[m/s]
par_dict['DUR'] = {} #[s]
for t_name in acc_names[1:]:
    par_dict['PGA'][t_name] = np.max(np.abs(mat_acc_A[t_name]))
    par_dict['PGV'][t_name] = \
    np.max(np.abs(np.cumsum(mat_acc_A[t_name]*uti_g)*dt))
    par_dict['PGD'][t_name] = \
    np.max(np.abs(np.cumsum(np.cumsum(mat_acc_A[t_name]*uti_g)*dt)*dt))
    fH = interpolate.interp1d(mat_PSV_A[acc_names[0]], mat_PSV_A[t_name])
    c_PSV = fH(TH)
    par_dict['HI'][t_name] = np.trapz(c_PSV, x=TH)
    for a_var in range(1,len(mat_acc_A)):
        mat_A_A[t_name][a_var] = mat_A_A[t_name][a_var-1] + \
        ((mat_acc_A[t_name][a_var]*uti_gm)**2) * dt * (np.pi/(2.*uti_gm))
    par_dict['AI'][t_name] = mat_A_A[t_name][-1]
    mat_A_tn = mat_A_A[t_name] / par_dict['AI'][t_name]
    ndx05 = (np.abs(mat_A_tn - 0.05)).argmin()
    ndx95 = (np.abs(mat_A_tn - 0.95)).argmin()
    par_dict['DUR'][t_name] = \
    mat_A_A[acc_names[0]][ndx95] - mat_A_A[acc_names[0]][ndx05]
    par_dict['AI'][t_name] = par_dict['AI'][t_name]

#~ ##############################

xy_mtri_O = mtri.Triangulation(mat_node_O['XORD'], mat_node_O['YORD'], triangles=xy_triang_ndxs_O)
xy_triang_STRESS = map(list, zip(*mat_pars_O))[3]
xy_triang_STRAIN = map(list, zip(*mat_pars_O))[4]
xy_triang_STRAIN = [0.0001 if not v else v for v in xy_triang_STRAIN]
xy_triang_XACC = map(list, zip(*mat_pars_O))[6]
xy_triang_YACC = map(list, zip(*mat_pars_O))[7]

par_STRESS = os_path.join(*[outfold, 'plot_STRESS.svg'])
par_STRAIN = os_path.join(*[outfold, 'plot_STRAIN.svg'])
par_XACC = os_path.join(*[outfold, 'plot_XACC.svg'])
par_YACC = os_path.join(*[outfold, 'plot_YACC.svg'])

save_png = input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_png"]]
if save_png:
    png_outfold = os_path.join(*[outfold, 'PNG_images'])
    if not os_path.isdir(png_outfold + ''): os_mkdir(png_outfold)
else: png_outfold = None

line_thicknesses_scale = input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_sca"]][json_elem_dict["pltp_sca_lt"]]
fonts_scale = input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_sca"]][json_elem_dict["pltp_sca_fo"]]
symbols_scale = input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_sca"]][json_elem_dict["pltp_sca_sy"]]

c_linewidth = 1.0 * line_thicknesses_scale
ax_linewidth = 0.5 * line_thicknesses_scale
ax_fontsize = 8.0 * fonts_scale
leg_fontsize = 7.0 * fonts_scale
ax_markersize = 1.0 * symbols_scale

plt_ax = plt.gca()

if input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_bws"]]: cmap = mcm.get_cmap('Greys')
else: cmap = mcm.get_cmap('jet')

# Peak elements stresses [N/M^2]
lprint_str = "STRESS plot"
xy_triang_PAR = xy_triang_STRESS
plt_title = 'Peak elements stresses [N/M^2]'
par_name = par_STRESS
c_bar_ticks = [100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
c_fmt_punct = '{:.0f}'
c_minVal, c_maxVal, logn = 100.0, 1000000.0, True
par_plot(plt, plt_ax, input_dict, json_elem_dict, lprint_str, xy_mtri_O, xy_triang_PAR, \
plt_title, par_name, c_minVal, c_maxVal, c_bar_ticks, c_fmt_punct, cmap, logn, \
c_linewidth, leg_fontsize, ax_linewidth, ax_fontsize, c_s_B, save_png, outfold, png_outfold)

# Peak elements strains [%]
lprint_str = "STRAIN plot"
xy_triang_PAR = xy_triang_STRAIN
plt_title = 'Peak elements strains [%]'
par_name = par_STRAIN
c_bar_ticks = [0.001, 0.01, 0.1, 1.0]
c_fmt_punct = '{:.0g}'
c_minVal, c_maxVal, logn = 0.001, 1.0, True
par_plot(plt, plt_ax, input_dict, json_elem_dict, lprint_str, xy_mtri_O, xy_triang_PAR, \
plt_title, par_name, c_minVal, c_maxVal, c_bar_ticks, c_fmt_punct, cmap, logn, \
c_linewidth, leg_fontsize, ax_linewidth, ax_fontsize, c_s_B, save_png, outfold, png_outfold)

# Peak horizontal acceleration [g]
lprint_str = "XACC plot"
xy_triang_PAR = xy_triang_XACC
plt_title = 'Peak horizontal acceleration [g]'
par_name = par_XACC
c_bar_ticks = [0.001, 0.01, 0.1, 1.0]
c_fmt_punct = '{:.0g}'
c_minVal, c_maxVal, logn = 0.001, 1.0, True
par_plot(plt, plt_ax, input_dict, json_elem_dict, lprint_str, xy_mtri_O, xy_triang_PAR, \
plt_title, par_name, c_minVal, c_maxVal, c_bar_ticks, c_fmt_punct, cmap, logn, \
c_linewidth, leg_fontsize, ax_linewidth, ax_fontsize, c_s_B, save_png, outfold, png_outfold)

# Peak vertical acceleration [g]
lprint_str = "YACC plot"
xy_triang_PAR = xy_triang_YACC
plt_title = 'Peak vertical acceleration [g]'
par_name = par_YACC
c_bar_ticks = [0.001, 0.01, 0.1, 1.0]
c_fmt_punct = '{:.0g}'
c_minVal, c_maxVal, logn = 0.001, 1.0, True
par_plot(plt, plt_ax, input_dict, json_elem_dict, lprint_str, xy_mtri_O, xy_triang_PAR, \
plt_title, par_name, c_minVal, c_maxVal, c_bar_ticks, c_fmt_punct, cmap, logn, \
c_linewidth, leg_fontsize, ax_linewidth, ax_fontsize, c_s_B, save_png, outfold, png_outfold)

#~ ##############################

c__Y_par = []
for cc__Y in c__Y_nodes:
    c__Y_par.append([cc__Y[1], \
    cc__Y[2], \
    par_dict['PGA']['Node' + str(cc__Y[0]) + 'Y'], \
    par_dict['PGV']['Node' + str(cc__Y[0]) + 'Y'], \
    par_dict['PGD']['Node' + str(cc__Y[0]) + 'Y'], \
    par_dict['HI']['Node' + str(cc__Y[0]) + 'Y'], \
    par_dict['AI']['Node' + str(cc__Y[0]) + 'Y'], \
    par_dict['DUR']['Node' + str(cc__Y[0]) + 'Y']])
c__Y_par_w = map(list, zip(*c__Y_par))
c__X_par = []
for cc__X in c__X_nodes:
    c__X_par.append([cc__X[1], \
    cc__X[2], \
    par_dict['PGA']['Node' + str(cc__X[0]) + 'X'], \
    par_dict['PGV']['Node' + str(cc__X[0]) + 'X'], \
    par_dict['PGD']['Node' + str(cc__X[0]) + 'X'], \
    par_dict['HI']['Node' + str(cc__X[0]) + 'X'], \
    par_dict['AI']['Node' + str(cc__X[0]) + 'X'], \
    par_dict['DUR']['Node' + str(cc__X[0]) + 'X']])
c__X_par_w = map(list, zip(*c__X_par))

#~ ##############################

c_c_tX = []
c_c_tY = []
for c_c_t in c_s_B:
    c_c_tX.append(c_c_t[0][0])
    c_c_tX.append(c_c_t[1][0])
    c_c_tY.append(c_c_t[0][1])
    c_c_tY.append(c_c_t[1][1])

x_axin_min = min(c_c_tX)
x_axin_max = max(c_c_tX)
y_axin_min = min(c_c_tY)
y_axin_max = max(c_c_tY)

#~ ##############################

#~ lprint('preparing "par" plots ...')

plt.clf()
figure, plt_ax = plt.subplots(2, 1, sharex=True)
mult_val = 1.05

lineH, = plt_ax[0].plot(c__X_par_w[0], c__X_par_w[2], marker='o', color='black', \
markeredgecolor=[0.5, 0.5, 0.5], markeredgewidth=0.9*ax_markersize, linestyle='None', \
markersize=8.0*ax_markersize, label='Horiz.')
lineV, = plt_ax[0].plot(c__Y_par_w[0], c__Y_par_w[2], marker='o', color='red', \
markeredgecolor=[0.0, 0.0, 0.0], markeredgewidth=0.9*ax_markersize, linestyle='None', \
markersize=8.0*ax_markersize, label='Vert.')
plt_leg0 = plt_ax[0].legend(fontsize=leg_fontsize, loc=1, facecolor=[1., 1., 1.], edgecolor=[1., 1., 1.], markerscale=0.8) #frameon=False
plt_ax[0].grid(True, linestyle=':', dash_capstyle='round', \
dash_joinstyle='round', linewidth=ax_linewidth, color=[0.5, 0.5, 0.5])
plt_ax[0].set_xlim([x_axin_min, x_axin_max])
plt_ax[0].tick_params(which='both', width=ax_linewidth, labelsize=ax_fontsize*0.85)
plt.setp(plt_ax[0].get_xticklabels(), visible=False)
plt_ax[0].spines['bottom'].set_visible(False)

for cc_s in c_s_B:
    plt_ax[1], n_c_s = plot_border_inner(plt_ax[1], [cc_s], 0, 1, [], c_ol=[.0, .0, .0], l_in=1.0*ax_linewidth)
plt_ax[1].plot(c__X_par_w[0], c__X_par_w[1], marker='P', color=[0.0, 1.0, 0.0], \
markeredgecolor=[0.0, 0.0, 0.0], markeredgewidth=0.6*ax_markersize, linestyle='None', \
markersize=4.0*ax_markersize, label="Output on ground")
plt_ax[1].legend(fontsize=leg_fontsize, loc=3, facecolor=[1., 1., 1.], edgecolor=[1., 1., 1.], markerscale=1.0)
plt, plt_ax[1] = finalize_plot(plt, plt_ax[1], ax_linewidth, ax_fontsize, box='box-forced', plt_box=True)
plt_ax[1].set_xlim([x_axin_min, x_axin_max])
plt_ax[1].set_ylim([y_axin_min/mult_val, y_axin_max*mult_val])

plt_ax[1].set_xlabel('[m]', fontsize=ax_fontsize)
plt_ax[1].yaxis.tick_right()
plt_ax[1].tick_params(width=ax_linewidth, labelsize=ax_fontsize*0.85)
plt_ax[1].annotate('', xy=(0.02, 0.55), xycoords='figure fraction', xytext=(0.02, 0.45), arrowprops=dict(arrowstyle="-", color='w'))
plt_ax[1].annotate('', xy=(0.98, 0.55), xycoords='figure fraction', xytext=(0.98, 0.45), arrowprops=dict(arrowstyle="-", color='w'))
plt_ax[1].yaxis.set_label_position("right")
plt_ax[1].set_ylabel('[m]', fontsize=ax_fontsize, rotation=270, verticalalignment='bottom')
plt_ax[1].spines['top'].set_visible(False)

plt_ax[1].axhline(y_axin_max*mult_val, linestyle=':', dash_capstyle='round', \
dash_joinstyle='round', linewidth=ax_linewidth, color=[0.5, 0.5, 0.5])

#~ plt_ax[1].tick_params(width=ax_linewidth, labelsize=ax_fontsize)

for axis in ['top','bottom','left','right']:
  plt_ax[0].spines[axis].set_linewidth(ax_linewidth)
  plt_ax[1].spines[axis].set_linewidth(ax_linewidth)

figure.canvas.draw()
pos0 = plt_ax[0].get_position()
pos1 = plt_ax[1].get_position()
plt_ax[0].set_position([pos1.x0, pos0.y0 - 0.5*pos1.height, pos1.width, pos0.height])
plt_ax[1].set_position([pos1.x0, pos0.y0 - 1.5*pos1.height, pos1.width, pos1.height])

#~ ##############################

par_PGA = os_path.join(*[outfold, 'profile_PGA.svg'])
par_PGV = os_path.join(*[outfold, 'profile_PGV.svg'])
par_PGD = os_path.join(*[outfold, 'profile_PGD.svg'])
par_HI = os_path.join(*[outfold, 'profile_HI.svg'])
par_AI = os_path.join(*[outfold, 'profile_AI.svg'])
par_DUR = os_path.join(*[outfold, 'profile_DUR.svg'])
par_ACC = os_path.join(*[outfold, 'profile_ACC.svg'])
par_SA = os_path.join(*[outfold, 'profile_SA.svg'])
par_names = [\
[par_PGA, 'PGA plot', 'PGA [g]', 'Peak Ground Acceleration'], \
[par_PGV, 'PGV plot', 'PGV [cm/s]', 'Peak Ground Velocity'], \
[par_PGD, 'PGD plot', 'PGD [cm]', 'Peak Ground Displacement'], \
[par_HI, 'HI plot', 'HI [cm]', 'Housner Intensity'], \
[par_AI, 'AI plot', 'AI [m/s]', 'Arias Intensity'], \
[par_DUR, 'DUR plot', 'DUR [s]', 'Duration'], \
[par_ACC, 'ACC plot', 'Time [s]', 'Horizontal component acceleration time histories'], \
[par_SA, 'SA plot', 'Period [s]', 'Horizontal component acceleration response spectra [g]']
]

#~ ##############################

for c_val in range(2,8):
    c_val_n = c_val - 2
    lprint(par_names[c_val_n][1])
    lineH.set_ydata(c__X_par_w[c_val])
    lineV.set_ydata(c__Y_par_w[c_val])
    plt_ax[0].set_ylim([.0, (mult_val**4)*max(max(c__X_par_w[c_val]),max(c__Y_par_w[c_val]))])
    plt_ax[0].set_ylabel(par_names[c_val_n][2], fontsize=ax_fontsize)
    plt_ax[0].set_title(par_names[c_val_n][3], fontsize=ax_fontsize)

    figure.savefig(par_names[c_val_n][0], format='svg', bbox_inches='tight')
    if save_png: figure.savefig((par_names[c_val_n][0][:-3] + 'png').replace(outfold, png_outfold), format='png', dpi=600, bbox_inches='tight')

#~ ##############################

if input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_bwc"]]: cmap = mcm.get_cmap('Greys')
else: cmap = mcm.get_cmap('jet')

lprint(par_names[-1][1])

lineH.set_ydata(None)
lineV.set_ydata(None)
plt_leg0.remove()

min_ax_v = 0.04
plt_ax[0].set_ylim([min_ax_v, 4.])
plt_ax[0].set_yscale('log')
plt_ax[0].set_yticks([0.04, 0.1, 0.4, 1, 4])
plt_ax[0].set_yticklabels(['0.04', '0.1', '0.4', '1', '4'])
plt_ax[0].set_ylabel(par_names[-1][2], fontsize=ax_fontsize)
plt_ax[0].set_title(par_names[-1][3], fontsize=ax_fontsize)
plt_ax[0].tick_params(labelsize=ax_fontsize*0.85)

c_bar_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
c_fmt_punct = '{:.1f}'
c_minVal, c_maxVal = 0.0, 1.0

len_c__Y = len(c__Y_nodes)
c__dist = [min(max_semiwidth,(c__Y_nodes[0][1] - x_axin_min)/2)]
if len_c__Y > 1:
    for c__ndx in range(1,len_c__Y):
        c__dist.append(min(max_semiwidth,(c__Y_nodes[c__ndx][1] - c__Y_nodes[c__ndx-1][1])/2))
    c__dist.append(min(max_semiwidth,(x_axin_max - c__Y_nodes[-1][1])/2))

vals_X = []
for h__ndx in range(1,len(mat_SA_A[acc_names[0]])):
    vals_X.append(mat_SA_A[acc_names[0]][h__ndx] - (mat_SA_A[acc_names[0]][h__ndx] - mat_SA_A[acc_names[0]][h__ndx-1])/2)
vals_X.append(mat_SA_A[acc_names[0]][-1])
vals_X[0] = mat_SA_A[acc_names[0]][1]
vals_X = np.array(vals_X)

vals_Xd = []
vals_Yd = []
vals_Zd = []
vals_Kd = []
X__ndx = 0
for c__ndx in range(len_c__Y):
    cc_str = 'Node' + str(c__Y_nodes[c__ndx][0]) + 'Y'
    vals_Ymin = c__Y_nodes[c__ndx][1] - c__dist[c__ndx]
    vals_Ymax = c__Y_nodes[c__ndx][1] + c__dist[c__ndx]
    for J__ndx in range(len(vals_X)-1):
        H__ndx = X__ndx*2
        vals_Xd.append(vals_X[J__ndx])
        vals_Xd.append(vals_X[J__ndx])
        vals_Yd.append(vals_Ymin)
        vals_Yd.append(vals_Ymax)
        vals_Kd.append([H__ndx+0, H__ndx+1, H__ndx+2])
        vals_Kd.append([H__ndx+1, H__ndx+3, H__ndx+2])
        vals_Zd.append(mat_SA_A[cc_str][J__ndx+1] / uti_g)
        vals_Zd.append(mat_SA_A[cc_str][J__ndx+1] / uti_g)
        X__ndx += 1
    vals_Xd.append(vals_X[J__ndx+1])
    vals_Xd.append(vals_X[J__ndx+1])
    vals_Yd.append(vals_Ymin)
    vals_Yd.append(vals_Ymax)
    X__ndx += 1

xy_mtri_SA = mtri.Triangulation(vals_Yd, vals_Xd, triangles=vals_Kd)

tpc = plt_ax[0].tripcolor(xy_mtri_SA, vals_Zd, shading='flat', cmap=cmap, \
vmin=c_minVal, vmax=c_maxVal)
#~ divider = make_axes_locatable(plt_ax[0])
#~ cax = divider.append_axes("right", size="5%", pad=0.05)
#~ cax = figure.add_axes([pos1.x0, pos1.y0, pos1.width, pos1.height])
cax = figure.add_axes([0.87, 0.52, 0.04, 0.25])

c_bar = plt.colorbar(tpc, ax=plt_ax[0], cax=cax)#, orientation='horizontal')
c_bar.ax.tick_params(width=ax_linewidth, labelsize=ax_fontsize*0.85)
c_bar.outline.set_linewidth(ax_linewidth)
c_bar.set_ticks(c_bar_ticks)
c_bar_ticklabels = map(lambda x: c_fmt_punct.format(x), c_bar_ticks)
c_bar.set_ticklabels(c_bar_ticklabels)

# ~ bbox_props = dict(fc="w", ec="w")
#~ cax_ann = plt_ax[0].annotate('Sa [g]', xy=(0.89, 0.805), xycoords='figure fraction', \
#~ horizontalalignment='left', verticalalignment='top', bbox=bbox_props, \
#~ fontsize=ax_fontsize)

plt_ax[1].axhline(y_axin_max*mult_val, linestyle=':', dash_capstyle='round', \
dash_joinstyle='round', linewidth=ax_linewidth, color=[0.5, 0.5, 0.5])
plt_ax[0].axhline(min_ax_v, linestyle=':', dash_capstyle='round', \
dash_joinstyle='round', linewidth=ax_linewidth, color=[0.5, 0.5, 0.5])
plt_ax[0].grid(True, linestyle=':', dash_capstyle='round', \
dash_joinstyle='round', linewidth=ax_linewidth, color=[0.5, 0.5, 0.5])

figure.savefig(par_names[-1][0], format='svg', bbox_inches='tight')
if save_png: figure.savefig((par_names[-1][0][:-3] + 'png').replace(outfold, png_outfold), format='png', dpi=600, bbox_inches='tight')

#~ ##############################

lprint('FFT plot')

tpc.remove()
cax.remove()

vals_M = []
for a__ndx in range(1,len(acc_names)): vals_M.append(np.max(mat_K_A[acc_names[a__ndx]]))

if c_fm > 10.0:
    plt_ax[0].set_yticks([0.04, 0.1, 0.4, 1, 4, 10])
    plt_ax[0].set_yticklabels(['0.04', '0.1', '0.4', '1', '4', '10'])

if c_fm < 4.0:
    plt_ax[0].set_yticks([0.04, 0.1, 0.4, 1])
    plt_ax[0].set_yticklabels(['0.04', '0.1', '0.4', '1'])

min_ax_v = c_gm
plt_ax[0].set_ylim([min_ax_v, c_fm])
factor_M = 10 ** -int(math.floor(np.log10(np.max(vals_M))))
val_M = math.ceil(np.max(vals_M) * factor_M) / factor_M
c_bar_ticks = [val_M/1000., val_M/100., val_M/10., val_M]
c_minVal, c_maxVal = val_M/1000., val_M
c_fmt_punct = '{:.0e}'

plt_ax[0].set_ylabel('Frequency [Hz]', fontsize=ax_fontsize)
plt_ax[0].set_title('Horizontal component Fourier spectra [g*s]', fontsize=ax_fontsize)

vals_X = []
for h__ndx in range(1,len(mat_K_A[acc_names[0]])):
    vals_X.append(mat_K_A[acc_names[0]][h__ndx] - (mat_K_A[acc_names[0]][h__ndx] - mat_K_A[acc_names[0]][h__ndx-1])/2)
vals_X.append(mat_K_A[acc_names[0]][-1])
vals_X[0] = mat_K_A[acc_names[0]][1]
vals_X = np.array(vals_X)

vals_Xd = []
vals_Yd = []
vals_Zd = []
vals_Kd = []
X__ndx = 0
for c__ndx in range(len_c__Y):
    cc_str = 'Node' + str(c__Y_nodes[c__ndx][0]) + 'Y'
    vals_Ymin = c__Y_nodes[c__ndx][1] - c__dist[c__ndx]
    vals_Ymax = c__Y_nodes[c__ndx][1] + c__dist[c__ndx]
    for J__ndx in range(len(vals_X)-1):
        H__ndx = X__ndx*2
        vals_Xd.append(vals_X[J__ndx])
        vals_Xd.append(vals_X[J__ndx])
        vals_Yd.append(vals_Ymin)
        vals_Yd.append(vals_Ymax)
        vals_Kd.append([H__ndx+0, H__ndx+1, H__ndx+2])
        vals_Kd.append([H__ndx+1, H__ndx+3, H__ndx+2])
        vals_Zd.append(mat_K_A[cc_str][J__ndx+1])
        vals_Zd.append(mat_K_A[cc_str][J__ndx+1])
        X__ndx += 1
    vals_Xd.append(vals_X[J__ndx+1])
    vals_Xd.append(vals_X[J__ndx+1])
    vals_Yd.append(vals_Ymin)
    vals_Yd.append(vals_Ymax)
    X__ndx += 1

xy_mtri_SA = mtri.Triangulation(vals_Yd, vals_Xd, triangles=vals_Kd)
tpc = plt_ax[0].tripcolor(xy_mtri_SA, vals_Zd, shading='flat', cmap=cmap, \
vmin=c_minVal, vmax=c_maxVal, norm=clr.LogNorm())

cax = figure.add_axes([0.87, 0.52, 0.04, 0.25])

c_bar = plt.colorbar(tpc, ax=plt_ax[0], cax=cax)
c_bar.ax.tick_params(width=ax_linewidth, labelsize=ax_fontsize*0.85)
c_bar.outline.set_linewidth(ax_linewidth)
c_bar.set_ticks(c_bar_ticks)
c_bar_ticklabels = map(lambda x: c_fmt_punct.format(x), c_bar_ticks)
c_bar.set_ticklabels(c_bar_ticklabels)

plt_ax[1].axhline(y_axin_max*mult_val, linestyle=':', dash_capstyle='round', \
dash_joinstyle='round', linewidth=ax_linewidth, color=[0.5, 0.5, 0.5])
plt_ax[0].axhline(min_ax_v, linestyle=':', dash_capstyle='round', \
dash_joinstyle='round', linewidth=ax_linewidth, color=[0.5, 0.5, 0.5])
plt_ax[0].grid(True, linestyle=':', dash_capstyle='round', \
dash_joinstyle='round', linewidth=ax_linewidth, color=[0.5, 0.5, 0.5])

out_fft = os_path.join(*[outfold, 'profile_FFT.svg'])
figure.savefig(out_fft, format='svg', bbox_inches='tight')
if save_png: figure.savefig((out_fft[:-3] + 'png').replace(outfold, png_outfold), format='png', dpi=600, bbox_inches='tight')

#~ ##############################

lprint(par_names[-2][1])
tpc.remove()
cax.remove()
#~ cax_ann.remove()
plt_ax[0].set_yscale('linear')

max_acc = 0
for c__ndx in range(len_c__Y):
    cc_str = 'Node' + str(c__Y_nodes[c__ndx][0]) + 'Y'
    max_acc = max(max_acc, max(abs(mat_acc_A[cc_str])))

scale_v = float(input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_maw"]])/max_acc

c_col = [0.50]*3
for c__ndx in range(len_c__Y):
    if np.allclose(c_col,[0.50]*3): c_col = [0.00]*3
    elif np.allclose(c_col,[0.00]*3): c_col = [0.75]*3
    elif np.allclose(c_col,[0.75]*3): c_col = [0.25]*3
    elif np.allclose(c_col,[0.25]*3): c_col = [0.50]*3
    cc_str = 'Node' + str(c__Y_nodes[c__ndx][0]) + 'Y'
    vals_Y = c__Y_nodes[c__ndx][1]
    plt_ax[0].plot(mat_acc_A[cc_str] * scale_v + vals_Y, mat_acc_A[acc_names[0]], linestyle='-', marker='', color=c_col, linewidth=c_linewidth)

plt_ax[0].yaxis.set_major_locator(AutoLocator())
plt_ax[0].set_ylim([.0, mat_acc_A[acc_names[0]][-1]])
plt_ax[0].set_ylabel(par_names[-2][2], fontsize=ax_fontsize)
plt_ax[0].set_title(par_names[-2][3], fontsize=ax_fontsize)
plt_ax[0].tick_params(labelsize=ax_fontsize*0.85)

figure.savefig(par_names[-2][0], format='svg', bbox_inches='tight')
if save_png: figure.savefig((par_names[-2][0][:-3] + 'png').replace(outfold, png_outfold), format='png', dpi=600, bbox_inches='tight')

#~ ##############################

hdr_str = ''
for hdr_i in c__nodes_numb: hdr_str += hdr_i + ';'
hdr_str = hdr_str[:-1] + '\r\n'

S_fid = open(S_filename,'w')
S_fid.write(hdr_str)
np.savetxt(S_fid, mat_SD_A, delimiter=';')
S_fid.write('\n')
S_fid.close()

#~ ##############################

F_fid = open(F_filename,'w')
F_fid.write(hdr_str.replace('Timesec', 'FreqHz'))
np.savetxt(F_fid, mat_F_A, delimiter=';')
F_fid.write('\n')
F_fid.close()

#~ ##############################

K_fid = open(K_filename,'w')
K_fid.write(hdr_str.replace('Timesec', 'FreqHz'))
np.savetxt(K_fid, mat_K_A, delimiter=';')
K_fid.write('\n')
K_fid.close()

#~ ##############################

c__X_nodes_surface = map(str, map(list, zip(*c__X_nodes))[0])
c__Y_nodes_surface = map(str, map(list, zip(*c__Y_nodes))[0])
c__X_nodes_surface = [s + 'X' for s in c__X_nodes_surface]
c__Y_nodes_surface = [s + 'Y' for s in c__Y_nodes_surface]

N_fid = open(N_filename,'w')
N_fid.write('{:10s}'.format('Node'))
N_fid.write('{:>14s}'.format('X-coord [m]'))
N_fid.write('{:>14s}'.format('Y-coord [m]'))
N_fid.write('{:>18s}'.format('on ground [Y|N]') + '\n')
for ix in range(len(c__X_nodes_numb)):
    N_fid.write('{:10s}'.format(c__X_nodes_numb[ix]))
    N_fid.write('{:14.2f}'.format(c__X_all[ix][0]))
    N_fid.write('{:14.2f}'.format(c__X_all[ix][1]))
    if c__X_nodes_numb[ix] in c__X_nodes_surface: str_S = 'Y'
    else: str_S = 'N'
    N_fid.write('{:>18s}'.format(str_S + '\n'))
for iy in range(len(c__Y_nodes_numb)):
    N_fid.write('{:10s}'.format(c__Y_nodes_numb[iy]))
    N_fid.write('{:14.2f}'.format(c__Y_all[iy][0]))
    N_fid.write('{:14.2f}'.format(c__Y_all[iy][1]))
    if c__Y_nodes_numb[iy] in c__Y_nodes_surface: str_S = 'Y'
    else: str_S = 'N'
    N_fid.write('{:>18s}'.format(str_S + '\n'))
N_fid.close()

#~ ##############################

#~ print '-----'
#~ print mat_elems_O['GMX']
#~ print '-----'
#~ print mat_pga_O['XACC']
#~ print np.max(mat_pga_O['XACC'])
#~ print '-----'
#~ print mat_strains_O['EPSMAX']
#~ print np.max(mat_strains_O['EPSMAX'])
#~ print '-----'
#~ print mat_acc_A['Timesec']
#~ print '-----'

#~ ##############################

sys_exit(0)

             #~ 1   -100.0       .0         .0469       10.5400         .0323        9.9200
          #~ ELM           SIG-X          SIG-Y         SIG-XY         EPS-MAX        AT TIME


             #~ 1        14142.6       115217.3       134850.0           .004         12.960
             #~ 2        10983.0         2110.6       157171.4           .004         12.940



#~ In [2]: mat_elems_O
#~ Out[2]: 
#~ array([ (   1,    1,   68,   69,    2, 1,  22000.,  0.392,  4075846.,  3668261.,  0.005,  27.52),
       #~ (   2,    2,   69,   70,    3, 1,  22000.,  0.392,  4047139.,  3642426.,  0.005,  27.95),
       #~ (   3,    3,   70,   71,    4, 1,  22000.,  0.392,  4065564.,  3659008.,  0.005,  27.52),
       #~ ...,
       #~ (8831, 8875, 8934, 8935, 8876, 1,  22000.,  0.392,  4097007.,  3687306.,  0.005,   4.62),
       #~ (8832, 8877, 8876, 8935, 8935, 1,  22000.,  0.392,  4064403.,  3657963.,  0.005,   2.31),
       #~ (8833, 8935, 8936, 8877, 8877, 1,  22000.,  0.392,  4094342.,  3684908.,  0.005,   2.31)], 
      #~ dtype=[('f0', '<i8'), ('f1', '<i8'), ('f2', '<i8'), ('f3', '<i8'), ('f4', '<i8'), ('f5', '<i8'), ('f6', '<f8'), ('f7', '<f8'), ('f8', '<f8'), ('f9', '<f8'), ('f10', '<f8'), ('f11', '<f8')])

#~ In [3]: mat_elems_O['f0']
#~ Out[3]: array([   1,    2,    3, ..., 8831, 8832, 8833])




    
    
    
   #~ ELM     NODE 1  NODE 2  NODE 3  NODE 4  MAT.TYPE  DENSITY    POISSON R.     GMX    SH. MODULUS  DAMP. RATIO     AREA
                                                     #~ (N/M^3)                (KN/M^2)    (KN/M^2)                   (M^2)

       #~ 1       1      68      69       2       1   22000.000        .392 4075846.000 3668261.000        .005      27.520
