from matplotlib import colors as clr
import numpy as np
from numpy.fft import fft as np_fft
import math
import copy
from shapely.geometry import Point, MultiPoint, LineString, LinearRing, Polygon
from shapely.ops import nearest_points
import matplotlib.path as mpltPath
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from os import path as os_path
import argparse


def common_def():
    common_def_dict = {}
    common_def_dict["version"] = "pro-QUAD4M 0.14.3"
    common_def_dict["json"] = "JSON-file containing model parameters"
    common_def_dict["out_folder"] = 'directory where outputs are saved (default: "var"); herein a directory will be created if not existent, whose name is specified in the JSON-file (see "json" positional argument above-mentioned) within the field "job_folder" of "modelling_parameters" section'
    common_def_dict["version_h"] = "print the pro-QUAD4M version number and exit"
    return common_def_dict


def is_valid_file(arg):
    if not os_path.exists(arg):
        raise argparse.ArgumentTypeError("File '%s' does not exist!" % arg)
    else: return arg


def is_valid_dir(arg):
    if not os_path.isdir(arg):
        raise argparse.ArgumentTypeError("Folder '%s' does not exist! Please create it before proceeding or furnish a valid folder using the specific command-line argument." % arg)
    else: return arg


def lprint(c_string, return_str=False):
    half_len = 25
    len_c_string = len(c_string)
    half_len_c_string = len_c_string / 2
    half_len_t = half_len - (half_len_c_string + 1)
    t_string = '=' * half_len_t
    if half_len_c_string >= half_len:
        o_string = c_string
    elif not len_c_string%2:
        o_string = t_string + " " + c_string + " =" + t_string
    else:
        o_string = t_string + " " + c_string + " " + t_string
    if return_str:
        return o_string
    else:
        print o_string
        return None


def strata_plot(plt_ax, c_st, c_edge_color, c_face_color, c_label, c_alpha, c_linewidth):
    c_x = zip(*c_st)[0]
    c_y = zip(*c_st)[1]
    plt_ax.fill(c_x, c_y, joinstyle='round',\
    edgecolor=clr.to_rgba(c_edge_color, alpha=1.0), \
    facecolor=clr.to_rgba(c_face_color, alpha=c_alpha), \
    linewidth=c_linewidth, label=c_label)
    return plt_ax


def add_tolerance(strata_s, tolerance_s, min_s, max_s):
    for c_strata_s in strata_s:
        cn_strata_s = c_strata_s[1] - tolerance_s
        if cn_strata_s >= min_s: c_strata_s[1] = cn_strata_s
        else: c_strata_s[1] = min_s
        cm_strata_s = c_strata_s[2] + tolerance_s
        if cm_strata_s <= max_s: c_strata_s[2] = cm_strata_s
        else: c_strata_s[2] = max_s
    return strata_s


def set_unions(a, b):
    #~ a = [[6.4, 16]]
    #~ b = [[1, 6.3], [7.4, 12], [15, 18]]
    #~ ranges = [[1, 6.3], [6.4, 18]]
    for a_range in a: b.append(a_range)
    ranges = []
    a_ranges = []
    for begin, end in sorted(b):
        if a_ranges and a_ranges[-1][1] >= begin:
            a_ranges[-1][1] = max(a_ranges[-1][1], end)
        else:
            a_ranges.append([begin, end])
    for c_range in a_ranges:
        if not c_range[0] == c_range[1]: ranges.append(c_range)
    return ranges


def set_intersections(a, b):
    #~ a = [[6.4, 16]]
    #~ b = [[1, 6.3], [7.4, 12], [15, 18]]
    #~ ranges = [[7.4, 12], [15, 16]]
    ranges = []
    a_ranges = []
    i = j = 0
    while i < len(a) and j < len(b):
        a_left, a_right = a[i]
        b_left, b_right = b[j]

        if a_right < b_right: i += 1
        else: j += 1

        if a_right >= b_left and b_right >= a_left:
            end_pts = sorted([a_left, a_right, b_left, b_right])
            middle = [end_pts[1], end_pts[2]]
            a_ranges.append(middle)
    ri = 0
    while ri < len(a_ranges)-1:
        if a_ranges[ri][1] == a_ranges[ri+1][0]:
            a_ranges[ri:ri+2] = [[a_ranges[ri][0], a_ranges[ri+1][1]]]
        ri += 1
    for c_range in a_ranges:
        if not c_range[0] == c_range[1]: ranges.append(c_range)
    return ranges


def find_nodes(c_dim, c_ranges, min_dim):
    c_lsps = []
    t_dim = min([c_dim, min_dim])
    for c_range in c_ranges:
        c_num = math.ceil(float(c_range[1] - c_range[0])/float(t_dim))
        if c_num < 1.0: c_num = 1.0
        c_lsp = np.linspace(start=c_range[0], stop=c_range[1], \
        num=int(c_num + 1), endpoint=True, retstep=False)
        c_lsps += np.around(c_lsp,2).tolist()
    return c_lsps


def find_intervals(strata_s, min_val, max_val, strata, min_dim):
    
    strata_st = []
    for c_ndx in range(strata):

        if not c_ndx: c_poi_rep_for = [[0, 0]]
        else: c_poi_rep_for = strata_st[c_ndx-1][2]

        a_arr_int = np.reshape(c_poi_rep_for, (1, len(c_poi_rep_for) * 2))[0].tolist()
        a_arr = [min_val] + a_arr_int + [max_val]
        c_poi_rep_pre = np.reshape(a_arr, (len(a_arr) / 2, 2)).tolist()
        c_poi_rep = [[strata_s[c_ndx][1], strata_s[c_ndx][2]]]
        c_poi_inter = set_intersections(c_poi_rep_pre, c_poi_rep)
        c_poi_unions = set_unions(c_poi_rep_for, c_poi_rep)
        c_dim = math.floor(strata_s[c_ndx][0]*100)/100
        c_lsps = find_nodes(c_dim, c_poi_inter, min_dim)
        #.tolist()
        strata_st.append([c_dim, c_poi_inter, c_poi_unions, strata_s[c_ndx][3], c_lsps])

    return strata_st


def all_intervals(c_st):
    c_lines = []
    for c_st_val in c_st: c_lines += c_st_val[4]
    c_lines = sorted(set(c_lines))
    return c_lines


def adjust_intervals(c_y_lines, c_pois_def, def_coe, dist_threshold, e_exp, i_exp):

    l_y = len(c_y_lines)
    l_y1 = l_y - 1
    d_diff = list(np.zeros(l_y))
    
    # adjustment of mesh borders
    s_diff = list(np.ones(l_y))
    for s_ndx in range(int(len(s_diff)/2)):
        g_ndx = -(s_ndx + 1)
        # extremities exponent EXP = e_exp
        s_diff[s_ndx] = ((c_y_lines[s_ndx] - c_y_lines[0]) / dist_threshold) ** e_exp
        s_diff[g_ndx] = ((c_y_lines[-1] - c_y_lines[g_ndx]) / dist_threshold) ** e_exp
    # set values greater than 1.0 to 1.0
    s_diff_np = np.array(s_diff)
    s_threshold_ndx = s_diff_np > 1.0
    s_diff_np[s_threshold_ndx] = 1.0
    s_diff = list(s_diff_np)    

    for k_poi in c_pois_def: 
        
        min_ndx = np.argmin(map(lambda x:abs(x-k_poi), c_y_lines))

        k_1 = c_y_lines[min_ndx]

        b_diff = list(np.zeros(l_y))
        c_diff = list(np.zeros(l_y))
        
        diff_k_poi = (k_poi - k_1) * def_coe

        #~ direction == 'down'/'left':
        for y_ndx in range(min_ndx+1):
            c_diff[y_ndx] = \
            1. - (k_1 - c_y_lines[y_ndx])/dist_threshold
        # set values lower than 0.0 to 0.0
        c_diff_np = np.array(c_diff)
        c_threshold_ndx = c_diff_np < 0.0
        c_diff_np[c_threshold_ndx] = 0.0
        c_diff = list(c_diff_np)
        
        #~ direction == 'up'/'right':
        for y_ndx in range(l_y1 - min_ndx - 1):
            b_diff[l_y1 - (y_ndx + 1)] = \
            1. - (c_y_lines[l_y1 - (y_ndx + 1)] - k_1)/dist_threshold
        # set values lower than 0.0 to 0.0
        b_diff_np = np.array(b_diff)
        b_threshold_ndx = b_diff_np < 0.0
        b_diff_np[b_threshold_ndx] = 0.0
        b_diff = list(b_diff_np)
        
        # generate d_diff
        # inline exponent EXP = i_exp
        a_diff = map(lambda x, y: diff_k_poi * ((x)**i_exp + (y)**i_exp), c_diff, b_diff)
        d_diff = map(lambda x, y: x + y, a_diff, d_diff)

    d_diff = map(lambda x, y: x * y, d_diff, s_diff)
    c_y_lines = map(lambda x, y: x + y, c_y_lines, d_diff)
    
    return c_y_lines, d_diff


def adjust_line(h_dist, h_dist_threshold, h_arr_line, h_diff, o_exp):
    if h_dist < h_dist_threshold:
        d_val = h_dist/h_dist_threshold
        # orthogonal exponent EXP = o_exp
        d_mp_val_ts = (1.0 - d_val) ** o_exp
        h_arr_line = map(lambda x, y: x + y * d_mp_val_ts, h_arr_line, h_diff)
    return h_arr_line


def adjust_all_intervals(w_arr_lines, c_dir, v_arr_lines, \
input_dict, strata, json_elem_dict, dist_threshold, exp_print, s_def_coe):
    
    if c_dir == 'X': c_dir_ndx = 1
    elif c_dir == 'Y': c_dir_ndx = 0
    
    # exponents
    over_exp = input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_ran"]]\
    [json_elem_dict["mesh_ran_ov"]]
    if over_exp:
        inli_exp = extr_exp = orth_exp = float(over_exp)
    else:
        inli_exp = float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_ran"]]\
        [json_elem_dict["mesh_ran_ex"]][json_elem_dict["mesh_ran_ex_i"]])
        extr_exp = float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_ran"]]\
        [json_elem_dict["mesh_ran_ex"]][json_elem_dict["mesh_ran_ex_e"]])
        orth_exp = float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_ran"]]\
        [json_elem_dict["mesh_ran_ex"]][json_elem_dict["mesh_ran_ex_o"]])
    
    if exp_print:
        lprint("inline exponent: " + str(inli_exp))
        lprint("(in)line-extremities exponent: " + str(extr_exp))
        lprint("orthogonal exponent: " + str(orth_exp))
        exp_print = False

    wv_arr_lines = generate_composite_ab(c_dir, w_arr_lines, v_arr_lines)
    for v_ndx in range(len(w_arr_lines)):

        
        w_poi = retrieve_poi_s(c_dir, wv_arr_lines[v_ndx], input_dict, strata, json_elem_dict)
        w_poi = sorted(set(w_poi))
        
        w_arr_lines[v_ndx], w_diff = \
        adjust_intervals(w_arr_lines[v_ndx], w_poi, s_def_coe, dist_threshold, extr_exp, inli_exp)
        
        # propagate adjustments to nearby lines
        if np.max(np.abs(w_diff)):

            #~ for j_ndx in range(1,8): # [1, 2, 3, 4, 5, 6, 7]
                #~ mp_val = (2.0/3.0) / (2.0**float(j_ndx-1))
                #~ d_ndx = v_ndx-j_ndx
                #~ if d_ndx in range(len(w_arr_lines)):
                    #~ w_arr_lines[d_ndx] = \
                    #~ map(lambda x, y: x + y * mp_val, w_arr_lines[d_ndx], w_diff)
                #~ e_ndx = v_ndx+j_ndx
                #~ if e_ndx in range(len(w_arr_lines)):
                    #~ w_arr_lines[e_ndx] = \
                    #~ map(lambda x, y: x + y * mp_val, w_arr_lines[e_ndx], w_diff)

            #~ od_dist_threshold = dist_threshold #/ 2.0
            v_v_mean = np.mean(zip(*wv_arr_lines[v_ndx])[c_dir_ndx])
            for j_ndx in range(1,len(w_arr_lines)/5): # [1, 2, 3, 4, ...]
                d_ndx = v_ndx - j_ndx
                if d_ndx in range(len(w_arr_lines)):
                    d_c_dist = abs(v_v_mean - np.mean(zip(*wv_arr_lines[d_ndx])[c_dir_ndx]))
                    w_arr_lines[d_ndx] = \
                    adjust_line(d_c_dist, dist_threshold, w_arr_lines[d_ndx], w_diff, orth_exp)
                e_ndx = v_ndx + j_ndx
                if e_ndx in range(len(w_arr_lines)):
                    e_c_dist = abs(np.mean(zip(*wv_arr_lines[e_ndx])[c_dir_ndx]) - v_v_mean)
                    w_arr_lines[e_ndx] = \
                    adjust_line(e_c_dist, dist_threshold, w_arr_lines[e_ndx], w_diff, orth_exp)
    return w_arr_lines, exp_print


def generate_composite_ab(c_dir, a_arr_lines, b_arr_lines):
    ab_arr_lines = copy.deepcopy(a_arr_lines)
    for a_ndx in range(len(a_arr_lines)):
        for b_ndx in range(len(b_arr_lines)):
            if c_dir == 'X':
                ab_arr_lines[a_ndx][b_ndx] = \
                [a_arr_lines[a_ndx][b_ndx], b_arr_lines[b_ndx][a_ndx]] 
            elif c_dir == 'Y':
                ab_arr_lines[a_ndx][b_ndx] = \
                [b_arr_lines[b_ndx][a_ndx], a_arr_lines[a_ndx][b_ndx]]
            else: return None
    return ab_arr_lines


def retrieve_poi_s(c_dir, ab_arr_line, input_dict, strata, json_elem_dict):

    c_ls = LineString(ab_arr_line)
    c_pois = []
    nodal_points = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_nps"]]
    for stratum_ndx in map(lambda x:x+1, range(strata)):
        c_stratum = json_elem_dict["prem"] + "%03d" % stratum_ndx
        c_sty = input_dict[c_stratum][json_elem_dict["prem_nod"]]
        c_st = [[float(nodal_points[str(a_st)][0]), float(nodal_points[str(a_st)][1])] for a_st in c_sty]
        c_ls_st = LinearRing(c_st)
        c_ls_inter = c_ls.intersection(c_ls_st)
        if c_ls_inter.type == 'MultiLineString':
            c_pois.append(eval("c_ls_inter.boundary[0]." + c_dir.lower()))
            c_pois.append(eval("c_ls_inter.boundary[1]." + c_dir.lower()))
        elif c_ls_inter.type == 'Point':
            c_pois.append(eval("c_ls_inter." + c_dir.lower()))
        else:
            for k_ndx in range(len(c_ls_inter)):
                if c_ls_inter[k_ndx].type == 'Point':
                    c_pois.append(eval("c_ls_inter[k_ndx]." + c_dir.lower()))
                elif c_ls_inter[k_ndx].type == 'LineString':
                    c_pois.append(eval("c_ls_inter[k_ndx].boundary[0]." + c_dir.lower()))
                    c_pois.append(eval("c_ls_inter[k_ndx].boundary[1]." + c_dir.lower()))
    return c_pois


def retrieve_pois_s(c_dir, a_arr_lines, b_arr_lines, input_dict, strata, json_elem_dict):

    ab_arr_lines = generate_composite_ab(c_dir, a_arr_lines, b_arr_lines)
    c_pois_def = []
    for ab_arr_line in ab_arr_lines:
        c_pois = retrieve_poi_s(c_dir, ab_arr_line, input_dict, strata, json_elem_dict)
        c_pois_def.append(sorted(set(c_pois)))
    return c_pois_def


def retrieve_pois_ndx(a_arr_lines, a_poi_def):
    a_poi_ndxs = []
    for b_ndx in range(len(a_arr_lines)):
        c_pois_def = a_poi_def[b_ndx]
        b_poi_ndxs = []
        for b_poi in c_pois_def:
            b_poi_ndx = np.argmin(map(lambda x: abs(x-b_poi), a_arr_lines[b_ndx]))
            b_poi_ndxs.append(b_poi_ndx)
        a_poi_ndxs.append(b_poi_ndxs)
    return a_poi_ndxs


def val_uncentanty(c_dict, c_val_lab, c_min_val_lab, c_unc_lab, c_log_lab):
    #~ c_dict = {u'min_value': None, u'uncentanty': 0, u'value': 19600, u'log_normal_tf': False}
    #~ c_val_lab = 'value'
    #~ c_min_val_lab = 'min_value'
    #~ c_unc_lab = 'uncentanty'
    #~ c_log_lab = 'log_normal_tf'
    
    if not c_dict[c_unc_lab] or (c_dict[c_unc_lab] == 1.0 and c_dict[c_log_lab]):
        # 'uncentanty' = 0 or null
        # or
        # 'uncentanty' = 1 and 'log_normal_tf' = true (log-normal distribution with no uncertanty)
        c_val = float(c_dict[c_val_lab])        
    else:
        if c_dict[c_log_lab]:
            # lognormal distribution
            c_val = np.random.lognormal(mean=np.log(float(c_dict[c_val_lab])), sigma=np.log(float(c_dict[c_unc_lab])))
        else:
            # normal distribution
            c_val = np.random.normal(loc=float(c_dict[c_val_lab]), scale=float(c_dict[c_unc_lab]))

        # min_value threshold
        if c_dict[c_min_val_lab] is not None:
            # 'min_value' != null
            # --> truncate distribution  (value < min_value --> value = min_value)
            if c_val < float(c_dict[c_min_val_lab]): c_val = float(c_dict[c_min_val_lab])

    return c_val

def add_element(c_node_a, c_node_b, c_node_c, c_node_d, strata, json_elem_dict, input_dict, \
element_lines_array, element_lines_plot, element_lines_write, elem_type, ccw_continue, stratum_fx):

    c_x_max = max([c_node_a[1], c_node_b[1], c_node_c[1], c_node_d[1]])
    c_x_min = min([c_node_a[1], c_node_b[1], c_node_c[1], c_node_d[1]])
    c_y_max = max([c_node_a[2], c_node_b[2], c_node_c[2], c_node_d[2]])
    c_y_min = min([c_node_a[2], c_node_b[2], c_node_c[2], c_node_d[2]])
    
    c_ins = False
    if stratum_fx == None: # 'None' (find soil-identifier) or soil: '0' (no soil) or '1', '2', etc.

        if elem_type == 'quadrilateral':
            c_ele_st = LinearRing([c_node_a[1:], c_node_b[1:], c_node_c[1:], c_node_d[1:]])
            c_center_0 = [c_x_min + ((c_x_max-c_x_min) / float(2.0)), \
            c_y_min + ((c_y_max-c_y_min) / float(2.0))]
        elif elem_type == 'triangle':
            c_ele_st = LinearRing([c_node_a[1:], c_node_b[1:], c_node_c[1:]])
            c_center_0 = [float(c_node_a[1] + c_node_b[1] + c_node_c[1]) / float(3.0), \
            float(c_node_a[2] + c_node_b[2] + c_node_c[2]) / float(3.0)]
        
        if not c_ele_st.is_ccw:
            ccw_continue = False
            print "WARNING: element " + str(c_node_a[0]) + "-" + str(c_node_b[0]) + "-" + str(c_node_c[0]) + "-" + str(c_node_d[0]) + " not in counter-clockwise order"
        
        c_radius = min([(c_x_max-c_x_min)/50.0, (c_y_max-c_y_min)/50.0])
        c_center_1 = [c_center_0[0] - c_radius, c_center_0[1] - c_radius]
        c_center_2 = [c_center_0[0] - c_radius, c_center_0[1] + c_radius]

        nodal_points = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_nps"]]
        for stratum_ndx in map(lambda x:x+1, range(strata)):
            c_stratum = json_elem_dict["prem"] + "%03d" % stratum_ndx
            c_sty = input_dict[c_stratum][json_elem_dict["prem_nod"]]
            c_st = [[float(nodal_points[str(a_st)][0]), float(nodal_points[str(a_st)][1])] for a_st in c_sty]
            c_st_path = mpltPath.Path(c_st)
            c_inside = c_st_path.contains_points([c_center_0])
            if c_inside[0] == True: c_ins = True; break
            c_inside = c_st_path.contains_points([c_center_1])
            if c_inside[0] == True: c_ins = True; break
            c_inside = c_st_path.contains_points([c_center_2])
            if c_inside[0] == True: c_ins = True; break
    else:
        stratum_ndx = stratum_fx
        c_stratum = json_elem_dict["prem"] + "%03d" % stratum_ndx

    if stratum_fx > 0 or c_ins == True:

        element_lines = [[\
        [c_node_a[1], c_node_a[2]], \
        [c_node_b[1], c_node_b[2]], \
        [c_node_c[1], c_node_c[2]], \
        [c_node_d[1], c_node_d[2]]], \
        stratum_ndx, elem_type[0]]

        element_lines_H = [[\
        [c_node_a[0], c_node_a[1], c_node_a[2]], \
        [c_node_b[0], c_node_b[1], c_node_b[2]], \
        [c_node_c[0], c_node_c[1], c_node_c[2]], \
        [c_node_d[0], c_node_d[1], c_node_d[2]]], \
        stratum_ndx, elem_type[0]]

        element_lines_array.append([c_node_a[0], c_node_b[0], c_node_c[0], c_node_d[0], \
        stratum_ndx, elem_type[0]])
        element_lines_plot.append(element_lines)

        c_val_DENS = val_uncentanty(input_dict[c_stratum][json_elem_dict["prem_dns"]], \
        json_elem_dict["etc_va"], json_elem_dict["etc_mv"], json_elem_dict["etc_uc"], json_elem_dict["etc_ln"])
        c_val___XL = val_uncentanty(input_dict[c_stratum][json_elem_dict["prem_xld"]], \
        json_elem_dict["etc_va"], json_elem_dict["etc_mv"], json_elem_dict["etc_uc"], json_elem_dict["etc_ln"])
        c_val__GMX = val_uncentanty(input_dict[c_stratum][json_elem_dict["prem_gmx"]], \
        json_elem_dict["etc_va"], json_elem_dict["etc_mv"], json_elem_dict["etc_uc"], json_elem_dict["etc_ln"])
        c_val___PO = val_uncentanty(input_dict[c_stratum][json_elem_dict["prem_por"]], \
        json_elem_dict["etc_va"], json_elem_dict["etc_mv"], json_elem_dict["etc_uc"], json_elem_dict["etc_ln"])
        G_start_percent = float(input_dict[c_stratum][json_elem_dict["prem_gmp"]]\
        [json_elem_dict["prem_gmp_pc"]]) / 100.0

        element_lines_write.append([c_node_a[0], c_node_b[0], c_node_c[0], c_node_d[0], \
        stratum_ndx, c_val_DENS, c_val___PO, c_val__GMX, c_val__GMX * G_start_percent, c_val___XL])

    else:

        element_lines = [[\
        [c_node_a[1], c_node_a[2]], \
        [c_node_b[1], c_node_b[2]], \
        [c_node_c[1], c_node_c[2]], \
        [c_node_d[1], c_node_d[2]]], \
        0, elem_type[0]]

        element_lines_H = [[\
        [c_node_a[0], c_node_a[1], c_node_a[2]], \
        [c_node_b[0], c_node_b[1], c_node_b[2]], \
        [c_node_c[0], c_node_c[1], c_node_c[2]], \
        [c_node_d[0], c_node_d[1], c_node_d[2]]], \
        0, elem_type[0]]
        #~ print "WARNING: element " + str(a_ndx) + "/" + str(b_ndx) + " not associated with geotechnical properties"

    return element_lines, element_lines_H, element_lines_array, element_lines_plot, element_lines_write, ccw_continue


def add_element_triang(xy_tri, xy_tri_ndxs, z_tri, c_ele, c_ele_w):
    xy_tri_len = len(xy_tri)
    if c_ele[-2]:
        if c_ele[-1] == 't':
            xy_tri.append(c_ele[0][0])
            xy_tri.append(c_ele[0][1])
            xy_tri.append(c_ele[0][2])
            xy_tri_ndxs.append([xy_tri_len+0, xy_tri_len+1, xy_tri_len+2])
            z_tri.append(c_ele_w[5:])
        if c_ele[-1] == 'q':
            xy_tri.append(c_ele[0][0])
            xy_tri.append(c_ele[0][1])
            xy_tri.append(c_ele[0][2])
            xy_tri.append(c_ele[0][3])
            xy_tri_ndxs.append([xy_tri_len+0, xy_tri_len+1, xy_tri_len+3])
            xy_tri_ndxs.append([xy_tri_len+1, xy_tri_len+2, xy_tri_len+3])
            z_tri.append(c_ele_w[5:])
            z_tri.append(c_ele_w[5:])
    return xy_tri, xy_tri_ndxs, z_tri


def find_np(modp_c_H, c_MP):
    c_def_list = []
    if modp_c_H:
        for c_point_list in modp_c_H:
            c__P = Point(c_point_list)
            c_np = nearest_points(c_MP, c__P)[0]
            c_def_list.append([c_np.x, c_np.y])
    return c_def_list


def plot_border_inner(plt_ax, c_ele_inner, o_ndx, p_ndx, i_c_s, \
c_ol=[1.0, 1.0, 1.0], l_in=0.30):
    c_xs = [c_ele_inner[0][o_ndx][0], c_ele_inner[0][p_ndx][0]]
    c_ys = [c_ele_inner[0][o_ndx][1], c_ele_inner[0][p_ndx][1]]
    plt_ax.plot(c_xs, c_ys, color=c_ol, linewidth=l_in, \
    solid_capstyle='round')
    i_c_s.append([[c_xs[0], c_ys[0]], [c_xs[1], c_ys[1]]])
    return plt_ax, i_c_s


def plot_border(plt_ax, c_ele, c_elf, c_dir, c_s, l_in):

    if c_ele[0] == 'q':

        if c_dir == 'X':
            if (c_elf[0] == 'q' and not c_ele[1][1] == c_elf[1][1]) \
            or (c_elf[0] == 'tdx' and not c_ele[1][1] == c_elf[2][1]) \
            or (c_elf[0] == 'tsx' and not c_ele[1][1] == c_elf[1][1]):
                plt_ax, c_s = plot_border_inner(plt_ax, c_ele[1], 1, 2, c_s, l_in=l_in)

        elif c_dir == 'Y':
            if not c_ele[1][1] == c_elf[1][1]:
                plt_ax, c_s = plot_border_inner(plt_ax, c_ele[1], 2, 3, c_s, l_in=l_in)

    elif c_ele[0] == 'tdx' or c_ele[0] == 'tsx':

        if c_dir == 'X':
            if c_ele[0] == 'tdx':
                if (c_elf[0] == 'q' and not c_ele[1][1] == c_elf[1][1]) \
                or (c_elf[0] == 'tdx' and not c_ele[1][1] == c_elf[2][1]) \
                or (c_elf[0] == 'tsx' and not c_ele[1][1] == c_elf[1][1]):
                    plt_ax, c_s = plot_border_inner(plt_ax, c_ele[1], 1, 2, c_s, l_in=l_in)
            elif c_ele[0] == 'tsx':
                if (c_elf[0] == 'q' and not c_ele[2][1] == c_elf[1][1]) \
                or (c_elf[0] == 'tdx' and not c_ele[2][1] == c_elf[2][1]) \
                or (c_elf[0] == 'tsx' and not c_ele[2][1] == c_elf[1][1]):
                    plt_ax, c_s = plot_border_inner(plt_ax, c_ele[2], 0, 1, c_s, l_in=l_in)

        elif c_dir == 'Y':
            if not c_ele[1][1] == c_ele[2][1]:
                plt_ax, c_s = plot_border_inner(plt_ax, c_ele[1], 0, 3, c_s, l_in=l_in)
            if not c_ele[2][1] == c_elf[1][1]:
                plt_ax, c_s = plot_border_inner(plt_ax, c_ele[2], 1, 2, c_s, l_in=l_in)

    return plt_ax, c_s


def add_markers(c_plt, no_array, marker_symbol, legend_description, c_color, \
c_markersize):
    c_leg = None
    if no_array:
        #~ add_legend = True
        #~ for c_no in no_array:
            #~ if add_legend:
                #~ c_label = legend_description
                #~ add_legend = False
            #~ else: c_label = None
        no_array_t = map(list, zip(*no_array))
        c_leg, = c_plt.plot(no_array_t[1], no_array_t[2], markerfacecolor=c_color, \
        markeredgecolor=[0.0, 0.0, 0.0], markeredgewidth=c_markersize/10.0, \
        linestyle='None', marker=marker_symbol, markersize=c_markersize, \
        label=legend_description)
    return c_plt, c_leg


def pre_plot(plt, input_dict, json_elem_dict, ax_fontsize, c_linewidth):
    plt.clf()
    plt.ylabel('[m]', fontsize=ax_fontsize)
    plt.xlabel('[m]', fontsize=ax_fontsize)
    add_legend_dict = {}
    plt_ax = plt.gca()
    strata = int(input_dict[json_elem_dict["modp"]][json_elem_dict["modp_str"]])
    nodal_points = input_dict[json_elem_dict["modp"]][json_elem_dict["modp_nps"]]
    for stratum_ndx in map(lambda x:x+1, range(strata)):
        c_stratum = json_elem_dict["prem"] + "%03d" % stratum_ndx
        c_rgb = [0.0, 0.0, 0.0]
        c_sty = input_dict[c_stratum][json_elem_dict["prem_nod"]]
        c_st = [[float(nodal_points[str(a_st)][0]), float(nodal_points[str(a_st)][1])] for a_st in c_sty]
        plt_ax = strata_plot(plt_ax, c_st, c_rgb, [1.0, 1.0, 1.0], None, 0.0, c_linewidth)
        exec("add_legend_dict['add_legend_st" + "%03d" % stratum_ndx + "'] = True")
    return plt, plt_ax, add_legend_dict


def finalize_plot(plt, plt_ax, ax_linewidth, ax_fontsize, box='box', plt_box=False):
    plt_ax.set_aspect('equal', box)
    plt.grid(True, linestyle=':', dash_capstyle='round', \
    dash_joinstyle='round', linewidth=ax_linewidth, color=[0.5, 0.5, 0.5])
    plt.box(plt_box)
    plt_ax.tick_params(width=ax_linewidth, labelsize=ax_fontsize)
    return plt, plt_ax


def plot_all_borders(plt_ax, element_lines, element_lines_t, c_s, l_in=0.3):
    
    for g_ndx in range(len(element_lines)):
        for h_ndx in range(len(element_lines[0])-1):
            c_ele = element_lines[g_ndx][h_ndx]
            c_elf = element_lines[g_ndx][h_ndx+1]
            plt_ax, c_s = plot_border(plt_ax, c_ele, c_elf, 'Y', c_s, l_in)

    for i_ndx in range(len(element_lines_t)):
        for l_ndx in range(len(element_lines_t[0])-1):
            c_ele = element_lines_t[i_ndx][l_ndx]
            c_elf = element_lines_t[i_ndx][l_ndx+1]
            plt_ax, c_s = plot_border(plt_ax, c_ele, c_elf, 'X', c_s, l_in)

    return plt_ax, c_s


def post_plot(plt, plt_ax, no_bc_array_1, no_bc_array_2, no_bc_array_3, no_bc_array_4, \
no_out_array_1, no_out_array_2, no_out_array_3, leg_fontsize, c_l_width):

    c_col = [1.0, 0.0, 0.0]
    plt, bc_leg_1 = add_markers(c_plt=plt, no_array=no_bc_array_1, marker_symbol=">", \
    legend_description="horizontal motion applied, free in vertical direction", c_color=c_col, c_markersize=c_l_width)
    plt, bc_leg_2 = add_markers(c_plt=plt, no_array=no_bc_array_2, marker_symbol="^", \
    legend_description="vertical motion applied, free in horizontal direction", c_color=c_col, c_markersize=c_l_width)
    plt, bc_leg_3 = add_markers(c_plt=plt, no_array=no_bc_array_3, marker_symbol="P", \
    legend_description="horizontal and vertical motion applied", c_color=c_col, c_markersize=c_l_width)
    plt, bc_leg_4 = add_markers(c_plt=plt, no_array=no_bc_array_4, marker_symbol="s", \
    legend_description="transmitting base node", c_color=c_col, c_markersize=c_l_width)

    c_col = [0.0, 1.0, 0.0]
    plt, out_leg_1 = add_markers(c_plt=plt, no_array=no_out_array_1, marker_symbol=">", \
    legend_description="horizontal output", c_color=c_col, c_markersize=c_l_width)
    plt, out_leg_2 = add_markers(c_plt=plt, no_array=no_out_array_2, marker_symbol="^", \
    legend_description="vertical output", c_color=c_col, c_markersize=c_l_width)
    plt, out_leg_3 = add_markers(c_plt=plt, no_array=no_out_array_3, marker_symbol="P", \
    legend_description="horizontal and vertical output", c_color=c_col, c_markersize=c_l_width)

    all_leg_2 = [bc_leg_1, bc_leg_2, bc_leg_3, bc_leg_4, out_leg_1, out_leg_2, out_leg_3]
    all_leg_2 = filter(None, all_leg_2)
    l2 = plt_ax.legend(handles=all_leg_2, bbox_to_anchor=(0., -0.40, 1., .11), loc=2, \
    mode="expand", borderaxespad=0., frameon=False, fontsize=leg_fontsize, markerscale=1.6)

    return plt, plt_ax


#~ def par_plot(plt, plt_ax, input_dict, json_elem_dict, lprint_str, xy_mtri, \
#~ xy_triang, plt_title, c_minVal, c_maxVal, c_bar_ticks, c_fmt_punct, cmap, \
#~ logn, c_linewidth, c_leg_fontsize, ax_fontsize, element_lines, element_lines_t):
    #~ # #############################
    #~ lprint(lprint_str)
    #~ plt, plt_ax, add_legend_dict = pre_plot(plt, input_dict, json_elem_dict, ax_fontsize, c_linewidth)
    #~ # #############################
    #~ if logn:
        #~ tpc = plt_ax.tripcolor(xy_mtri, xy_triang, shading='flat', cmap=cmap, \
        #~ vmin=c_minVal, vmax=c_maxVal, norm=clr.LogNorm())
    #~ else:
        #~ tpc = plt_ax.tripcolor(xy_mtri, xy_triang, shading='flat', cmap=cmap, \
        #~ vmin=c_minVal, vmax=c_maxVal)        
    #~ divider = make_axes_locatable(plt_ax)
    #~ # #############################
    #~ plt.title(plt_title, fontsize=ax_fontsize)
    #~ # #############################
    #~ cax = divider.append_axes("right", size="5%", pad=0.05)
#    c_fmt = FuncFormatter(lambda x, p: c_fmt_punct.format(x))
    #~ c_bar = plt.colorbar(tpc, ax=plt_ax, cax=cax)#, format=c_fmt)
    #~ c_bar.ax.tick_params(labelsize=c_leg_fontsize)
    #~ c_bar.set_ticks(c_bar_ticks)
    #~ c_bar_ticklabels = map(lambda x: c_fmt_punct.format(x), c_bar_ticks)
    #~ c_bar.set_ticklabels(c_bar_ticklabels)
#    c_ticks = [float(t.get_text()) for t in c_bar.ax.get_ymajorticklabels()]
#    if len(c_ticks) < 3:
#        c_ticklabels = [t.get_text() for t in c_bar.ax.get_ymajorticklabels()]
#        c_bar.set_ticks([c_minVal, c_maxVal] + c_ticks)
#        c_bar.set_ticklabels([c_fmt_punct.format(c_minVal), \
#        c_fmt_punct.format(c_maxVal)] + c_ticklabels)
    #~ # #############################
    #~ plt_ax = plot_all_borders(plt_ax, element_lines, element_lines_t)
    #~ # #############################
    #~ return plt, plt_ax


def par_plot(plt, plt_ax, input_dict, json_elem_dict, lprint_str, xy_mtri, xy_triang, \
plt_title, par_name, c_minVal, c_maxVal, c_bar_ticks, c_fmt_punct, cmap, logn, \
c_linewidth, c_leg_fontsize, ax_linewidth, ax_fontsize, c_s, save_png, outfold, png_outfold):
    # #############################
    lprint(lprint_str)
    plt, plt_ax, add_legend_dict = pre_plot(plt, input_dict, json_elem_dict, ax_fontsize, c_linewidth)
    # #############################
    if logn:
        tpc = plt_ax.tripcolor(xy_mtri, xy_triang, shading='flat', cmap=cmap, \
        vmin=c_minVal, vmax=c_maxVal, norm=clr.LogNorm())
    else:
        tpc = plt_ax.tripcolor(xy_mtri, xy_triang, shading='flat', cmap=cmap, \
        vmin=c_minVal, vmax=c_maxVal)        
    divider = make_axes_locatable(plt_ax)
    # #############################
    plt.title(plt_title, fontsize=ax_fontsize)
    # #############################
    for cc_s in c_s:
        plt_ax, n_c_s = plot_border_inner(plt_ax, [cc_s], 0, 1, [], l_in=c_linewidth*0.3)
    # #############################
    plt, plt_ax = finalize_plot(plt, plt_ax, ax_linewidth, ax_fontsize)
    # #############################
    cax = divider.append_axes("right", size="5%", pad=0.05)
    #~ c_fmt = FuncFormatter(lambda x, p: c_fmt_punct.format(x))
    c_bar = plt.colorbar(tpc, ax=plt_ax, cax=cax)#, format=c_fmt)
    c_bar.ax.tick_params(width=ax_linewidth, labelsize=c_leg_fontsize)
    #~ for axis in ['top','bottom','left','right']: c_bar.ax.spines[axis].set_linewidth(ax_linewidth)
    c_bar.outline.set_linewidth(ax_linewidth)
    c_bar.set_ticks(c_bar_ticks)
    c_bar_ticklabels = map(lambda x: c_fmt_punct.format(x), c_bar_ticks)
    c_bar.set_ticklabels(c_bar_ticklabels)
    #~ c_ticks = [float(t.get_text()) for t in c_bar.ax.get_ymajorticklabels()]
    #~ if len(c_ticks) < 3:
        #~ c_ticklabels = [t.get_text() for t in c_bar.ax.get_ymajorticklabels()]
        #~ c_bar.set_ticks([c_minVal, c_maxVal] + c_ticks)
        #~ c_bar.set_ticklabels([c_fmt_punct.format(c_minVal), \
        #~ c_fmt_punct.format(c_maxVal)] + c_ticklabels)
    # #############################
    plt_ax.tick_params(labelsize=ax_fontsize*0.85)
    plt_ax.annotate('', xy=(0.02, 0.55), xycoords='figure fraction', xytext=(0.02, 0.45), arrowprops=dict(arrowstyle="-", color='w'))
    plt_ax.annotate('', xy=(0.98, 0.55), xycoords='figure fraction', xytext=(0.98, 0.45), arrowprops=dict(arrowstyle="-", color='w'))
    figure = plt.gcf()
    figure.savefig(par_name, format='svg', bbox_inches='tight')
    if save_png: figure.savefig((par_name[:-3] + 'png').replace(outfold, png_outfold), format='png', dpi=600, bbox_inches='tight')
    # #############################
    return c_s


def write_soi(c_soi, c_fid):
    for jj in range(2):
        for kk in range(len(c_soi)):
            c_fid.write('{:10.4f}'.format(c_soi[kk][jj]))
            if (kk+1) == len(c_soi): c_fid.write('\n'); break
            if not (kk+1)%8: c_fid.write('\n')
    return c_fid


def create_soi(no_soi, c_val):
    c_soi = []
    for kk in range(len(no_soi)):
        c_soi.append([no_soi[kk], c_val])
    return c_soi


def resp_spectra_acc(dt,acc_data,zz=0.05,st=0.04,ed=4.00,nu=80):

    # #############################
    #~ INPUT:
    #~ dt = sampling interval [s]
    #~ acc_data = vector of acceleration values [e.g. cm/s^2 or g]
    #~ zz = damping [decimal] - default: 0.05
    #~ st = first period computed (after period=0) [s] - default: 0.04
    #~ ed = last period computed [s] - default: 4.00
    #~ nu = number of spectral values (periods: [0,st...ed]) - default: 80
    # #############################
    #~ OUTPUT:
    #~ SA_UE = vector of spectral acceleration values [same as "acc_data"]
    #~ SD_UE = vector of spectral displacement values [same as "acc_data"*s^2]
    #~ T = vector of periods [s]
    # #############################

    T = np.append([.0], np.logspace(np.log10(st), np.log10(ed), num=nu-1, \
    endpoint=True, base=10))
    nT = len(T)
    siz = len(acc_data)
    amax = max(abs(acc_data))
    SA_UE = np.zeros(nT); SA_UE[0] = amax
    SD_UE = np.zeros(nT); SD_UE[0] = 0
    for j in xrange(1,nT):
        w0 = (2 * np.pi) / T[j]
        wd = w0 * np.sqrt(1 - zz**2)
        w2 = w0**2
        w3 = w2 * w0
        nacc = siz-1
        xd = np.zeros(nacc)
        xv = np.zeros(nacc)
        AA = np.zeros(nacc)
        za = 0; zv = 0; zd = 0
        E = np.exp(-zz*w0*dt)
        S = np.sin(wd*dt)
        C = np.cos(wd*dt)
        H1 = (wd * E * C) - (zz * w0 * E * S)
        H2 = (wd * E * S) + (zz * w0 * E * C)
        for k in xrange(0,nacc-1):
            dacc = acc_data[k+1] - acc_data[k]
            z1 = (1. / w2) * dacc
            z2 = (1. / w2) * acc_data[k]
            z3 = ((2. * zz) / (w3 * dt)) * dacc
            z4 = z1 / dt
            B = xd[k] + z2 - z3
            A = ((1 / wd) * xv[k]) + (((zz * w0) /wd) * B) + ((1 / wd) * z4)
            xd[k+1] = A * E * S + B * E * C + z3 - z2 - z1
            xv[k+1] = A * H1 - B * H2 - z4
            AA[k+1] = - 2 * zz * w0 * xv[k+1] - w2 * xd[k+1]
        SA_UE[j] = max(abs(AA))
        SD_UE[j] = max(abs(xd))
    return SA_UE, SD_UE, T


def Fourier_spectra_amplitude(dt,acc_data,tap=2.5):

    # #############################
    #~ INPUT:
    #~ dt = sampling interval [s]
    #~ acc_data = vector of acceleration values [e.g. cm/s^2 or g]
    #~ tap = % of taper applied to acc_data (at both start and end)
    # #############################
    #~ OUTPUT:
    #~ FFT_amp = vector of amplitudes [same as "acc_data"]
    #~ FFT_freq = vector of frequencies [Hz]
    # #############################

    fs = 1./dt
    l_acc = len(acc_data)
    l_tap = int(l_acc*tap/100.0)
    w_tap = np.hanning(2*l_tap)
    acc_data[:l_tap] = np.multiply(acc_data[:l_tap],w_tap[:l_tap])
    acc_data[-l_tap:] = np.multiply(acc_data[-l_tap:],w_tap[-l_tap:])

    npts = int(np.exp2(np.ceil(np.log2(l_acc))))
    fft_outst = np_fft(acc_data, n=npts)
    FFT_amp = np.absolute(fft_outst[0:npts/2]*dt)
    FFT_freq = fs*np.arange(0,npts/2)/npts

    return FFT_amp, FFT_freq


def KonnoOhmachi(SP_in,freq,b):

    #~ SMOOTHA SECONDO KONNO - OHMACHI
    #~ INPUT  : SP_in - spettro  N.B.: VETTORE NUMPY
    #~          freq  - frequenze  N.B.: VETTORE NUMPY
    #~          b     - coefficiente 'b'
    #~ OUTPUT : SP_ou - spettro smoothato
    
    l_SP = len(SP_in)    
    SP_ou = np.zeros_like(SP_in)
    KOvec = np.zeros_like(SP_in)
    f_fc = np.zeros_like(SP_in)
    for i in xrange(1,l_SP):
        f_fc[1:] = np.log10(np.power(np.multiply(freq[1:],np.divide(1.0,freq[i])),float(b)))
        ndxs = range(1, i) + range(i+1, l_SP)
        KOvec[ndxs] = np.power(np.divide(np.sin(f_fc[ndxs]),f_fc[ndxs]),4.0)
        KOvec[i] = 1.
        SP_ou[i] = np.dot(KOvec,SP_in) / np.sum(KOvec)
    SP_ou[0] = SP_ou[1]
    
    return SP_ou


def SP_resample(SP_in,freq,min_freq,max_freq):
    K_freq = np.logspace(np.log10(min_freq),np.log10(max_freq),num=100)
    K_f = interp1d(freq, SP_in, fill_value="extrapolate")
    K_SP = K_f(K_freq)
    return K_SP, K_freq


def mesh_ds(element_lines_H, element_lines_H_mask, ref_soil, min_F_nodes, json_elem_dict, \
input_dict, w_fac_mul, h_fac_mul, dict_width_s, dict_height_s, diff_x_lines, diff_y_lines, meth_s):
    
    c_vals_H = [2, 4, 8, 16, 32, 64]
    
    len_H = len(element_lines_H)
    len_H0 = len(element_lines_H[0])
    if len_H == len(diff_x_lines):
        mm_dir = 'x'
        s_fac_mul = w_fac_mul
        d_ref_s = dict_width_s
        diff_lines = diff_x_lines
        p_diff_lines = diff_y_lines
    elif len_H == len(diff_y_lines):
        mm_dir = 'y'
        s_fac_mul = h_fac_mul
        d_ref_s = dict_height_s
        diff_lines = diff_y_lines
        p_diff_lines = diff_x_lines
    dl_H0 = len_H0-2 # taking into account the two "lines" of fake elements added to the right and to the top of the grid

    mm_range = range(len_H-1)
    jj_range = range(len_H0-1)
    strata = int(input_dict[json_elem_dict["modp"]][json_elem_dict["modp_str"]])
    for mm in mm_range:
        
        start_hs = True # searching for... True: 'hs1', 'hd1', 'vs1', 'vd1' - False: 'q', 'hs2', 'hd2', 'vs2', 'vd2'
        
        for jj in jj_range:

            mask_mm = np.where(element_lines_H_mask[mm+1:,jj])[0]
            if not len(mask_mm): continue
            mk = mask_mm[0] + mm+1

            if not element_lines_H_mask[mm][jj]: continue
            if not element_lines_H[mm][jj][0] == 'q': continue
            if not element_lines_H[mm][jj][1][1] == ref_soil: continue
            
            mask_mk = np.where(element_lines_H_mask[mk+1:,jj])[0]
            if not len(mask_mk): continue
            mh = mask_mk[0] + mk+1
            
            if not element_lines_H_mask[mk][jj]: continue
            if not element_lines_H[mk][jj][0] == 'q': continue
            if not element_lines_H[mk][jj][1][1] == ref_soil: continue
            
            mask_jj = np.where(element_lines_H_mask[mm,jj+1:])[0]
            if not len(mask_jj): continue
            jk = mask_jj[0] + jj+1
            
            mask_jk = np.where(element_lines_H_mask[mm,jk+1:])[0]
            if not len(mask_jk): continue
            jh = mask_jk[0] + jk+1
            
            if start_hs:
                stratum_H = json_elem_dict["prem"] + "%03d" % ref_soil
                max_dimension = d_ref_s[stratum_H] # m
                act_dimension = sum(diff_lines[mm:mk])
                nxt_dimension = sum(diff_lines[mm:mh])
                ndx_H = np.searchsorted(c_vals_H, max_dimension/act_dimension, side='left')
                if not ndx_H: continue
                
                nn_cont = -1    #  0: ('hs1', 'hd1') or ('vs1', 'vd1') have been found!
                                # -1: continue searching for ('hs1', 'hd1') or ('vs1', 'vd1')
                
                jn = jj
                nn_range = range(min_F_nodes - int(1))
                for nn in nn_range:
                    jm = jn
                    
                    mask_nn = np.where(element_lines_H_mask[mm,jn+1:])[0]
                    if not len(mask_nn): break
                    jn = mask_nn[0] + jn+1
                    
                    if not element_lines_H_mask[mm][jn]: continue
                    if not element_lines_H[mm][jn][0] == 'q': break
                    if not element_lines_H[mm][jn][1][1] == ref_soil: break
                    
                    if not element_lines_H_mask[mk][jn]: continue
                    if not element_lines_H[mk][jn][0] == 'q': break
                    if not element_lines_H[mk][jn][1][1] == ref_soil: break
                    
                    p_act_dimension = sum(p_diff_lines[jm:jn])
                    if nxt_dimension/p_act_dimension > s_fac_mul: break # check shape factor
                    
                    if nn == nn_range[-1]: nn_cont = 0
            
            p_jkh_dimension = sum(p_diff_lines[jk:jh])
            
            cond_H = element_lines_H_mask[mm][jk] and element_lines_H_mask[mk][jk] \
            and element_lines_H[mm][jk][0] == 'q' and element_lines_H[mk][jk][0] == 'q' \
            and element_lines_H[mm][jk][1][1] == ref_soil and element_lines_H[mk][jk][1][1] == ref_soil \
            and nxt_dimension/p_jkh_dimension < s_fac_mul
            
            if not nn_cont:
                start_hs = False
                nn_cont += 1
                if jj == 0: p_elm = 'Q' # i.e. on first endpoint of each polygonal-chain
                else: p_elm = '1' # ('hs1', 'hd1') or ('vs1', 'vd1')
            else:
                if start_hs: continue
                if cond_H or 0 < nn_cont < min_F_nodes - int(2):
                    nn_cont += 1
                    p_elm = 'Q' # Q = q[mm][jj] + q[mk][jj]
                else:
                    start_hs = True
                    nn_cont = -1
                    if jk == dl_H0: p_elm = 'Q' # i.e. on last endpoint of each polygonal-chain
                    else: p_elm = '2' # ('hs2', 'hd2') or ('vs2', 'vd2')
            
            mm_p0 = element_lines_H[mm][jj][1][0][0]
            mm_p1 = element_lines_H[mm][jj][1][0][1]
            mm_p2 = element_lines_H[mm][jj][1][0][2]
            mm_p3 = element_lines_H[mm][jj][1][0][3]
            mk_p0 = element_lines_H[mk][jj][1][0][0]
            mk_p1 = element_lines_H[mk][jj][1][0][1]
            mk_p2 = element_lines_H[mk][jj][1][0][2]
            mk_p3 = element_lines_H[mk][jj][1][0][3]
            
            if p_elm == '1':
                element_lines_H[mm][jj][0] = 's1' # 'hs1' or 'vs1'
                if meth_s:
                    element_lines_H[mm][jj][1] = [[mm_p0, mm_p1, mm_p3, mm_p3], ref_soil, 't']
                    element_lines_H[mm][jj].append([[mm_p1, mk_p2, mm_p3, mm_p3], ref_soil, 't'])
                else:
                    element_lines_H[mm][jj][1] = [[mm_p0, mm_p1, mk_p2, mm_p3], ref_soil, 'q']
                element_lines_H[mk][jj][0] = 'd1' # 'hd1' or 'vd1'
                if mm_dir == 'x':
                    element_lines_H[mk][jj][1] = [[mk_p0, mk_p1, mk_p2, mk_p2], ref_soil, 't']
                elif mm_dir == 'y':
                    element_lines_H[mk][jj][1] = [[mk_p0, mk_p2, mk_p3, mk_p3], ref_soil, 't']
            elif p_elm == 'Q':
                if mm_dir == 'x':
                    element_lines_H[mm][jj][1] = [[mm_p0, mk_p1, mk_p2, mm_p3], ref_soil, 'q']
                elif mm_dir == 'y':
                    element_lines_H[mm][jj][1] = [[mm_p0, mm_p1, mk_p2, mk_p3], ref_soil, 'q']
                # set mask[mk][jj] = False
                element_lines_H_mask[mk][jj] = False
            elif p_elm == '2':
                element_lines_H[mm][jj][0] = 's2' # 'hs2' or 'vs2'
                if mm_dir == 'x':
                    if meth_s:
                        element_lines_H[mm][jj][1] = [[mm_p0, mm_p2, mm_p3, mm_p3], ref_soil, 't']
                        element_lines_H[mm][jj].append([[mm_p0, mk_p1, mm_p2, mm_p2], ref_soil, 't'])
                    else:
                        element_lines_H[mm][jj][1] = [[mm_p0, mk_p1, mm_p2, mm_p3], ref_soil, 'q']
                elif mm_dir == 'y':
                    if meth_s:
                        element_lines_H[mm][jj][1] = [[mm_p0, mm_p2, mk_p3, mk_p3], ref_soil, 't']
                        element_lines_H[mm][jj].append([[mm_p0, mm_p1, mm_p2, mm_p2], ref_soil, 't'])
                    else:
                        element_lines_H[mm][jj][1] = [[mm_p0, mm_p1, mm_p2, mk_p3], ref_soil, 'q']
                element_lines_H[mk][jj][0] = 'd2' # 'hd2' or 'vd2'
                element_lines_H[mk][jj][1] = [[mk_p1, mk_p2, mk_p3, mk_p3], ref_soil, 't']

    element_lines_array, element_lines_plot, element_lines_write, xy_triang, xy_triang_ndxs, z_triang = \
    redefine_elements(element_lines_H, element_lines_H_mask, strata, json_elem_dict, input_dict)
    
    element_lines_H = element_lines_numpy(element_lines_H)
    element_lines_H_mask = np.array(element_lines_H_mask)
    
    return element_lines_H, element_lines_H_mask, element_lines_array, \
    element_lines_plot, element_lines_write, xy_triang, xy_triang_ndxs, z_triang


def redefine_elements(element_lines_H, element_lines_H_mask, strata, json_elem_dict, input_dict):

    # redefine element_lines_array, element_lines_plot, element_lines_write, xy_triang, xy_triang_ndxs, z_triang
    element_lines_array = []
    element_lines_plot = []
    element_lines_write = []
    xy_triang = []
    xy_triang_ndxs = []
    z_triang = []
    len_H = len(element_lines_H)
    len_H0 = len(element_lines_H[0])
    
    dict_elm_type = {'q': 'quadrilateral', 't': 'triangle'}
    ccw_continue = True
    for mm in range(len_H):
        for jj in range(len_H0):
            if not element_lines_H_mask[mm][jj]: continue
            for elm_H in element_lines_H[mm][jj][1:]:

                c_node_a = elm_H[0][0]
                c_node_b = elm_H[0][1]
                c_node_c = elm_H[0][2]
                c_node_d = elm_H[0][3]
                
                c_element_lines, c_element_lines_H, element_lines_array, \
                element_lines_plot, element_lines_write, ccw_continue = \
                add_element(c_node_a, c_node_b, c_node_c, c_node_d, strata, \
                json_elem_dict, input_dict, element_lines_array, element_lines_plot, \
                element_lines_write, dict_elm_type[elm_H[2]], ccw_continue, elm_H[1])

                xy_triang, xy_triang_ndxs, z_triang = \
                add_element_triang(xy_triang, xy_triang_ndxs, z_triang, \
                c_element_lines, element_lines_write[-1])
    
    return element_lines_array, element_lines_plot, element_lines_write, xy_triang, xy_triang_ndxs, z_triang


def add_fake_elements(element_lines_H, diff_x_lines, diff_y_lines):
    # add two "lines" of fake elements to the right of the grid
    lst_diff = diff_x_lines[-1]
    element_lines_H.append(copy.deepcopy(element_lines_H[-1]))
    for f_elm_H in element_lines_H[-1]:
        if f_elm_H[1][2] == 't': return f_elm_H, None, 1
        f_elm_H[1][1] = 0
        f_elm_H[1][0][0] = copy.deepcopy(f_elm_H[1][0][1])
        f_elm_H[1][0][3] = copy.deepcopy(f_elm_H[1][0][2])
        f_elm_H[1][0][1][1] = f_elm_H[1][0][1][1] + lst_diff
        f_elm_H[1][0][2][1] = f_elm_H[1][0][2][1] + lst_diff
        f_elm_H[1][0][1][0] = f_elm_H[1][0][1][0] + 100000
        f_elm_H[1][0][2][0] = f_elm_H[1][0][2][0] + 100000
    element_lines_H.append(copy.deepcopy(element_lines_H[-1]))
    for f_elm_H in element_lines_H[-1]:
        f_elm_H[1][0][0] = copy.deepcopy(f_elm_H[1][0][1])
        f_elm_H[1][0][3] = copy.deepcopy(f_elm_H[1][0][2])
        f_elm_H[1][0][1][1] = f_elm_H[1][0][1][1] + lst_diff
        f_elm_H[1][0][2][1] = f_elm_H[1][0][2][1] + lst_diff
        f_elm_H[1][0][1][0] = f_elm_H[1][0][1][0] + 100000
        f_elm_H[1][0][2][0] = f_elm_H[1][0][2][0] + 100000
    diff_x_lines = np.append(np.append(diff_x_lines, lst_diff), lst_diff)
    element_lines_H = map(list, zip(*element_lines_H))

    # add two "lines" of fake elements to the top of the grid
    lst_diff = diff_y_lines[-1]
    element_lines_H.append(copy.deepcopy(element_lines_H[-1]))
    for f_elm_H in element_lines_H[-1]:
        if f_elm_H[1][2] == 't': return f_elm_H, 1, None
        f_elm_H[1][1] = 0
        f_elm_H[1][0][0] = copy.deepcopy(f_elm_H[1][0][3])
        f_elm_H[1][0][1] = copy.deepcopy(f_elm_H[1][0][2])
        f_elm_H[1][0][3][2] = f_elm_H[1][0][3][2] + lst_diff
        f_elm_H[1][0][2][2] = f_elm_H[1][0][2][2] + lst_diff
        f_elm_H[1][0][3][0] = f_elm_H[1][0][3][0] + 100000
        f_elm_H[1][0][2][0] = f_elm_H[1][0][2][0] + 100000
    element_lines_H.append(copy.deepcopy(element_lines_H[-1]))
    for f_elm_H in element_lines_H[-1]:
        f_elm_H[1][0][0] = copy.deepcopy(f_elm_H[1][0][3])
        f_elm_H[1][0][1] = copy.deepcopy(f_elm_H[1][0][2])
        f_elm_H[1][0][3][2] = f_elm_H[1][0][3][2] + lst_diff
        f_elm_H[1][0][2][2] = f_elm_H[1][0][2][2] + lst_diff
        f_elm_H[1][0][3][0] = f_elm_H[1][0][3][0] + 100000
        f_elm_H[1][0][2][0] = f_elm_H[1][0][2][0] + 100000
    diff_y_lines = np.append(np.append(diff_y_lines, lst_diff), lst_diff)
    element_lines_H = map(list, zip(*element_lines_H))
    
    return element_lines_H, diff_x_lines, diff_y_lines


def element_lines_numpy(element_lines_H):
    element_lines_H_NEW = np.full((len(element_lines_H),len(element_lines_H[0])), np.nan, dtype=np.object)
    for ira in range(len(element_lines_H)):
        for jra in range(len(element_lines_H[0])):
            element_lines_H_NEW[ira][jra] = element_lines_H[ira][jra]
    element_lines_H = copy.deepcopy(element_lines_H_NEW)
    return element_lines_H
    # ~ return element_lines_H_NEW


def search_id(element_lines_H, mo_arr):
    for e_l_H_arr in element_lines_H:
        for e_l_H in e_l_H_arr:
            for e_l in e_l_H[1][0]:
                if mo_arr == e_l[1:]:
                    return e_l
