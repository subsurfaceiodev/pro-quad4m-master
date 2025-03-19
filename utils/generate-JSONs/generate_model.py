#!/usr/bin/env python

from collections import OrderedDict 
nodal_points = OrderedDict()

# ############################# #
# ############################# #
#   START EDIT HEREAFTER ...    #
# ############################# #
# ############################# #


# ----------------------------------start-
# ---- block ["modelling_parameters"] ----
# ----------------------------------start-

strata = 6
job_title = 'test #1'
job_folder = 'test_1'
NUMB_iteration = 4
half_space_VP_m_s = 3200
half_space_VS_m_s = 1350
half_space_RHO_N_m_3 = 22000

inside_X = None
inside_Y = None
inside_XY = [[200, 164.5], [200, 145], [200, 113]]
ground_X = None
ground_Y = None
ground_XY = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]

# start-of-inner-block ["modelling_parameters"]["boundary_conditions"]
# ----------------------------------------------------------------------
horizontal_boundary = {'position': 0, 'value': 4}
vertical_left_boundary = {'position': -100, 'value': 2}
vertical_right_boundary = {'position': 500, 'value': 2}
# ----------------------------------------------------------------------
# end-of-inner-block ["modelling_parameters"]["boundary_conditions"]

# start-of-inner-block ["modelling_parameters"]["nodal_points"]
# ----------------------------------------------------------------------
nodal_points[1] = [-100, 193]
nodal_points[2] = [10, 193]
nodal_points[3] = [25, 200]
nodal_points[4] = [35, 200]
nodal_points[5] = [50, 195]
nodal_points[6] = [60, 195]
nodal_points[7] = [75, 190]
nodal_points[8] = [90, 190]
nodal_points[9] = [125, 185]
nodal_points[10] = [140, 182]
nodal_points[11] = [160, 183]
nodal_points[12] = [165, 184]
nodal_points[13] = [175, 185]
nodal_points[14] = [190, 182]
nodal_points[15] = [210, 188]
nodal_points[16] = [225, 187]
nodal_points[17] = [235, 187]
nodal_points[18] = [255, 190]
nodal_points[19] = [265, 193]
nodal_points[20] = [300, 194]
nodal_points[21] = [315, 190]
nodal_points[22] = [335, 185]
nodal_points[23] = [355, 180]
nodal_points[24] = [365, 180]
nodal_points[25] = [380, 185]
nodal_points[26] = [500, 185]
nodal_points[27] = [90, 183]
nodal_points[28] = [110, 177]
nodal_points[29] = [130, 176]
nodal_points[30] = [140, 178]
nodal_points[31] = [150, 168]
nodal_points[32] = [160, 164]
nodal_points[33] = [200, 165]
nodal_points[34] = [210, 168]
nodal_points[35] = [140, 158]
nodal_points[36] = [160, 148]
nodal_points[37] = [180, 128]
nodal_points[38] = [200, 115]
nodal_points[39] = [220, 108]
nodal_points[40] = [230, 112]
nodal_points[41] = [260, 125]
nodal_points[42] = [275, 130]
nodal_points[43] = [280, 137]
nodal_points[44] = [290, 147]
nodal_points[45] = [285, 155]
nodal_points[46] = [300, 175]
nodal_points[47] = [280, 175]
nodal_points[48] = [260, 177]
nodal_points[49] = [240, 182]
nodal_points[50] = [310, 180]
nodal_points[51] = [325, 182]
nodal_points[52] = [330, 183]
nodal_points[53] = [335, 182]
nodal_points[54] = [325, 175]
nodal_points[55] = [330, 170]
nodal_points[56] = [320, 165]
nodal_points[57] = [315, 152]
nodal_points[58] = [310, 140]
nodal_points[59] = [300, 135]
nodal_points[60] = [500, 0]
nodal_points[61] = [-100, 0]
# ----------------------------------------------------------------------
# end-of-inner-block ["modelling_parameters"]["nodal_points"]

# -----------------------------------end--
# ---- block ["modelling_parameters"] ----
# -----------------------------------end--

# ----------------------------------start-
# ------ block ["mesh_generation"] -------
# ----------------------------------start-

maximum_frequency = 12.0

# start-of-inner-block ["mesh_generation"]["maximum_element_shape_factor"]
# ----------------------------------------------------------------------
max_H_V = 2.0
max_V_H = 1.5
# ----------------------------------------------------------------------
# end-of-inner-block ["mesh_generation"]["maximum_element_shape_factor"]

uniform_initial_grid = False

# start-of-inner-block ["mesh_generation"]["rearrange_nodes"]
# ----------------------------------------------------------------------
rearrange_nodes_tf = True
scaling_factor = 0.95
iterations = 20
vertical_deformation_coefficient = 0.30
horizontal_deformation_coefficient = 0.15
distance_threshold_m = 50.0
exponents = {'inline': 1.5, 'inline_extremities': 1.0, 'orthogonal': 2.0}
overall_exponent = None
# ----------------------------------------------------------------------
# end-of-inner-block ["mesh_generation"]["rearrange_nodes"]

boundaries_triangular_elements_tf = True

# start-of-inner-block ["mesh_generation"]["mesh_downsampling"]
# ----------------------------------------------------------------------
mesh_downsampling_tf = True
min_F_nodes = int(6)
multiplier_H_V = float(2.2)
multiplier_V_H = float(1.5)
triangular_extremities_tf = True
# ----------------------------------------------------------------------
# end-of-inner-block ["mesh_generation"]["mesh_downsampling"]

# -----------------------------------end--
# ------ block ["mesh_generation"] -------
# -----------------------------------end--

# ----------------------------------start-
# --------- block ["waveforms"] ----------
# ----------------------------------start-

N1EQ_first_time_step_for_last_iteration = 1
KGEQ_last_time_step_for_last_iteration = 2250
N2EQ_first_time_step_for_first_iterations = 300
N3EQ_last_time_step_for_first_iterations = 1300
HDRX_header_lines_in_horizontal_input_time_history = 64
HDRY_header_lines_in_vertical_input_time_history = 64
DTEQ_time_step_s = 0.02
multiplier_factor_to_g = 0.0010197
PRINPUT_period_of_maximum_spectral_hz_acc_s = 0.22

# -----------------------------------end--
# --------- block ["waveforms"] ----------
# -----------------------------------end--

# ----------------------------------start-
# --- block "dynamic soil properties" ----
# ----------------------------------start-

# formation CD
# G/GMX
G_GMX_strain_CD = [0.0001, 0.0039, 0.0059, 0.0076, 0.0092, 0.0106, 0.0140, \
0.0160, 0.0174, 0.0187, 0.0200, 0.0491, 0.0741, 0.1127, 0.1785, 0.3081, 0.6332]
G_GMX_CD = [1.0000, 0.9900, 0.9800, 0.9700, 0.9600, 0.9500, 0.9250, 0.9100, \
0.9000, 0.8900, 0.8800, 0.7000, 0.6000, 0.5000, 0.4000, 0.3000, 0.2000]
# XL
XL_strain_CD = [0.0001, 0.0244, 0.0335, 0.0405, 0.0467, 0.0522, 0.0648, \
0.0718, 0.0764, 0.0809, 0.0854, 0.1770, 0.2505, 0.3588, 0.5353, 0.6620, 0.6621]
XL_CD = [3.0000, 3.3400, 3.6800, 4.0200, 4.3600, 4.7000, 5.5500, 6.06, \
6.40, 6.74, 7.08, 13.20, 16.60, 20, 23.40, 24.998, 25]
# formation AA
# G/GMX
G_GMX_strain_AA = [0.0001, 0.0060, 0.0084, 0.0103, 0.0119, 0.0134, 0.0169, \
0.0188, 0.0200, 0.0213, 0.0225, 0.0480, 0.0686, 0.0992, 0.1494, 0.2442, 0.4696]
G_GMX_AA = [1.0000, 0.9900, 0.9800, 0.9700, 0.9600, 0.9500, 0.9250, 0.9100, \
0.9000, 0.8900, 0.8800, 0.7000, 0.6000, 0.5000, 0.4000, 0.3000, 0.2000]
# XL
XL_strain_AA = [0.0001, 0.0084, 0.0127, 0.0163, 0.0196, 0.0226, 0.0298, \
0.0339, 0.0366, 0.0394, 0.0421, 0.1020, 0.1533, 0.2320, 0.3659, 0.6283, 0.8585]
XL_AA = [2.7700, 3.0669, 3.3638, 3.6607, 3.9576, 4.2545, 4.9968, 5.44, \
5.74, 6.04, 6.33, 11.68, 14.64, 17.61, 20.58, 23.55, 24.98]
# formation AG
# G/GMX
G_GMX_strain_AG = [0.0001, 0.0091, 0.0119, 0.0139, 0.0157, 0.0173, 0.0208, \
0.0227, 0.0239, 0.0252, 0.0264, 0.0503, 0.0689, 0.0959, 0.1389, 0.2176, 0.3971]
G_GMX_AG = [1.0000, 0.9900, 0.9800, 0.9700, 0.9600, 0.9500, 0.9250, 0.9100, \
0.9000, 0.8900, 0.8800, 0.7000, 0.6000, 0.5000, 0.4000, 0.3000, 0.2000]
# XL
XL_strain_AG = [0.0001, 0.0127, 0.0181, 0.0224, 0.0262, 0.0296, 0.0376, \
0.0421, 0.0450, 0.0480, 0.0509, 0.1118, 0.1619, 0.2367, 0.3606, 0.5964, 0.6906]
XL_AG = [2.5000, 2.8103, 3.1206, 3.4309, 3.7412, 4.0515, 4.8273, 5.29, \
5.60, 5.91, 6.22, 11.81, 14.91, 18.01, 21.12, 24.22, 25.00]

# -----------------------------------end--
# --- block "dynamic soil properties" ----
# -----------------------------------end--

# ----------------------------------start-
# ------ block "strata properties" -------
# ----------------------------------start-

# start-of-inner-block [stratum001]
# ----------------------------------------------------------------------
st001_name = 'st. 1'
# stratum 1: nodal points
st001_n_ps = [1, 2, 3, 4, 5, 6, 7, 8, 27, 28, 29, 35, 36, 37, 38, 39, \
40, 41, 42, 59, 58, 57, 56, 55, 54, 52, 53, 22, 23, 24, 25, 26, 60, 61]
st001_DENS_N_m_3 = {'value': 22000, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
st001_PO_ratio = {'value': 0.392, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
st001_GMX_kN_m_3 = {'value': 4907000, 'min_value': 4498000, 'uncentanty': 204400, 'log_normal_tf': False}
st001_G_kN_m_3 = {'percent_of_GMX': 100}
st001_XL_decimal = {'value': 0.005, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
# dynamic soil properties
st001_soil_TYPE_G_GMX = None
st001_soil_TYPE_G_GMX_strain = None
st001_soil_TYPE_XL_decimal = None
st001_soil_TYPE_XL_strain = None
# model_G_GMX_XL
st001_model_G_GMX_XL = {'model_G_GMX_XL_tf': False, 'depth_of_water_table_m': 0.0, \
'coef_of_lateral_earth_pressure_K0': 0.5, 'mean_effective_stress_atm': None, \
'plasticity_index_PI_percent': 0.0, 'over_consolidation_ratio_OCR': 1.0, 'XL_uncentanty': True}
# ----------------------------------------------------------------------
# end-of-inner-block [stratum001]

# start-of-inner-block [stratum002]
# ----------------------------------------------------------------------
st002_name = 'st. 2'
# stratum 2: nodal points
st002_n_ps = [52, 54, 55, 56, 57, 58, 59, 42, 43, 44, 45, 46, 50, 51]
st002_DENS_N_m_3 = {'value': 21200, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
st002_PO_ratio = {'value': 0.481, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
st002_GMX_kN_m_3 = {'value': 440600, 'min_value': 363600, 'uncentanty': 18360, 'log_normal_tf': False}
st002_G_kN_m_3 = {'percent_of_GMX': 90}
st002_XL_decimal = {'value': 0.025, 'min_value': 0.023, 'uncentanty': 0.001, 'log_normal_tf': False}
# dynamic soil properties
st002_soil_TYPE_G_GMX = G_GMX_AG
st002_soil_TYPE_G_GMX_strain = G_GMX_strain_AG
st002_soil_TYPE_XL_decimal = XL_AG
st002_soil_TYPE_XL_strain = XL_strain_AG
# model_G_GMX_XL
st002_model_G_GMX_XL = {'model_G_GMX_XL_tf': False, 'depth_of_water_table_m': 0.0, \
'coef_of_lateral_earth_pressure_K0': 0.5, 'mean_effective_stress_atm': None, \
'plasticity_index_PI_percent': 0.0, 'over_consolidation_ratio_OCR': 1.0, 'XL_uncentanty': True}
# ----------------------------------------------------------------------
# end-of-inner-block [stratum002]

# start-of-inner-block [stratum003]
# ----------------------------------------------------------------------
st003_name = 'st. 3'
# stratum 3: nodal points
st003_n_ps = [29, 30, 31, 32, 33, 34, 16, 17, 49, 48, 47, 46, 45, 44, 43, \
42, 41, 40, 39, 38, 37, 36, 35]
st003_DENS_N_m_3 = {'value': 21200, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
st003_PO_ratio = {'value': 0.477, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
st003_GMX_kN_m_3 = {'value': 594100, 'min_value': 490200, 'uncentanty': 24760, 'log_normal_tf': False}
st003_G_kN_m_3 = {'percent_of_GMX': 90}
st003_XL_decimal = {'value': None, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
# dynamic soil properties
st003_soil_TYPE_G_GMX = None
st003_soil_TYPE_G_GMX_strain = None
st003_soil_TYPE_XL_decimal = None
st003_soil_TYPE_XL_strain = None
# model_G_GMX_XL
st003_model_G_GMX_XL = {'model_G_GMX_XL_tf': True, 'depth_of_water_table_m': 0.0, \
'coef_of_lateral_earth_pressure_K0': 0.5, 'mean_effective_stress_atm': None, \
'plasticity_index_PI_percent': 0.0, 'over_consolidation_ratio_OCR': 1.0, 'XL_uncentanty': False}
# ----------------------------------------------------------------------
# end-of-inner-block [stratum003]

# start-of-inner-block [stratum004]
# ----------------------------------------------------------------------
st004_name = 'st. 4'
# stratum 4: nodal points
st004_n_ps = [27, 28, 29, 30, 10, 9, 8]
st004_DENS_N_m_3 = {'value': 21150, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
st004_PO_ratio = {'value': 0.489, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
st004_GMX_kN_m_3 = {'value': 202700, 'min_value': 148600, 'uncentanty': 8445, 'log_normal_tf': False}
st004_G_kN_m_3 = {'percent_of_GMX': 80}
st004_XL_decimal = {'value': 0.0277, 'min_value': 0.0257, 'uncentanty': 0.001, 'log_normal_tf': False}
# dynamic soil properties
st004_soil_TYPE_G_GMX = G_GMX_AA
st004_soil_TYPE_G_GMX_strain = G_GMX_strain_AA
st004_soil_TYPE_XL_decimal = XL_AA
st004_soil_TYPE_XL_strain = XL_strain_AA
# model_G_GMX_XL
st004_model_G_GMX_XL = {'model_G_GMX_XL_tf': False, 'depth_of_water_table_m': 0.0, \
'coef_of_lateral_earth_pressure_K0': 0.5, 'mean_effective_stress_atm': None, \
'plasticity_index_PI_percent': 0.0, 'over_consolidation_ratio_OCR': 1.0, 'XL_uncentanty': True}
# ----------------------------------------------------------------------
# end-of-inner-block [stratum004]

# start-of-inner-block [stratum005]
# ----------------------------------------------------------------------
st005_name = 'st. 5'
# stratum 5: nodal points
st005_n_ps = [17, 18, 19, 20, 21, 22, 53, 52, 51, 50, 46, 47, 48, 49]
st005_DENS_N_m_3 = {'value': 21200, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
st005_PO_ratio = {'value': 0.485, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
st005_GMX_kN_m_3 = {'value': 331300, 'min_value': 273300, 'uncentanty': 13800, 'log_normal_tf': False}
st005_G_kN_m_3 = {'percent_of_GMX': 90}
st005_XL_decimal = {'value': 0.025, 'min_value': 0.023, 'uncentanty': 0.001, 'log_normal_tf': False}
# dynamic soil properties
st005_soil_TYPE_G_GMX = G_GMX_AG
st005_soil_TYPE_G_GMX_strain = G_GMX_strain_AG
st005_soil_TYPE_XL_decimal = XL_AG
st005_soil_TYPE_XL_strain = XL_strain_AG
# model_G_GMX_XL
st005_model_G_GMX_XL = {'model_G_GMX_XL_tf': False, 'depth_of_water_table_m': 0.0, \
'coef_of_lateral_earth_pressure_K0': 0.5, 'mean_effective_stress_atm': None, \
'plasticity_index_PI_percent': 0.0, 'over_consolidation_ratio_OCR': 1.0, 'XL_uncentanty': True}
# ----------------------------------------------------------------------
# end-of-inner-block [stratum005]

# start-of-inner-block [stratum006]
# ----------------------------------------------------------------------
st006_name = 'st. 6'
# stratum 6: nodal points
st006_n_ps = [10, 11, 12, 13, 14, 15, 16, 34, 33, 32, 31, 30]
st006_DENS_N_m_3 = {'value': 19600, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
st006_PO_ratio = {'value': 0.493, 'min_value': None, 'uncentanty': None, 'log_normal_tf': False}
st006_GMX_kN_m_3 = {'value': 55160, 'min_value': 35400, 'uncentanty': 2295, 'log_normal_tf': False}
st006_G_kN_m_3 = {'percent_of_GMX': 70}
st006_XL_decimal = {'value': 0.030, 'min_value': 0.027, 'uncentanty': 0.0015, 'log_normal_tf': False}
# dynamic soil properties
st006_soil_TYPE_G_GMX = G_GMX_CD
st006_soil_TYPE_G_GMX_strain = G_GMX_strain_CD
st006_soil_TYPE_XL_decimal = XL_CD
st006_soil_TYPE_XL_strain = XL_strain_CD
# model_G_GMX_XL
st006_model_G_GMX_XL = {'model_G_GMX_XL_tf': False, 'depth_of_water_table_m': 0.0, \
'coef_of_lateral_earth_pressure_K0': 0.5, 'mean_effective_stress_atm': None, \
'plasticity_index_PI_percent': 0.0, 'over_consolidation_ratio_OCR': 1.0, 'XL_uncentanty': True}
# ----------------------------------------------------------------------
# end-of-inner-block [stratum006]

# -----------------------------------end--
# ------ block "strata properties" -------
# -----------------------------------end--

# ----------------------------------start-
# ------ block ["plot_parameters"] -------
# ----------------------------------start-

BW_scale_tf = False
profile_SA_max_width = 30.0
profile_SA_BW_scale_tf = False
profile_ACC_max_width = 45.0
profile_FFT_lower_limit_Hz = 0.07
scale = {'line_thicknesses': 0.7, 'fonts': 0.7, 'symbols': 0.7}
save_PNG_tf = True

# -----------------------------------end--
# ------ block ["plot_parameters"] -------
# -----------------------------------end--


# ############################# #
# ############################# #
#      ... STOP EDIT HERE       #
# ############################# #
# ############################# #


# ############################# #
# ############################# #
#  !DO NOT EDIT THE FOLLOWING!  #
# ############################# #
# ############################# #


from to_json import to_json
import argparse

p = argparse.ArgumentParser(description='description: Generate input JSON-file.')
p.add_argument("json", action="store", type=str, help='filename of JSON-file to be generated')
opts = p.parse_args()

modelling_parameters = OrderedDict()
plot_parameters = OrderedDict()
waveforms = OrderedDict()
mesh_generation = OrderedDict()
rearrange_nodes = OrderedDict()
mesh_downsampling = OrderedDict()
shape_factor = OrderedDict()
m_shape_factor = OrderedDict()
input_dict = OrderedDict()
acceleration_output = OrderedDict()

def v_u_create(a_dict, a_key='uncentanty', b_key=None, c_key=None):
    v_u_dict = OrderedDict()
    v_u_dict['value'] = a_dict['value']
    if b_key: v_u_dict[b_key] = a_dict[b_key]
    v_u_dict[a_key] = a_dict[a_key]
    if c_key: v_u_dict[c_key] = a_dict[c_key]
    return v_u_dict

modelling_parameters['job_title'] = job_title
modelling_parameters['job_folder'] = job_folder
modelling_parameters['strata'] = strata
modelling_parameters['nodal_points'] = nodal_points
modelling_parameters['NUMB_iteration'] = NUMB_iteration
modelling_parameters['half_space_VP_m_s'] = half_space_VP_m_s
modelling_parameters['half_space_VS_m_s'] = half_space_VS_m_s
modelling_parameters['half_space_RHO_N_m_3'] = half_space_RHO_N_m_3

boundary_conditions = {\
'horizontal': v_u_create(horizontal_boundary, a_key='position'), \
'vertical-left': v_u_create(vertical_left_boundary, a_key='position'), \
'vertical-right': v_u_create(vertical_right_boundary, a_key='position') \
}

acceleration_output['X_direction'] = {'inside': inside_X, 'ground': ground_X}
acceleration_output['Y_direction'] = {'inside': inside_Y, 'ground': ground_Y}
acceleration_output['XY_direction'] = {'inside': inside_XY, 'ground': ground_XY}

m_shape_factor['H/V'] = multiplier_H_V
m_shape_factor['V/H'] = multiplier_V_H
mesh_downsampling['mesh_downsampling_tf'] = mesh_downsampling_tf
mesh_downsampling['minimum_module_length'] = min_F_nodes
mesh_downsampling['element_shape_factor_multiplier'] = m_shape_factor
mesh_downsampling['triangular_extremities_tf'] = triangular_extremities_tf

rearrange_nodes['rearrange_nodes_tf'] = rearrange_nodes_tf
rearrange_nodes['scaling_factor'] = scaling_factor
rearrange_nodes['iterations'] = iterations
rearrange_nodes['vertical_deformation_coefficient'] = vertical_deformation_coefficient
rearrange_nodes['horizontal_deformation_coefficient'] = horizontal_deformation_coefficient
rearrange_nodes['distance_threshold_m'] = distance_threshold_m
rearrange_nodes['exponents'] = exponents
rearrange_nodes['overall_exponent'] = overall_exponent
shape_factor['H/V'] = max_H_V
shape_factor['V/H'] = max_V_H

waveforms['N1EQ_first_time_step_for_last_iteration'] = N1EQ_first_time_step_for_last_iteration
waveforms['KGEQ_last_time_step_for_last_iteration'] = KGEQ_last_time_step_for_last_iteration
waveforms['N2EQ_first_time_step_for_first_iterations'] = N2EQ_first_time_step_for_first_iterations
waveforms['N3EQ_last_time_step_for_first_iterations'] = N3EQ_last_time_step_for_first_iterations
waveforms['DTEQ_time_step_s'] = DTEQ_time_step_s
waveforms['HDRX_header_lines_in_horizontal_input_time_history'] = HDRX_header_lines_in_horizontal_input_time_history
waveforms['HDRY_header_lines_in_vertical_input_time_history'] = HDRY_header_lines_in_vertical_input_time_history
waveforms['multiplier_factor_to_g'] = multiplier_factor_to_g
waveforms['PRINPUT_period_corresponding_to_maximum_spectral_acceleration_of_horizontal_input_motion_s'] = PRINPUT_period_of_maximum_spectral_hz_acc_s

plot_parameters['BW_scale_tf'] = BW_scale_tf
plot_parameters['profile_SA_max_width'] = profile_SA_max_width
plot_parameters['profile_SA_BW_scale_tf'] = profile_SA_BW_scale_tf
plot_parameters['profile_ACC_max_width'] = profile_ACC_max_width
plot_parameters['profile_FFT_lower_limit_Hz'] = profile_FFT_lower_limit_Hz
plot_parameters['scale'] = scale
plot_parameters['save_PNG_tf'] = save_PNG_tf

mesh_generation['maximum_frequency'] = maximum_frequency
mesh_generation['maximum_element_shape_factor'] = shape_factor
mesh_generation['uniform_initial_grid'] = uniform_initial_grid
mesh_generation['rearrange_nodes'] = rearrange_nodes
mesh_generation['boundaries_triangular_elements_tf'] = boundaries_triangular_elements_tf
mesh_generation['mesh_downsampling'] = mesh_downsampling

#~ modelling_parameters['DENS_IDW'] = False
#~ modelling_parameters['GMX_IDW'] = False
#~ modelling_parameters['XL_IDW'] = False
modelling_parameters['boundary_conditions'] = boundary_conditions
modelling_parameters['acceleration_output'] = acceleration_output

def d_model_G_GMX_XL(st_model_G_GMX_XL):
    model_G_GMX_XL = OrderedDict()
    model_G_GMX_XL['model_G_GMX_XL_tf'] = st_model_G_GMX_XL['model_G_GMX_XL_tf']
    model_G_GMX_XL['depth_of_water_table_m'] = st_model_G_GMX_XL['depth_of_water_table_m']
    model_G_GMX_XL['coef_of_lateral_earth_pressure_K0'] = st_model_G_GMX_XL['coef_of_lateral_earth_pressure_K0']
    model_G_GMX_XL['mean_effective_stress_atm'] = st_model_G_GMX_XL['mean_effective_stress_atm']
    model_G_GMX_XL['plasticity_index_PI_percent'] = st_model_G_GMX_XL['plasticity_index_PI_percent']
    model_G_GMX_XL['over_consolidation_ratio_OCR'] = st_model_G_GMX_XL['over_consolidation_ratio_OCR']
    model_G_GMX_XL['XL_uncentanty'] = st_model_G_GMX_XL['XL_uncentanty']
    return model_G_GMX_XL

def soil_data_strain(G_GMX, G_GMX_strain, XL_decimal, XL_strain):
    TYPE_material = OrderedDict()
    if not G_GMX or not G_GMX_strain or not XL_decimal or not XL_strain:
        TYPE_material['G_GMX'] = None
        TYPE_material['XL'] = None
    else:
        G_GMX_vs_strain = map(list, zip(*[G_GMX_strain, G_GMX]))
        XL_vs_strain = map(list, zip(*[XL_strain, XL_decimal]))
        TYPE_material['G_GMX'] = G_GMX_vs_strain
        TYPE_material['XL'] = XL_vs_strain
    return TYPE_material

input_dict['modelling_parameters'] = modelling_parameters

for jjj in range(strata):
    iii = '{:03d}'.format(jjj+1)
    exec('stratum' + iii + ' = OrderedDict()')
    exec("stratum" + iii + "['name'] = st" + iii + "_name")
    exec("stratum" + iii + "['nodes'] = st" + iii + "_n_ps")
    exec("stratum" + iii + "['DENS_N_m_3'] = v_u_create(st" + iii + "_DENS_N_m_3, b_key='min_value', c_key='log_normal_tf')")
    exec("stratum" + iii + "['PO_ratio'] = v_u_create(st" + iii + "_PO_ratio, b_key='min_value', c_key='log_normal_tf')")
    exec("stratum" + iii + "['GMX_kN_m_3'] = v_u_create(st" + iii + "_GMX_kN_m_3, b_key='min_value', c_key='log_normal_tf')")
    exec("stratum" + iii + "['G_kN_m_3'] = st" + iii + "_G_kN_m_3")
    exec("stratum" + iii + "['XL_decimal'] = v_u_create(st" + iii + "_XL_decimal, b_key='min_value', c_key='log_normal_tf')")
    exec("st" + iii + "_soil_TYPE = soil_data_strain(st" + iii + \
    "_soil_TYPE_G_GMX, st" + iii + "_soil_TYPE_G_GMX_strain, st" + iii + \
    "_soil_TYPE_XL_decimal, st" + iii + "_soil_TYPE_XL_strain)")
    exec("stratum" + iii + "['soil_TYPE'] = st" + iii + "_soil_TYPE")
    exec("stratum" + iii + "['model_G_GMX_XL'] = d_model_G_GMX_XL(st" + iii + "_model_G_GMX_XL)")
    exec("input_dict['stratum" + iii + "'] = stratum" + iii + "")

input_dict['mesh_generation'] = mesh_generation
input_dict['waveforms'] = waveforms
input_dict['plot_parameters'] = plot_parameters

input_json = to_json(input_dict)
with open(opts.json, 'w') as h_input_json:
    h_input_json.write(input_json + '\n')

print opts.json + " has been updated"
