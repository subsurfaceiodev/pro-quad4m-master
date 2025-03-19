#!/usr/bin/env python

from to_json import to_json
from collections import OrderedDict 
import argparse

p = argparse.ArgumentParser(description='description: Generate schema JSON-file (see https://json-schema.org/).')
p.add_argument("json", action="store", type=str, help='filename of JSON-schema to be generated')
opts = p.parse_args()

# OrderedDict(s)
schema_dict_modelling_properties = OrderedDict()
schema_dict_plot_properties = OrderedDict()
schema_dict_stratum_properties = OrderedDict()
schema_dict_waveforms_properties = OrderedDict()
schema_dict = OrderedDict()
np_schema_dict = OrderedDict()
schema_dict_properties = OrderedDict()
boolean_dict = OrderedDict()
acc_out_inner_dict = OrderedDict()
acc_out_dict = OrderedDict()
exp_dict = OrderedDict()
soil_schema_inner_dict = OrderedDict()
acceleration_output_inner_dict = OrderedDict()
model_G_GMX_XL_schema_inner_dict = OrderedDict()
mesh_generation_inner_dict = OrderedDict()
rearrange_nodes_inner_dict = OrderedDict()
mesh_downsampling_inner_dict = OrderedDict()
shape_factor_inner_dict = OrderedDict()
scale_inner_dict = OrderedDict()
ground_nodes_inner_dict = OrderedDict()
ground_inside_inner_dict = OrderedDict()

# field arrays
modelling_required = ['strata', 'NUMB_iteration', 'nodal_points', 'boundary_conditions', \
'half_space_VS_m_s', 'half_space_VP_m_s', 'half_space_RHO_N_m_3', 'acceleration_output']
plot_required = ['BW_scale_tf', 'profile_SA_max_width', 'profile_SA_BW_scale_tf', 'scale', \
'profile_ACC_max_width', 'save_PNG_tf']
stratum_required = ['name', 'nodes', 'soil_TYPE', 'DENS_N_m_3', 'PO_ratio', \
'GMX_kN_m_3', 'G_kN_m_3', 'XL_decimal']
transmitting_required = ['horizontal','vertical-left','vertical-right']
waveforms_required = [\
'N1EQ_first_time_step_for_last_iteration', \
'KGEQ_last_time_step_for_last_iteration', \
'N2EQ_first_time_step_for_first_iterations', \
'N3EQ_last_time_step_for_first_iterations', \
'DTEQ_time_step_s', 'multiplier_factor_to_g', \
'PRINPUT_period_corresponding_to_maximum_spectral_acceleration_of_horizontal_input_motion_s', \
'HDRX_header_lines_in_horizontal_input_time_history', \
'HDRY_header_lines_in_vertical_input_time_history']

# boolean_dict
boolean_dict['type'] = 'boolean'

# acc_out_dict
#~ acc_out_inner_dict = { 'type': 'integer', 'minimum': 0, 'maximum': 4 }
acc_out_inner_dict['type'] = 'integer'
acc_out_inner_dict['minimum'] = 0
acc_out_inner_dict['maximum'] = 4
#~ acc_out_dict = {'position': {'type': 'number'}, 'value': acc_out_inner_dict}
acc_out_dict['position'] = {'type': 'number'}
acc_out_dict['value'] = acc_out_inner_dict

# exponents_dict
exp_dict['inline'] = {'type': 'number', 'minimum': 0.01}
exp_dict['inline_extremities'] = {'type': 'number', 'minimum': 0.01}
exp_dict['orthogonal'] = {'type': 'number', 'minimum': 0.01}

def core_schema_dict(required_array, dict_properties):
    c_schema_dict = OrderedDict()
    c_schema_dict['type'] = 'object'
    c_schema_dict['required'] = required_array
    c_schema_dict['additionalProperties'] = False
    c_schema_dict['uniqueProperties'] = True
    c_schema_dict['properties'] = dict_properties
    return c_schema_dict

def tm_schema_dict(type_str, minimum_value=None, maximum_value=None):
    c_schema_dict = OrderedDict()
    c_schema_dict['type'] = type_str
    if minimum_value is not None: c_schema_dict['minimum'] = minimum_value
    if maximum_value is not None: c_schema_dict['maximum'] = maximum_value
    return c_schema_dict

def val_schema_dict(value_dict, min_value_dict, uncentanty_dict, log_normal_dict):
    c_schema_dict = OrderedDict()
    c_schema_dict['value'] = value_dict
    c_schema_dict['min_value'] = min_value_dict
    c_schema_dict['uncentanty'] = uncentanty_dict
    c_schema_dict['log_normal_tf'] = log_normal_dict
    return c_schema_dict

def arr_schema_dict(type_arr, minItems=None):
    #~ type_arr = 'array' | ['array', 'null']
    inner_dict = OrderedDict()
    # inner_dict = {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number'}}
    inner_dict['type'] = 'array'
    inner_dict['minItems'] = 2
    inner_dict['maxItems'] = 2
    inner_dict['items'] = {'type': 'number'}
    c_schema_dict = OrderedDict()
    # c_schema_dict = {'type': ['array', 'null'], 'minItems': 8, 'items': inner_dict}
    c_schema_dict['type'] = type_arr
    if minItems: c_schema_dict['minItems'] = minItems
    c_schema_dict['items'] = inner_dict
    return c_schema_dict

# pos_val_schema_dict
#~ pos_val_schema_dict = {'type':'object', 'required': ['position', 'value'], 'properties': acc_out_dict}
pos_val_schema_dict = core_schema_dict(['position', 'value'], acc_out_dict)

# boundaries_schema_dict
boundaries_schema_inner_dict = {'horizontal': pos_val_schema_dict, 'vertical-left': pos_val_schema_dict, 'vertical-right': pos_val_schema_dict}
#~ boundaries_schema_dict = {'type':'object', 'required': transmitting_required, 'properties': boundaries_schema_inner_dict}
boundaries_schema_dict = core_schema_dict(transmitting_required, boundaries_schema_inner_dict)

# soil_schema_dict
#~ soil_schema_inner_dict = {'G_GMX': arr_schema_dict(['array', 'null'], minItems=8), 'XL': arr_schema_dict(['array', 'null'], minItems=8)}
soil_schema_inner_dict['G_GMX'] = arr_schema_dict(['array', 'null'], minItems=8)
soil_schema_inner_dict['XL'] = arr_schema_dict(['array', 'null'], minItems=8)
#~ soil_schema_dict = {'type':'object', 'required': ['G_GMX','XL'], 'properties': soil_schema_inner_dict}
soil_schema_dict = core_schema_dict(['G_GMX','XL'], soil_schema_inner_dict)

# model_G_GMX_XL_schema_dict
model_G_GMX_XL_schema_inner_dict['model_G_GMX_XL_tf'] = boolean_dict
model_G_GMX_XL_schema_inner_dict['depth_of_water_table_m'] = tm_schema_dict('number')
model_G_GMX_XL_schema_inner_dict['coef_of_lateral_earth_pressure_K0'] = tm_schema_dict('number', 0.1, 10)
model_G_GMX_XL_schema_inner_dict['mean_effective_stress_atm'] = tm_schema_dict(['number', 'null'], 0.05)
model_G_GMX_XL_schema_inner_dict['plasticity_index_PI_percent'] = tm_schema_dict('number', 0, 300)
model_G_GMX_XL_schema_inner_dict['over_consolidation_ratio_OCR'] = tm_schema_dict('number', 0.5, 100)
model_G_GMX_XL_schema_inner_dict['XL_uncentanty'] = boolean_dict
model_G_GMX_XL_schema_dict = core_schema_dict(['model_G_GMX_XL_tf','depth_of_water_table_m','coef_of_lateral_earth_pressure_K0','mean_effective_stress_atm','plasticity_index_PI_percent','over_consolidation_ratio_OCR', 'XL_uncentanty'], model_G_GMX_XL_schema_inner_dict)

# mesh_generation_dict
rearrange_nodes_inner_dict['rearrange_nodes_tf'] = boolean_dict
rearrange_nodes_inner_dict['iterations'] = tm_schema_dict('integer', 3, 100)
rearrange_nodes_inner_dict['scaling_factor'] = tm_schema_dict('number', 0.5, 1)
rearrange_nodes_inner_dict['vertical_deformation_coefficient'] = tm_schema_dict('number', 0.0, 1.0)
rearrange_nodes_inner_dict['horizontal_deformation_coefficient'] = tm_schema_dict('number', 0.0, 1.0)
rearrange_nodes_inner_dict['distance_threshold_m'] = tm_schema_dict('number', 10, 100)
rearrange_nodes_inner_dict['exponents'] = core_schema_dict(['inline', 'inline_extremities', 'orthogonal'], exp_dict)
rearrange_nodes_inner_dict['overall_exponent'] = tm_schema_dict(['number', 'null'], 0.01)
rearrange_nodes_dict = core_schema_dict(['rearrange_nodes_tf', 'iterations', 'scaling_factor', 'distance_threshold_m', 'vertical_deformation_coefficient', 'horizontal_deformation_coefficient', 'overall_exponent', 'exponents'], rearrange_nodes_inner_dict)
shape_factor_inner_dict['H/V'] = tm_schema_dict('number', 1, 10)
shape_factor_inner_dict['V/H'] = tm_schema_dict('number', 1, 10)
shape_factor_dict = core_schema_dict(['H/V', 'V/H'], shape_factor_inner_dict)
mesh_downsampling_inner_dict['mesh_downsampling_tf'] = boolean_dict
mesh_downsampling_inner_dict['minimum_module_length'] = tm_schema_dict('integer', 3, 50)
mesh_downsampling_inner_dict['element_shape_factor_multiplier'] = shape_factor_dict
mesh_downsampling_inner_dict['triangular_extremities_tf'] = boolean_dict
mesh_downsampling_dict = core_schema_dict(['mesh_downsampling_tf', 'minimum_module_length', 'element_shape_factor_multiplier', 'triangular_extremities_tf'], mesh_downsampling_inner_dict)
mesh_generation_inner_dict['maximum_frequency'] = tm_schema_dict('number', 0.5)
mesh_generation_inner_dict['maximum_element_shape_factor'] = shape_factor_dict
mesh_generation_inner_dict['uniform_initial_grid'] = boolean_dict
mesh_generation_inner_dict['rearrange_nodes'] = rearrange_nodes_dict
mesh_generation_inner_dict['boundaries_triangular_elements_tf'] = boolean_dict
mesh_generation_inner_dict['mesh_downsampling'] = mesh_downsampling_dict
mesh_generation_dict = core_schema_dict(['maximum_frequency', 'uniform_initial_grid', 'boundaries_triangular_elements_tf', 'rearrange_nodes', 'maximum_element_shape_factor', 'mesh_downsampling'], mesh_generation_inner_dict)

# DENS_schema_dict, GMX_schema_dict, G_schema_dict, XL_schema_dict
DENS_schema_inner_dict = \
val_schema_dict(tm_schema_dict('number', 1), tm_schema_dict(['number', 'null'], 1), tm_schema_dict(['number', 'null'], 0), boolean_dict)
PO_schema_inner_dict = \
val_schema_dict(tm_schema_dict('number', -1.0, 0.5), tm_schema_dict(['number', 'null'], -1.0, 0.5), tm_schema_dict(['number', 'null'], 0), boolean_dict)
GMX_schema_inner_dict = DENS_schema_inner_dict
XL_schema_inner_dict = \
val_schema_dict(tm_schema_dict(['number', 'null'], 0.0005, 1.0), tm_schema_dict(['number', 'null'], 0.0005, 1.0), tm_schema_dict(['number', 'null'], 0), boolean_dict)
DENS_schema_dict = core_schema_dict(['value', 'min_value', 'uncentanty', 'log_normal_tf'], DENS_schema_inner_dict)
PO_schema_dict = core_schema_dict(['value', 'min_value', 'uncentanty', 'log_normal_tf'], PO_schema_inner_dict)
GMX_schema_dict = core_schema_dict(['value', 'min_value', 'uncentanty', 'log_normal_tf'], GMX_schema_inner_dict)
G_schema_dict = core_schema_dict(['percent_of_GMX'], {'percent_of_GMX': tm_schema_dict('number', 10, 100)})
XL_schema_dict = core_schema_dict(['value', 'min_value', 'uncentanty', 'log_normal_tf'], XL_schema_inner_dict)

# schema_dict_modelling_properties
schema_dict_modelling_properties['job_title'] = {'type': 'string', 'minLength': 1, 'maxLength': 80}
schema_dict_modelling_properties['job_folder'] = {'type': 'string', 'minLength': 1, 'maxLength': 80, "pattern": "^[a-zA-Z0-9_]*$"}
schema_dict_modelling_properties['strata'] = tm_schema_dict('integer', 1)

arr_sd = arr_schema_dict(type_arr='array')
# ~ arr_sd['items'] = {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number'}}
np_schema_dict['type'] = 'object'
np_schema_dict['required'] = ['1', '2', '3', '4', '5', '6']
np_schema_dict['additionalProperties'] = False
np_schema_dict['uniqueProperties'] = True
np_schema_dict['patternProperties'] = {'^[0-9]{1,3}$' : arr_sd['items']}
schema_dict_modelling_properties['nodal_points'] = np_schema_dict

schema_dict_modelling_properties['half_space_VP_m_s'] = tm_schema_dict('number', 1, 10000)
schema_dict_modelling_properties['half_space_VS_m_s'] = tm_schema_dict('number', 1, 10000)
schema_dict_modelling_properties['half_space_RHO_N_m_3'] = tm_schema_dict('number', 1, 100000)
#~ schema_dict_modelling_properties['DENS_IDW'] = boolean_dict
#~ schema_dict_modelling_properties['GMX_IDW'] = boolean_dict
#~ schema_dict_modelling_properties['XL_IDW'] = boolean_dict
schema_dict_modelling_properties['NUMB_iteration'] = tm_schema_dict('integer', 1)

ground_nodes_inner_dict['type'] = ['array', 'null']
ground_nodes_inner_dict['items'] = {'type': 'number'}
ground_inside_inner_dict['ground'] = ground_nodes_inner_dict
ground_inside_inner_dict['inside'] = arr_schema_dict(['array', 'null'], minItems=None)
acceleration_output_inner_dict['X_direction'] = core_schema_dict(['ground','inside'], ground_inside_inner_dict)
acceleration_output_inner_dict['Y_direction'] = core_schema_dict(['ground','inside'], ground_inside_inner_dict)
acceleration_output_inner_dict['XY_direction'] = core_schema_dict(['ground','inside'], ground_inside_inner_dict)
acceleration_output_dict = core_schema_dict(['X_direction', 'Y_direction', 'XY_direction'], acceleration_output_inner_dict)
schema_dict_modelling_properties['acceleration_output'] = acceleration_output_dict
schema_dict_modelling_properties['boundary_conditions'] = boundaries_schema_dict

# schema_dict_plot_properties
schema_dict_plot_properties['BW_scale_tf'] = boolean_dict
schema_dict_plot_properties['profile_SA_max_width'] = tm_schema_dict('number', 0.1)
schema_dict_plot_properties['profile_SA_BW_scale_tf'] = boolean_dict
schema_dict_plot_properties['profile_ACC_max_width'] = tm_schema_dict('number', 0.1)
schema_dict_plot_properties['profile_FFT_lower_limit_Hz'] = tm_schema_dict('number', 0.001, 0.3)

scale_inner_dict['line_thicknesses'] = tm_schema_dict('number', 0.01, 100)
scale_inner_dict['fonts'] = tm_schema_dict('number', 0.01, 100)
scale_inner_dict['symbols'] = tm_schema_dict('number', 0.01, 100)
scale_dict = core_schema_dict(['line_thicknesses', 'fonts', 'symbols'], scale_inner_dict)

schema_dict_plot_properties['scale'] = scale_dict
schema_dict_plot_properties['save_PNG_tf'] = boolean_dict

# schema_dict_waveforms_properties
schema_dict_waveforms_properties['N1EQ_first_time_step_for_last_iteration'] = tm_schema_dict('integer', 1)
schema_dict_waveforms_properties['KGEQ_last_time_step_for_last_iteration'] = tm_schema_dict('integer', 1)
schema_dict_waveforms_properties['N2EQ_first_time_step_for_first_iterations'] = tm_schema_dict('integer', 1)
schema_dict_waveforms_properties['N3EQ_last_time_step_for_first_iterations'] = tm_schema_dict('integer', 1)
schema_dict_waveforms_properties['DTEQ_time_step_s'] = tm_schema_dict('number', 0.01, 1.0)
schema_dict_waveforms_properties['PRINPUT_period_corresponding_to_maximum_spectral_acceleration_of_horizontal_input_motion_s'] = tm_schema_dict(['number', 'null'], 0.001, 1.0)
schema_dict_waveforms_properties['HDRX_header_lines_in_horizontal_input_time_history'] = tm_schema_dict('integer', 0)
schema_dict_waveforms_properties['HDRY_header_lines_in_vertical_input_time_history'] = tm_schema_dict('integer', 0)
schema_dict_waveforms_properties['multiplier_factor_to_g'] = tm_schema_dict('number', 0)

# schema_dict_stratum_properties
schema_dict_stratum_properties['name'] = {'type': 'string', 'minLength': 1, 'maxLength': 7}
schema_dict_stratum_properties['nodes'] = {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 3}
schema_dict_stratum_properties['DENS_N_m_3'] = DENS_schema_dict
schema_dict_stratum_properties['PO_ratio'] = PO_schema_dict
schema_dict_stratum_properties['GMX_kN_m_3'] = GMX_schema_dict
schema_dict_stratum_properties['G_kN_m_3'] = G_schema_dict
schema_dict_stratum_properties['XL_decimal'] = XL_schema_dict
schema_dict_stratum_properties['soil_TYPE'] = soil_schema_dict
schema_dict_stratum_properties['model_G_GMX_XL'] = model_G_GMX_XL_schema_dict

#schema_dict_modelling
schema_dict_modelling = \
core_schema_dict(modelling_required, schema_dict_modelling_properties)

#schema_dict_plot
schema_dict_plot = \
core_schema_dict(plot_required, schema_dict_plot_properties)

#schema_dict_stratum
schema_dict_stratum = \
core_schema_dict(stratum_required, schema_dict_stratum_properties)

#schema_dict_waveforms
schema_dict_waveforms = \
core_schema_dict(waveforms_required, schema_dict_waveforms_properties)

#schema_dict_properties
schema_dict_properties['modelling_parameters'] = schema_dict_modelling
schema_dict_properties['mesh_generation'] = mesh_generation_dict
schema_dict_properties['waveforms'] = schema_dict_waveforms
schema_dict_properties['plot_parameters'] = schema_dict_plot

# schema_dict
schema_dict['$schema'] = 'http://json-schema.org/draft-04/schema#'
schema_dict['type'] = 'object'
schema_dict['required'] = ['modelling_parameters', 'mesh_generation', 'waveforms', 'stratum001', 'stratum002', 'plot_parameters']
schema_dict['additionalProperties'] = False
schema_dict['uniqueProperties'] = True
schema_dict['properties'] = schema_dict_properties
schema_dict['patternProperties'] = {'^stratum[0-9]{3}$' : schema_dict_stratum}

schema_json = to_json(schema_dict)
with open(opts.json, 'w') as h_schema_json:
    h_schema_json.write(schema_json + '\n')

print opts.json + " has been updated"

# ######################
# jsonschema first-tests
# ######################

#~ import jsonschema
#~ dict_in = {'horizontal': {'position': 23.4, 'value': 3}}
#~ acc_out_inner_dict = { 'type': 'integer', 'minimum': 0, 'maximum': 4 }
#~ acc_out_dict = {'position': {'type': 'number'}, 'value': acc_out_inner_dict}
#~ pos_val_schema_dict = {'type':'object', 'required': ['position', 'value'], 'properties': acc_out_dict}
#~ schema_dict = {'type':'object', 'required': ['horizontal'], 'properties': {'horizontal': pos_val_schema_dict}}
#~ jsonschema.validate(instance=dict_in,schema=schema_dict)

#~ dict_in = {'G_GMX': G_GMX_vs_shear_strain_3, 'XL': XL_vs_shear_strain_3}
#~ schema_dict = {'type':'object', 'required': ['G_GMX','XL'], 'properties': {'G_GMX': arr_schema_dict(['array', 'null'], minItems=8), 'XL': arr_schema_dict(['array', 'null'], minItems=8)}}
#~ jsonschema.validate(instance=dict_in,schema=schema_dict)

# ######################
# jsonschema development
# ######################

#~ dict_schema_modelling = {'type':'object', \
#~ 'required': ['strata', 'maximum_frequency', 'NUMB_iteration', 'boundary_conditions', 'half_space_VS_m_s', 'half_space_VP_m_s', 'half_space_RHO_N_m_3', 'closest_nodes_XY_direction_acceleration_output', 'closest_nodes_X_direction_acceleration_output', 'closest_nodes_Y_direction_acceleration_output', 'GMX_IDW', 'XL_IDW', 'DENS_IDW'], \
#~ 'properties': {\
#~ 'strata': {'type': 'integer', 'minimum': 1}, \
#~ 'maximum_frequency': {'type': 'number', 'minimum': 0.1}, \
#~ 'half_space_VP_m_s': {'type': 'number', 'minimum': 1}, \
#~ 'half_space_VS_m_s': {'type': 'number', 'minimum': 1}, \
#~ 'half_space_RHO_N_m_3': {'type': 'number', 'minimum': 1}, \
#~ 'DENS_IDW': {'type': 'boolean'}, \
#~ 'GMX_IDW': {'type': 'boolean'}, \
#~ 'XL_IDW': {'type': 'boolean'}, \
#~ 'NUMB_iteration': {'type': 'integer', 'minimum': 1}, \
#~ 'closest_nodes_X_direction_acceleration_output': {'type': ['array', 'null'], 'items': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number'}}}, \
#~ 'closest_nodes_Y_direction_acceleration_output': {'type': ['array', 'null'], 'items': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number'}}}, \
#~ 'closest_nodes_XY_direction_acceleration_output': {'type': ['array', 'null'], 'items': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number'}}}, \
#~ 'boundary_conditions': {'type':'object', 'required': ['horizontal','vertical-left','vertical-right'], 'properties': {'horizontal': {'type':'object', 'required': ['position', 'value'], 'properties': {'position': {'type': 'number'}, 'value': { 'type': 'integer', 'minimum': 0, 'maximum': 4 }}}, 'vertical-left': {'type':'object', 'required': ['position', 'value'], 'properties': {'position': {'type': 'number'}, 'value': { 'type': 'integer', 'minimum': 0, 'maximum': 4 }}}, 'vertical-right': {'type':'object', 'required': ['position', 'value'], 'properties': {'position': {'type': 'number'}, 'value': { 'type': 'integer', 'minimum': 0, 'maximum': 4 }}}} \
#~ }}, \
#~ 'additionalProperties': False, 'uniqueProperties': True}

#~ jsonschema.validate(instance=modelling_parameters,schema=dict_schema_modelling)

#~ dict_schema_stratum = {'type': 'object', \
#~ 'required': ['name', 'nodes', 'soil_TYPE', 'DENS_N_m_3', 'GMX_kN_m_3', 'G_kN_m_3', 'XL_decimal'], \
#~ 'properties': {\
#~ 'name': {'type': 'string', 'minLength': 1}, \
#~ 'nodes': {'type': 'array', 'items': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number'}}}, \
#~ 'soil_TYPE': {'type':'object', 'required': ['G_GMX','XL'], 'properties': {'G_GMX': {'type': ['array', 'null'], 'minItems': 8, 'items': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number'}}}, 'XL': {'type': ['array', 'null'], 'minItems': 8, 'items': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number'}}}}}, \
#~ 'DENS_N_m_3': {'type':'object', 'required': ['value', 'uncentanty'], 'properties': {'value': { 'type': 'number', 'minimum': 1}, 'uncentanty': {'type': 'number', 'minimum': 0}}}, \
#~ 'GMX_kN_m_3': {'type':'object', 'required': ['value', 'uncentanty'], 'properties': {'value': { 'type': 'number', 'minimum': 1}, 'uncentanty': {'type': 'number', 'minimum': 0}}}, \
#~ 'G_kN_m_3': {'type':'object', 'required': ['percent_of_GMX'], 'properties': {'percent_of_GMX': { 'type': 'number', 'minimum': 1, 'maximum': 100}}}, \
#~ 'XL_decimal': {'type':'object', 'required': ['value', 'uncentanty'], 'properties': {'value': { 'type': 'number', 'minimum': 0.0005}, 'uncentanty': {'type': 'number', 'minimum': 0}}}, \
#~ }, \
#~ 'additionalProperties': False, 'uniqueProperties': True}

#~ jsonschema.validate(instance=stratum001,schema=dict_schema_stratum)

#~ dict_schema = {'$schema': 'http://json-schema.org/draft-04/schema#', 'type': 'object', 'required': ['modelling_parameters', 'stratum001', 'stratum002'], 'properties': {'modelling_parameters': dict_schema_modelling},'patternProperties': {'^stratum[0-9]{3}$' : dict_schema_stratum}, 'additionalProperties': False, 'uniqueProperties': True}

#~ jsonschema.validate(instance=input_dict,schema=dict_schema)

