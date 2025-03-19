#!/usr/bin/env python

import json
import math
import numpy as np
from os import path as os_path
from sys import argv as sys_argv

from sys import path as sys_path
sys_path.append(os_path.join(*[os_path.dirname(sys_argv[0]), '..', '..', 'lib']))
from def_QUAD4M import is_valid_file
from def_QUAD4M import common_def

import argparse

common_def_dict = common_def()

p = argparse.ArgumentParser(description='description: rotate input waveforms')
p.add_argument("json", action="store", type=is_valid_file, help=common_def_dict['json'] + ', togheter with waveform parameters')
p.add_argument("uacc", action="store", type=is_valid_file, help="single-column horizontal acceleration time-history along the x-axis (e.g. the E component)")
p.add_argument("vacc", action="store", type=is_valid_file, help="single-column horizontal acceleration time-history along the y-axis (e.g. the N component)")
p.add_argument("deg", action="store", type=float, help="rotation angle in degree from y-axis in clockwise direction")
opts = p.parse_args()

infile = opts.json
inwaveE = opts.uacc
inwaveN = opts.vacc
proj_angle = opts.deg

schemafile = os_path.join(*[os_path.dirname(sys_argv[0]), '..', '..', 'lib', 'schema.json'])
elemfile = os_path.join(*[os_path.dirname(sys_argv[0]), '..', '..', 'lib', 'json_elements.json'])
with open(elemfile, 'r') as c_file: json_elem_dict = json.load(c_file)

# JSON-input
with open(infile, 'r') as input_json: input_dict = json.load(input_json)
# JSON-schema
with open(schemafile, 'r') as schema_json: dict_schema = json.load(schema_json)

wave_hrx = int(input_dict[json_elem_dict["wave"]][json_elem_dict["wave_hrx"]])

hdrx_lines = []
c_fx = open(inwaveN, 'r')
for kk in range(wave_hrx): hdrx_lines.append(c_fx.readline())
c_fx.close()

waveE = np.loadtxt(fname=inwaveE, skiprows=wave_hrx)
waveN = np.loadtxt(fname=inwaveN, skiprows=wave_hrx)

theta = math.radians(proj_angle)
waveX = waveN*math.cos(theta) + waveE*math.sin(theta)

WF_format = '%-.6f'
cmp0 = inwaveN + '.xacc'
cmp1 = inwaveN + '.xacc'
for ii in range(wave_hrx):
    if hdrx_lines[ii][:6] == 'STREAM':
        cmp0 = '.' + hdrx_lines[ii][-4:-1] + '.'
        cmp1 = '.' + hdrx_lines[ii][-4:-2] + 'X.'

inwaveX = inwaveN.replace(cmp0,cmp1) + '.xacc'

print 'max: ' + str(max(waveE)) + ' min: ' + str(min(waveE)) + ' (' + os_path.basename(inwaveE) + ')'
print 'max: ' + str(max(waveN)) + ' min: ' + str(min(waveN)) + ' (' + os_path.basename(inwaveN) + ')'
print 'max: ' + str(max(waveX)) + ' min: ' + str(min(waveX)) + ' (' + os_path.basename(inwaveX) + ')'

WFX_fname = inwaveX
WFX_fid = open(WFX_fname,'w')
for ii in range(wave_hrx):
    if hdrx_lines[ii][:6] == 'STREAM':
        cmp0 = hdrx_lines[ii][-4:-1] + ' STREAM'
        WFX_fid.write(hdrx_lines[ii][:-2] + 'X\n')
    elif hdrx_lines[ii][:5] == 'USER5':
        WFX_fid.write('USER5: PROJECTED BY ' + '{:.1f}'.format(proj_angle) + ' DEGREE WITH RESPECT TO ' + cmp0 + '\n')
    else: WFX_fid.write(hdrx_lines[ii])
np.savetxt(WFX_fid, waveX, fmt=WF_format)
WFX_fid.write('\n')
WFX_fid.close()
