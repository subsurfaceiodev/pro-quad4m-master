#!/usr/bin/env python

import json
import numpy as np
from os import path as os_path
from sys import argv as sys_argv

# from sys import path as sys_path
# sys_path.append(os_path.join(*[os_path.dirname(sys_argv[0]), '..', '..', 'lib']))
from lib.def_QUAD4M import (
    resp_spectra_acc,
    Fourier_spectra_amplitude,
    KonnoOhmachi,
    SP_resample,
    is_valid_file,
    common_def,
)

import argparse

common_def_dict = common_def()

p = argparse.ArgumentParser(
    description='description: generate corresponding spectra (displacement responses (5% damped), non-smoothed and smoothed Fourier amplitudes) to the input waveform. Smoothing is performed using the Konno and Ohmachi operator with b=40.')
p.add_argument("json", action="store", type=is_valid_file,
               help=common_def_dict['json'] + ', togheter with waveform parameters')
p.add_argument("acc", action="store", type=is_valid_file,
               help="single-column acceleration time-history, whose parameters are contained in 'json' file above")
opts = p.parse_args()

infile = opts.json
inwaveX = opts.acc
schemafile = os_path.join(*[os_path.dirname(sys_argv[0]), '..', '..', 'lib', 'schema.json'])
elemfile = os_path.join(*[os_path.dirname(sys_argv[0]), '..', '..', 'lib', 'json_elements.json'])
with open(elemfile, 'r') as c_file: json_elem_dict = json.load(c_file)

# JSON-input
with open(infile, 'r') as input_json: input_dict = json.load(input_json)
# JSON-schema
with open(schemafile, 'r') as schema_json: dict_schema = json.load(schema_json)

wave_hrx = int(input_dict[json_elem_dict["wave"]][json_elem_dict["wave_hrx"]])
waveX = np.loadtxt(fname=inwaveX, skiprows=wave_hrx)
hdrx_lines = []
c_fx = open(inwaveX, 'r')
for kk in range(wave_hrx): hdrx_lines.append(c_fx.readline())
c_fx.close()

wave_mpf = float(input_dict[json_elem_dict["wave"]][json_elem_dict["wave_mpf"]])
waveX = waveX * wave_mpf  # conversion to g

nu = 50
K_nu = 100
uti_g = 980.6  # [cm/s^2]
uti_gm = uti_g / 100.0  # [m/s^2]
c_gm = float(input_dict[json_elem_dict["pltp"]][json_elem_dict["pltp_fft"]])
c_fm = max(12., float(input_dict[json_elem_dict["mesh"]][json_elem_dict["mesh_mfe"]]) * 2.0)
dt = float(input_dict[json_elem_dict["wave"]][json_elem_dict["wave_tss"]])

wave_A, freq_A = Fourier_spectra_amplitude(dt, waveX)
K_SP_A = KonnoOhmachi(wave_A, freq_A, b=40.0)
wave_K, freq_K = SP_resample(K_SP_A, freq_A, min_freq=c_gm, max_freq=c_fm)

mat_SA_A, mat_SD_A, T = resp_spectra_acc(dt, waveX * uti_g, nu=nu)

WF_format = '%16.8f'

WFX_fname = inwaveX + '.fft.txt'
WFX_fid = open(WFX_fname, 'w')
for ii in range(wave_hrx): WFX_fid.write(hdrx_lines[ii])
WFX_fid.write('# -------- FOURIER SPECTRA [Hz]-[g] ----------\n')
np.savetxt(WFX_fid, np.column_stack([freq_A, wave_A]), fmt=WF_format)
WFX_fid.write('\n')
WFX_fid.close()

WFX_fname = inwaveX + '.fft.ko.txt'
WFX_fid = open(WFX_fname, 'w')
for ii in range(wave_hrx): WFX_fid.write(hdrx_lines[ii])
WFX_fid.write('# -------- SMOOTHED FOURIER SPECTRA [Hz]-[g] ----------\n')
np.savetxt(WFX_fid, np.column_stack([freq_K, wave_K]), fmt=WF_format)
WFX_fid.write('\n')
WFX_fid.close()

WFX_fname = inwaveX + '.sd.txt'
WFX_fid = open(WFX_fname, 'w')
for ii in range(wave_hrx): WFX_fid.write(hdrx_lines[ii])
WFX_fid.write('# -------- DISPLACEMENT RESPONSE SPECTRA [s]-[cm] ----------\n')
np.savetxt(WFX_fid, np.column_stack([T, mat_SD_A]), fmt=WF_format)
WFX_fid.write('\n')
WFX_fid.close()
