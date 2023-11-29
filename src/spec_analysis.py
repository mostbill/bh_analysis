# Read into Sculptor models and extract line profiles
import argparse
import corner
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM

from sculptor import specfit as scfit
from sculptor import specmodel as scmod
from astropy import units
from sculptor import speconed as sod

from IPython import embed

import numpy as np

# Import the SpecAnalysis and Cosmology modules
from sculptor import specanalysis as scana

def print_fit_report(fit):
    '''
    Generate fit report for every fit
    '''
    for element in fit:
        if hasattr(element, 'name'):
            filename = f"{element.name}_fit_report.txt"
            with open(filename, 'w') as file:
                file.write(str(element.fit_result.fit_report()))
            print("[INFO]Fit report of {0} saved to {1}".format(element.name, filename))
        else:
            print("[WARNING]Object does not have a 'name' attribute.")

def analyze_line_profile(line_fit):
    '''
    Analyze line profile with given wavelength (which line is this) and the source's redshift
    Prefix determines what Sculptor fit component you want to integrate into this analysis
    '''
    # Prompt a message
    user_input = input("[INFO]Enter redshift, wavelength(lab), line prefix (e.g. ['CIV_A_','CIV_B_']), seperated by space:\n")

    # Split the input by space and assign to a variable
    values = user_input.split()
    redshift=values[0]
    line_wav=values[1]
    prefix=values[2]

    # Display the assigned values
    print("[INFO]Values entered:\nRedshift:{}\nLine Wavelength(lab):{}\nLine prefix:{}".format(redshift, line_wav, prefix))

    line_spec = scana.build_model_flux(line_fit, prefix)
    line_spec.plot(ymin=-0.1, ymax=1.0)
    print("[DEBUG]Turns out it worked!")
    # civ_peak_fluxden = np.max(civ_spec.fluxden)*civ_spec.fluxden_unit
    # print('CIV peak flux density: {:.2e}'.format(civ_peak_fluxden))

    # civ_flux = scana.get_integrated_flux(civ_spec)
    # print('CIV integrated flux: {:.2e}'.format(civ_flux))

    # civ_z = scana.get_peak_redshift(civ_spec, 1549.06)
    # print('CIV peak redshift: {:.3f}'.format(civ_z))

    # civ_shift = ((civ_z-6.442)*3*10**5)/(1+6.442)
    # print('CIV line shift: {:.3f}'.format(civ_shift))

    # # Calculate the CIV FWHM
    # civ_fwhm = scana.get_fwhm(civ_spec)
    # print('CIV FWHM: {:.2f}'.format(civ_fwhm))

    # # Calculate the CIV FWHM, taking into account a spectral resolution of R=1000
    # civ_fwhm = scana.get_fwhm(civ_spec, resolution=600)
    # print('CIV FWHM (accounting for spectral resolution): {:.2f}'.format(civ_fwhm))

def analyze_continuum(fit, cont_wave=4400):
    '''
    Analyze continuum with given wavelength (where at you want to analyze on the continuum) and the source's redshift
    '''

    dispersion = np.linspace(0, 100000, 1000000)

    # Use the continuum analysis function
    cont_result = scana.analyze_continuum(fit, ['PLBC_'],
                                        [cont_wave],
                                        cosmo, redshift=fit.redshift, width=10, dispersion=dispersion)

    # Print the results from the dictionary
    filename = f"Continuum_{cont_wave}AA.txt"
    with open(filename, 'w') as file:
        for key in cont_result.keys():
            result_string = '{} = {:.2e}'.format(key, cont_result[key])
            file.write(result_string + '\n')

        
    print("[INFO]Continuum analysis at {0} is done and saved into {1}".format(cont_wave, filename))
    
    return 0

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add an argument for the 'ra' variant
    parser.add_argument('-fit', type=str, help='Fit folder from Sculptor')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the value of the 'ra' variant
    fit_folder = args.fit
    
    # Define Cosmology for cosmological conversions
    global cosmo
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # Instantiate an empty SpecFit object
    fit = scfit.SpecFit()
    # Load the example spectrum fit
    fit.load(fit_folder)   
    
    print_fit_report(fit.specmodels)
    
    # needs import to assign the profile names
    #analyze_line_profile(fit)
    analyze_continuum(fit, cont_wave=4400)
    

    # for idx, specmodel in enumerate(fit.specmodels):
    #     print(idx, specmodel.name)

if __name__ == '__main__':
    main()