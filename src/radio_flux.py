# Derive radio luminosity and calculate the radio loudness

import numpy as np
from scipy.optimize import curve_fit

import ast

from astropy.cosmology import FlatLambdaCDM

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_and_save_spectrum(frequency, flux, predict, output_filename='radio_spectrum.png'):
    """
    Plot a radio spectrum from given frequency and flux data.
    Fit a curve to the data and save the figure.

    Parameters:
    - frequency (numpy.ndarray): Array of frequency values.
    - flux (numpy.ndarray): Array of flux values.
    - output_filename (str): Output filename for the saved figure.

    Returns:
    - None
    """
    # Plot scatter plot
    plt.scatter(frequency, flux, label='obs. data')
    plt.scatter(predict[0], predict[1], label='predicted at {} GHz'.format(predict[0]))
    
    plt.annotate(f'({predict[0]:.2f}, {predict[1]:.2f})', (predict[0], predict[1]), textcoords="offset points", xytext=(0,10), ha='center')
    for i, (x, y) in enumerate(zip(frequency, flux)):
            plt.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    # Fit a curve
    def fitted_curve(x, a, b):
        return b * x**a

    params, covariance = curve_fit(fitted_curve, frequency, flux)
    a, b = params

    # Plot the fitted curve
    x_fit = np.linspace(np.minimum(np.min(frequency), predict[0]), np.maximum(np.max(frequency), [predict[0]]), 100)
    y_fit = fitted_curve(x_fit, a, b)
    plt.plot(x_fit, y_fit, label='{:.2f}*x**{:.2f}'.format(b,a), color='red')

    # Set plot parameters and save the figure
    plt.xlabel('Restframe Frequency')
    plt.ylabel('Flux')
    plt.legend()
    plt.title('Radio Spectrum')
    plt.savefig(output_filename)
    #plt.show()

def calc_pred_flux(frequency, flux, f_pred, slope=None):
    """
    Fit a power-law relation y = b * x^a to frequency and flux data and
    predict the flux at x = 10.

    Args:
    - frequency (list): List of frequency values.
    - flux (list): List of flux values.
    - slope (float, optional): The value of the slope (a) to use for fitting.
      If None, it will be determined from the data.

    Returns:
    - float: The predicted flux at x = 10.
    """

    # # Convert input lists to NumPy arrays
    # frequency = np.array(frequency)
    # flux = np.array(flux)

    # If slope is not provided, perform linear regression to find it
    if len(flux)>1:
        def power_law(x, a, b):
            return b * x**a

        # Perform the curve fit
        params, _ = curve_fit(power_law, frequency, flux)
        a, b = params
    else:
        # Use the provided slope value
        b = flux[0] / frequency[0]**slope
        a = slope
        

    # Predict flux at x = 10 using the fitted parameters
    x_predict = f_pred
    y_predict = b * x_predict**a
    
    plot_and_save_spectrum(frequency, flux, [x_predict, y_predict], output_filename='radio_spectrum.png')

    return y_predict



def main():   
    # Prompt a message
    user_input = input("[INFO]Enter redshift, predicting frequency(GHz, in restframe), slope of the radio spectrum; seperated by space:\n")

    # Split the input by space and assign to a variable
    values = user_input.split()
    redshift=float(values[0])
    f_pred=float(values[1])
    slope_value=values[2]
    if slope_value!='None':
        slope_value=float(values[2])
    
    frequency_data = []
    flux_data = []

    while True:
        user_input = input("[INFO]Enter space-separated values in float for freqs (GHz, in restframe)] and fluxes (in muJy) (after each datapoint inputted press enter, input 'y' to stop):\n")

        if user_input.lower() == 'y':
            break

        elements = user_input.split()

        try:
            # Assuming the input has at least two elements (for both arrays)
            frequency_data.append(float(elements[0]))
            flux_data.append(float(elements[1]))
        except (ValueError, IndexError):
            print("Invalid input. Please enter at least two space-separated values in float.")

    # Display the assigned values
    print("[INFO]Values entered:\nRedshift:{}\nPredicting flux at:{} GHz\nDatapoints at restframe:\nFreqs:{} GHz\nFlux:{} muJy\nSlope (if only one datapoint):{}".format(redshift, f_pred, frequency_data, flux_data, slope_value))

    # Example usage:
    #slope_value = -0.98  # Set to None if you want to calculate the slope from data

    predicted_flux = calc_pred_flux(np.array(frequency_data)*(1+redshift), np.array(flux_data), f_pred, slope=slope_value)
    print(f"Predicted flux at 5 GHz (restframe, Jy): {predicted_flux}")

if __name__ == '__main__':
    main()

