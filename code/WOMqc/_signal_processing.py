# Importing Libraries
import numpy as np
import pandas as pd

from scipy.special import erfc
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve
#from moku.instruments import Oscilloscope


'''
Contains numerical and analytical functions for waveform baseline correction, 
integration, pulse shape models, and statistical grouping.'''
'''
├── riemann_sum_peak()
├── gaussian()
├── EMG()
├── correct_baseline_min()
├── gain()
├── create_baseline_data()
├── grouped_mean_std()
├── calculate_eventwise_ratios()
├── apply_sipm_corrections()
'''
'''
Single-position measurement
'''
"""
Integrate a waveform around its peak using a peak-centered window.
The function locates the maximum absolute signal amplitude and
defines an asymmetric integration window around it:
    - int_window_min before the peak
    - int_window_max after the peak
Workflow:
    1. Compute time step from acquisition settings
    2. Find peak index (maximum absolute amplitude)
    3. Convert peak index to physical time
    4. Define integration window around peak
    5. Convert window limits back to index range
    6. Integrate waveform using Riemann sum
Returns:
    tuple:
        start_idx : int
        end_idx   : int
        area      : float
"""
def riemann_sum_peak(self, data, int_window_min, int_window_max):
    """
    Integrate a waveform around its maximum (peak-centered).
    """
    #calculate the stepwidth in seconds
    dx = (self.rec_time_max - self.rec_time_min) / self.frame_length_max
    peak_idx = np.argmax(np.abs(data))  # find peak
    #Calculate the time of the maximum
    peak_time = self.rec_time_min + peak_idx * dx
    #Calculate the integration window relativ to the peak
    start_time = peak_time - int_window_min
    end_time = peak_time + int_window_max
    #Calculate the index where the integration should start and end
    start_idx = max(0, round((start_time - self.rec_time_min) / dx))
    end_idx = min(self.frame_length_max, round((end_time - self.rec_time_min) / dx))
    #Sum up the bins within the integration window
    area = np.sum(data[start_idx:end_idx]) * dx
    return start_idx, end_idx, area    

def riemann_sum_peak_vectorized(
    self,
    data,
    int_window_min,
    int_window_max
):
    dx = (
        self.rec_time_max - self.rec_time_min
    ) / self.frame_length_max
    # Find peak for every waveform
    peak_idx = np.argmax(
        np.abs(data),
        axis=-1
    )
    # Peak time
    peak_time = (
        self.rec_time_min
        + peak_idx * dx
    )
    # Integration window
    start_time = peak_time - int_window_min
    end_time = peak_time + int_window_max
    start_idx = np.maximum(
        0,
        np.round(
            (start_time - self.rec_time_min) / dx
        ).astype(int)
    )
    end_idx = np.minimum(
        self.frame_length_max,
        np.round(
            (end_time - self.rec_time_min) / dx
        ).astype(int)
    )
    # ----------------------------------------
    # Integrate each waveform
    # ----------------------------------------
    areas = np.zeros(
        data.shape[:-1],
        dtype=np.float32
    )
    for index in np.ndindex(data.shape[:-1]):
        areas[index] = (
            np.sum(
                data[
                    index + (
                        slice(
                            start_idx[index],
                            end_idx[index]
                        ),
                    )
                ]
            ) * dx
        )
    return areas

"""
Gaussian model for waveform shape.
Evaluates a standard Gaussian pulse:
    height * exp(-0.5 * ((x - mean)/sigma)^2)
Returns:
    np.ndarray or float
"""
@staticmethod
def gaussian(x, mean, height, sigma):
    return height * np.exp(-0.5 * ((x - mean) / sigma)**2)    
"""
Exponentially modified Gaussian (EMG) waveform model.
Models asymmetric detector pulses as a convolution of:
    - Gaussian (detector response)
    - Exponential decay (physical process)
Parameters:
    x     : array-like
    mean  : float
    height: float
    sigma : float (Gaussian width)
    tau   : float (exponential decay constant)
Returns:
    np.ndarray
"""


@staticmethod
def EMG(x, mean, height, sigma, tau): #Exponential modified Gaussian distribution
    res1 = height * np.exp((sigma**2 - 2 * tau * (x - mean)) / (2 * tau**2))# (2*np.abs(tau)) # 1/(2 * tau) *
    res2 = erfc((sigma**2 - tau * (x - mean)) / (np.sqrt(2) * sigma * np.abs(tau)))
    return res1 * res2
'''
Baseline correction using minimum sum in range for correction.
    Corrects the baseline (DC offset) of all waveforms.
    Searches for the minimum of the sum of bins in a window of length window[0] in range window[1]-window[2]
    Make sure the search range is shortly before the triggered signal is expected to arrive.
    Takes:
        waveform, 
        window=(integration window length, start index, end index), 
        sigma= 0 (if > 0, applies smoothing before baseline correction),
        smooth_method= 0 (running average), 2 (gaussian)
    returns:
    corrected, baseline_value, min_index
'''


@staticmethod
def correct_baseline_min_old(waveform, window, sigma, smooth_method):
    wf = np.asarray(waveform, dtype=np.float32)
    n = len(wf)
    # --- Optional smoothing ---
    if sigma > 0:
        if smooth_method == 0:
            kernel = np.ones(int(2 * sigma + 1)) / (2 * sigma + 1)
            wf = np.convolve(wf, kernel, mode='same')
        elif smooth_method == 2:
            wf = gaussian_filter1d(wf, sigma=sigma)
    # --- Window params ---
    length, start, end = window
    start = max(0, int(start))
    end = min(n - length, int(end))
    # 🚀 Vectorized moving average
    kernel = np.ones(length) / length
    means_full = np.convolve(wf, kernel, mode='valid')
    # restrict to search region
    means = means_full[start:end]
    # find minimum
    min_index_local = np.argmin(np.abs(means))
    min_index = start + min_index_local
    baseline_value = means[min_index_local]
    corrected = wf - baseline_value
    return corrected, baseline_value, min_index

@staticmethod
def correct_baseline_min(waveforms, window,sigma, smooth_method):
    wf = np.asarray(waveforms, dtype=np.float32)
    if wf.ndim != 3:
        raise ValueError(
            f"Expected (n, m, samples), got {wf.shape}"
        )
    n_samples = wf.shape[-1]

    # --------------------------------------------------
    # Optional smoothing
    # --------------------------------------------------
    if sigma > 0:
        if smooth_method == 0:
            kernel_size = int(2 * sigma + 1)
            kernel = (
                np.ones(kernel_size, dtype=np.float32)
                / kernel_size
            )
            wf = convolve(
                wf,
                kernel.reshape(1, 1, -1),
                mode="same"
            )
        elif smooth_method == 2:
            wf = gaussian_filter1d(
                wf,
                sigma=sigma,
                axis=-1
            )
        else:
            raise ValueError(
                f"Unknown smooth_method: {smooth_method}"
            )
    # --------------------------------------------------
    # Baseline window
    # --------------------------------------------------
    length, start, end = window
    length = int(length)
    start = max(0, int(start))
    end = min(n_samples - length, int(end))
    if end <= start:
        raise ValueError(
            f"Invalid baseline window: {window}"
        )

    # --------------------------------------------------
    # Moving average
    # --------------------------------------------------
    cumsum = np.cumsum(wf, axis=-1, dtype=np.float32)

    means_full = (
        cumsum[..., length:]
        - cumsum[..., :-length]
    ) / length

    means = means_full[..., start:end]
    # --------------------------------------------------
    # Find minimum absolute baseline
    # --------------------------------------------------
    min_index_local = np.argmin(
        np.abs(means),
        axis=-1
    )
    baseline_value = np.take_along_axis(
        means,
        min_index_local[..., None],
        axis=-1
    )[..., 0]

    min_index = start + min_index_local

    # --------------------------------------------------
    # Baseline correction
    # --------------------------------------------------
    corrected = wf - baseline_value[..., None]

    return corrected, baseline_value, min_index


@staticmethod
def correct_baseline_min_not_vectorized(waveform, window=(20, 10, 90), sigma=0, smooth_method=2):
    """
    Baseline correction using minimum-sum search over a moving window.
    """
    wf = np.array(waveform, dtype=np.float32)
    # Converts the input waveform to a float numpy array.
    n = len(wf)
    # Gets the length of the waveform.
    # --- Optional smoothing ---
    if sigma > 0:
        if smooth_method == 0:
            # Simple running average (boxcar)
            kernel = np.ones(int(2 * sigma + 1)) / (2 * sigma + 1)
            # Creates a boxcar kernel of width 2*sigma+1.
            wf = np.convolve(wf, kernel, mode='same')
            # Applies running average smoothing.
        elif smooth_method == 2:
            wf = gaussian_filter1d(wf, sigma=sigma)
            # Applies Gaussian smoothing.
    # --- Extract window parameters ---
    length, start, end = window
    # Unpacks window parameters: averaging length, start, end.
    start = max(0, int(start))
    # Ensures start index is not negative.
    end = min(n - length, int(end))
    # Ensures end index does not exceed waveform length.
    # --- Search for the minimum average ---
    means = np.array([
        np.mean(wf[i:i + length])
        for i in range(start, end)
    ])
    # For each window position, computes the mean value.
    min_index_local = np.argmin(np.abs(means))
    # Finds the index of the minimum mean value.
    min_index = start + min_index_local
    # Converts local index to global index in waveform.
    # --- Determine baseline value ---
    baseline_value = means[min_index_local]
    # Gets the minimum mean value as the baseline.
    corrected = wf - baseline_value
    # Subtracts the baseline value from the waveform.
    return corrected, baseline_value, min_index
    # Returns the corrected waveform, baseline value, and baseline index.



@staticmethod
def gain(T, Vset, dVset, dT):
    """
    Calculate the gain of the SiPM based on the applied voltage and temperature.
    Slope:     m = (7.838615914826629 ± 0.001483241521232523) e5 1/V
    Intercept: b = (9.809975406709395 ± 0.004662604806026203) e5
    #Gain(Overvoltage) = (7.83861591 ± 0.00148324)*1e5/V * Overvoltage + (9.80997541 ± 0.00466260)*1e5 -  at 25 degrees C - 0.12 V geringere overvoltage - 12% weniger Gain pro 4 Grad Temperaturerhöhung
    #Breakdownvoltage(T) = (39.16+-0.01)V + (33.66e-3+-0.30e-3) V/K * T
    #If the Voltage is set to Vset than the 
    # Overvoltage is Vset-Breakdownvoltage(T) 
    # = (Vset+-0.1)V - (39.16+-0.01)V - (33.66e-3+-0.30e-3) V/K * T 
    # Gain over Temperature: 
    # Gain(T) = (7.83861591 ± 0.00148324)*1e5/V * ((Vset+-0.1)V - (39.16+-0.01)V - (33.66e-3+-0.30e-3) V/K * T) + (9.80997541 ± 0.00466260)*1e5 
    # Gain(T) = A * (Vset - B - C * T) + D
    # A = 7.83861591e5 #
    # dA = 0.00148324e5
    # B = 39.16
    # dB = 0.01
    # C = 33.66e-3
    # dC = 0.30e-3
    # D = 9.80997541e5
    # dD = 0.00466260e5
    # dT = 0.1
    # dVset = 0.1
    # dGain(T) = sqrt((dA * (Vset + B + C*T))^2 + (A * dVset)^2 + (dB*A)^2 + (dC * T*A)^2 + (dT*C*A)+ (dD)^2) 
    # SiPM output should proportional to Gain(T) and therefore the SiPM output should decrease with increasing temperature.
    Parameters
    ----------
    T : float/array of floats
        Temperature of the PCB in degrees Celsius.
    Returns
    -------
    Gain : float
        Calculated gain.
    dGain: float    
        Calculated uncertainty of the gain.
    """
    Gain_per_volt = 7.83861591e5 #Gain change per Volt [V^-1] Fit parameter determined from Gain_overvoltage.csv
    dGain_per_volt = 0.00148324e5 
    Breakdown_voltage = 39.16 #Breakdown voltage [V] constant Fit parameter determined from breackdownvoltage_temperature.csv
    dBreakdown_voltage = 0.01
    Breakdown_per_K = 33.66e-3 #Breakdown voltage change per Kelvin [V/K] Fit parameter determined from breackdownvoltage_temperature.csv
    dBreakdown_per_K = 0.30e-3
    Gain_const = 9.80997541e5 #Gain constant [V] Fit parameter determined from Gain_overvoltage.csv
    dGain_const = 0.00466260e5
    Gain = Gain_per_volt * (Vset - Breakdown_voltage - Breakdown_per_K * T) + Gain_const
    dGain = np.sqrt((dGain_per_volt * (Vset - Breakdown_voltage - Breakdown_per_K * T))**2 + (Gain_per_volt * dVset)**2 + (dBreakdown_voltage * Gain_per_volt)**2 + (dBreakdown_per_K * T* Gain_per_volt)**2 + (dT * Breakdown_per_K * Gain_per_volt)**2 + (dGain_const)**2)
    return Gain, dGain


"""
Calculate group-wise statistics from event-wise waveform data.
This function operates on the event-wise integrated data produced by
`calculate_eventwise_ratios()`. The data are grouped according to either
the detector row (`i`) or column (`j`).
For each group:
    1. All waveform-level values from all events belonging to that group
        are collected.
    2. The nested waveform arrays are flattened into a single 1D array.
    3. The mean, standard deviation, and standard error of the mean (SEM)
        are calculated.
The SEM is calculated as:
    SEM = std / sqrt(N)
where N is the total number of waveform-level measurements in the group.
Parameters
----------
value_col : str
    Name of the column to analyze
    (e.g. "PMT_out_ref", "PMT_in", ...).
group_col : str
    Grouping variable. Must be either:
        - "i" : detector row
        - "j" : detector column
step_size : float, optional
    Physical spacing between neighboring detector positions in mm.
ny : int, optional
    Number of detector positions. If None, the number of unique groups
    is used.
Returns
-------
y_phys : np.ndarray
    Physical coordinate corresponding to each group.
groups : dict
    Dictionary containing all values grouped by i or j.
group_means : list
    Mean value for each group.
group_stds : list
    Standard deviation for each group.
group_sem : list
    Standard error of the mean for each group.
Raises
------
ValueError
    If `group_col` is not "i" or "j".
ValueError
    If event-wise data have not yet been generated by
    `calculate_eventwise_ratios()`.
"""
def grouped_mean_std(
    self,
    value_col,
    group_col,
    data="eventwise",
    step_size=6.55,
    ny=None,
):
    """Calculate mean, std and SEM grouped by detector row or column."""

    if group_col not in ["i", "j"]:
        raise ValueError("group_col must be 'i' or 'j'")

    # Select data source
    if data == "eventwise":
        dataframe = self.baseline_integrated_data_eventwise
    elif data == "integrated":
        dataframe = self.baseline_integrated_data
    else:
        raise VaslueError(
            "data must be either 'eventwise' or 'integrated'"
        )

    # Check that requested value exists
    if value_col not in dataframe:
        raise ValueError(
            f"'{value_col}' does not exist in the selected data."
        )

    # Extract grouping variable
    group_ids = dataframe[group_col]

    # Extract values
    values = dataframe[value_col]

    # Collect values for each group
    groups = {}

    for val, gid in zip(values, group_ids):
        groups.setdefault(gid, []).append(val)

    # Calculate statistics
    group_means = []
    group_stds = []
    group_sem = []

    for gid, vals in groups.items():

        # Flatten event/waveform structure
        vals = np.asarray(vals).flatten()

        # Ignore zero values
        vals = vals[vals != 0]

        group_means.append(vals.mean())
        group_stds.append(vals.std(ddof=1))
        group_sem.append(
            vals.std(ddof=1) / np.sqrt(len(vals))
        )

    # Number of groups
    if ny is None:
        ny = len(groups)

    # Physical coordinate
    if group_col == "j":

        y_phys = np.linspace(0, 360, ny)

    else:

        if (
            "_in" in value_col
            or "ref_in" in value_col
            or "Noise_in" in value_col
            or "norm_in" in value_col
        ):
            y_phys = np.linspace(
                202,
                202 - ny * step_size,
                ny
            )

        elif (
            "_out" in value_col
            or "ref_out" in value_col
            or "Noise_out" in value_col
            or "norm_out" in value_col
        ):
            y_phys = np.linspace(
                195,
                195 - ny * step_size,
                ny
            )

        else:
            y_phys = np.linspace(
                199,
                199 - ny * step_size,
                ny
            )

    return (
        y_phys,
        groups,
        group_means,
        group_stds,
        group_sem,
    ) 

           
"""
Calculate event-wise signal ratios between selected detector channels
and their corresponding reference channels.
For each detector channel defined in `ratio_map`, the function:
1. Integrates every waveform within each event using the appropriate
integration window.
2. Applies a sign correction for PMT channels because PMT pulses are
recorded with negative polarity.
3. Computes the ratio of the integrated signal to the integrated
reference signal for every waveform in the event.
4. Calculates the mean and standard deviation of these ratios within
each event.
5. Stores the event-wise mean ratios and their standard deviations in
`self.baseline_integrated_data`.
The resulting arrays are saved under:
    "{channel}_ref"      : mean ratio per event
    "{channel}_ref_std"  : standard deviation of ratios per event
Returns
-------
PM_int : list of np.ndarray
    Integrated signal values for all processed events and channels.
    [0] - "PMT_in"
    [1] - "SiPMout_in"
    [2] - "PMT_out"
    [3] - "SiPMin_out"
REF_int : list of np.ndarray
    Integrated reference signal values for all processed events and
    channels.
    [0] - "SiPMin_in"
    [1] - "SiPMin_in"
    [2] - "SiPMout_out"
    [3] - "SiPMout_out"        
"""
def calculate_eventwise_ratios_old(self):
    # ----------------------------------------
    # Mapping of signal channels to their
    # corresponding reference channels
    # ----------------------------------------
    if self.baseline_integrated_data_eventwise == None:
        ValueError("self.baseline_integrated_data_eventwise does not exist.\n Please run load() with option redo=True")    
    else: 
        if self.baseline_integrated_data_eventwise.get("SiPMin_in_gain") is not None:
            ratio_map = {
                    "PMT_in": "SiPMin_in_gain",
                    "SiPMout_in_gain": "SiPMin_in_gain",
                    "PMT_out": "SiPMout_out_gain",
                    "SiPMin_out_gain": "SiPMout_out_gain"
            }  
        else:
            ratio_map = {
                    "PMT_in": "SiPMin_in",
                    "SiPMout_in": "SiPMin_in",
                    "PMT_out": "SiPMout_out",
                    "SiPMin_out": "SiPMout_out"
            } 
        for pm, ref in ratio_map.items():
            ratios_mean = []
            ratios_std = []
            Ratios = [] 
            # Loop over all events
            for pm_row, ref_row in zip(
                self.baseline_integrated_data_eventwise[pm],
                self.baseline_integrated_data_eventwise[ref]
            ):
                # Convert rows to 1D float arrays
                pm_arr = np.array(pm_row, dtype=np.float32)
                ref_arr = np.array(ref_row, dtype=np.float32)
                # Calculate ratio element-wise; place np.nan where ref_arr == 0
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = np.where(ref_arr != 0, pm_arr / ref_arr, np.nan)
                    
                # Calculate statistics while ignoring np.nan entries
                
                ratios_mean.append(np.nanmean(ratios))
                ratios_std.append(np.nanstd(ratios))
                Ratios.append(ratios)

            # Store results
            self.baseline_integrated_data[f"{pm}_ref"] = np.array(ratios_mean)
            self.baseline_integrated_data[f"{pm}_ref_std"] = np.array(ratios_std)
            self.baseline_integrated_data_eventwise[f"{pm}_ref"] = np.array(Ratios, dtype=object)

            #self.savecsv(name="baseline_integrated_data")    

def calculate_eventwise_ratios(self):

    if self.baseline_integrated_data_eventwise is None:
        raise ValueError(
            "self.baseline_integrated_data_eventwise does not exist.\n"
            "Please run load() with option redo=True"
        )

    if self.baseline_integrated_data_eventwise.get("SiPMin_in_gain") is not None:
        ratio_map = {
            "PMT_in": "SiPMin_in_gain",
            "SiPMout_in_gain": "SiPMin_in_gain",
            "PMT_out": "SiPMout_out_gain",
            "SiPMin_out_gain": "SiPMout_out_gain"
        }
    else:
        ratio_map = {
            "PMT_in": "SiPMin_in",
            "SiPMout_in": "SiPMin_in",
            "PMT_out": "SiPMout_out",
            "SiPMin_out": "SiPMout_out"
        }

    for pm, ref in ratio_map.items():
        print("Calculating event-wise ratios for:", pm, "with reference:", ref)
        pm_arr = np.asarray(
            self.baseline_integrated_data_eventwise[pm],
            dtype=np.float32
        )

        ref_arr = np.asarray(
            self.baseline_integrated_data_eventwise[ref],
            dtype=np.float32
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.divide(
                pm_arr,
                ref_arr,
                out=np.full_like(pm_arr, np.nan),
                where=ref_arr != 0
            )

        # Mean and std over the waveforms of each event
        ratios_mean = np.nanmean(ratios, axis=1)
        ratios_std = np.nanstd(ratios, axis=1)

        # Store eventwise ratios
        self.baseline_integrated_data[f"{pm}_ref"] = ratios_mean
        self.baseline_integrated_data[f"{pm}_ref_std"] = ratios_std

        self.baseline_integrated_data_eventwise[
            f"{pm}_ref"
        ] = ratios
        print("Saved event-wise ratios for:", pm, "in shape", ratios.shape)


def apply_sipm_corrections(self, Vset = 40.7, dVset = 0.1, dT = 0.1):
    '''
    - calculate gain
    - first correct all SiPM values with the gain 
    - Normalize SiPM values to the 999 measurement if it exists
    '''
    #name == "integrated_data":   
    #    integrated_data = self.integrated_data
    #elif name == "baseline_integrated_data":    
    #    integrated_data = self.baseline_integrated_data
    print("Applying SiPM gain corrections")
    self.baseline_integrated_data["gain_out"], self.baseline_integrated_data["dgain_out"] = self.gain(self.baseline_integrated_data["temp_out"], Vset = Vset, dVset = dVset, dT = dT)
    self.baseline_integrated_data["gain_in"], self.baseline_integrated_data["dgain_in"] = self.gain(self.baseline_integrated_data["temp_in"], Vset = Vset, dVset = dVset, dT = dT)
    PM = ["SiPMout_in", "SiPMin_in", "SiPMout_out", "SiPMin_out"]
    Gain = ["gain_out", "gain_in", "gain_out", "gain_in"]

    for pm, g in zip(PM, Gain):  
        print("Correcting SiPM values for:", pm)
        gain_matrix = self.baseline_integrated_data[g].to_numpy(
            dtype=np.float32
        )

        dgain_matrix = self.baseline_integrated_data[f"d{g}"].to_numpy(
            dtype=np.float32
        )
        eventwise = np.asarray(
                    self.baseline_integrated_data_eventwise[pm],
                    dtype=np.float32
        )
        # Add singleton dimensions to gain/dgain until they
        # have the same number of dimensions as eventwise.
        gain_matrix = gain_matrix.reshape((len(gain_matrix),) + (1,) * (eventwise.ndim - 1))
        dgain_matrix = dgain_matrix.reshape((len(dgain_matrix),) + (1,) * (eventwise.ndim - 1))
        #gain_matrix = np.stack(self.baseline_integrated_data[g].values)[:, None , None]
        #print(np.shape(gain_matrix), np.shape(self.baseline_integrated_data_eventwise[pm]))
        #dgain_matrix = np.stack(self.baseline_integrated_data[f"d{g}"].values)[:, None , None]

        print("eventwise:", eventwise.shape)
        print("gain:", gain_matrix.shape)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(
                eventwise,
                gain_matrix,
                out=np.full_like(eventwise, np.nan),
                where=gain != 0
            )

            ratio_err = np.abs(ratio) * np.abs(dgain_matrix / gain_matrix)

        print("ratio:", ratio.shape)
        
        self.baseline_integrated_data_eventwise[f"{pm}_gain"] =  ratio
        self.baseline_integrated_data_eventwise[f"{pm}_gain_err"] = ratio_err
    if self.integrated_data["j"][0] == 99999999:
        for pm in PM:
            ref = self.baseline_integrated_data_eventwise[f"{pm}_gain"][0][self.baseline_integrated_data_eventwise[pm][0]!= 0].mean()
            if ref != 0:
                ratio = self.baseline_integrated_data_eventwise[f"{pm}_gain"][1:]/ref
            #ratio_err = np.abs(ratio) * np.sqrt(1/30*
            #                (self.baseline_integrated_data_eventwise[f"{pm}_std"][1:]/self.baseline_integrated_data_eventwise[pm][1:])**2 +
            #                (self.baseline_integrated_data_eventwise[f"{pm}_std"][0]/self.baseline_integrated_data_eventwise[pm][0])**2
            #            )
                self.baseline_integrated_data_eventwise[f"{pm}_gain"][1:] = ratio
            else:
                print(pm, "ref = 0", ref)
            #self.baseline_integrated_data_eventwise[f"{pm}_err"] = ratio_err

            

            
"""
Create baseline-corrected waveform data and corresponding integrated values.    
For each channel:
    1. Applies baseline correction to every waveform using
    `correct_baseline_min()`.
    2. Integrates the corrected waveform using
    `riemann_sum_peak()`.
    3. Computes the mean and standard deviation of the
    integrals for each scan position.    
The baseline correction searches for the minimum summed
region inside a predefined pre-signal window and subtracts
the corresponding baseline offset.    
PMT channels are integrated with negative polarity,
SiPM channels with positive polarity.
Takes arguments for baseline correction:
        window=(20, 0, 100), (integration window length, start index, end index)
        sigma= 0 (if > 0, applies smoothing before baseline correction),
        smooth_method=2 (running average), 2 (gaussian)
Creates:
    self.baseline_wf :
        Dictionary containing baseline-corrected waveforms.    
    self.baseline_integrated_data :
        Pandas DataFrame containing:
            - scan coordinates
            - timestamps
            - temperatures
            - mean integrated charge per position
            - standard deviation per position    
The generated data are automatically saved using:
    savecsv(name="baseline_integrated_data")
    savebin(name="baseline_waveforms")
"""
def create_baseline_data_old(self, window=(20, 0, 100), sigma=0, smooth_method=2):
    PM = [
        "PMT_in", "SiPMin_in", "SiPMout_in",
        "PMT_out", "SiPMin_out", "SiPMout_out"
    ]
    self.baseline_wf = {}
    self.baseline_integrated_data = {}
    # Store integrated signal and reference values
    self.baseline_integrated_data_eventwise = {} 
    # ----------------------------------------
    # Copy metadata
    # ----------------------------------------
    for key in ["i", "j", "time", "temp_in", "temp_out"]:
        self.baseline_integrated_data[key] = (
            self.integrated_data[key].copy()
        )
    self.baseline_wf["i"] = self.waveforms["i"].copy()
    self.baseline_wf["j"] = self.waveforms["j"].copy()
    self.baseline_integrated_data_eventwise["i"] = self.waveforms["i"].copy()
    self.baseline_integrated_data_eventwise["j"] = self.waveforms["j"].copy()
    # ----------------------------------------
    # Baseline correction + integration
    # ----------------------------------------
    for pm in PM:
        corrected_rows = []
        integrals = []
        std_devs = []
        PM_int = []
        if "PMT" in pm:
            int_window_min = self.int_window_min_PMT
            int_window_max = self.int_window_max_PMT
            sign = -1
        elif "SiPM" in pm:
            int_window_min = self.int_window_min_SiPM
            int_window_max = self.int_window_max_SiPM
            sign = 1
        else:
            raise ValueError(f"Unknown PM type: {pm}")
        #print("self.waveforms[pm].shape", self.waveforms[pm].shape)
            
        for row in self.waveforms[pm]:
            corrected_row = []
            integral = []
            for wf in row:
                # Baseline correction
                wf_corr, baseline, index = self.correct_baseline_min(
                    wf,
                    window=window,
                    sigma=sigma,
                    smooth_method=smooth_method
                )[0]
                print(f"Baseline correction for {pm}: baseline={baseline}, index={index}")
                corrected_row.append(wf_corr)
                # Integration
                _, _, area = self.riemann_sum_peak(
                    wf_corr,
                    int_window_min,
                    int_window_max
                )
                integral.append(sign * area) 
                
            integral = np.asarray(integral)
            nonzero = integral[integral != 0]
            if len(nonzero) > 0:
                integrals.append(nonzero.mean())
            else:
                integrals.append(np.nan)

            if len(nonzero) > 1:
                std_devs.append(nonzero.std(ddof=1))
            else:
                std_devs.append(np.nan)

            PM_int.append(np.array(integral))
            corrected_rows.append(corrected_row)
        self.baseline_wf[pm] = np.array(corrected_rows, dtype=np.float32)
        self.baseline_integrated_data_eventwise[pm] = np.array(PM_int, dtype=np.float32)
        self.baseline_integrated_data[pm] = np.array(integrals)
        self.baseline_integrated_data[pm + "_std"] = np.array(std_devs)

    #print("\nLengths before DataFrame:")
    #for key, value in self.baseline_integrated_data.items():
    #    print(key, len(value))

    self.baseline_integrated_data = pd.DataFrame(
        self.baseline_integrated_data
    )
    # ----------------------------------------
    # Save
    # ----------------------------------------
    #self.savecsv(name="baseline_integrated_data")
    #self.savebin(name="baseline_waveforms")
    #self.savebin(name="baseline_integrated_data_eventwise")
def create_baseline_data(self, window=(20, 0, 100), sigma=0, smooth_method=2):
    PM = [
        "PMT_in", "SiPMin_in", "SiPMout_in", "PMT_out", "SiPMin_out", "SiPMout_out"]
    self.baseline_wf = {}
    self.baseline_integrated_data = {}
    self.baseline_integrated_data_eventwise = {}
    # ----------------------------------------
    # Copy metadata
    # ----------------------------------------
    for key in ["i", "j", "time", "temp_in", "temp_out"]:
        self.baseline_integrated_data[key] = (
            self.integrated_data[key].copy()
        )
    self.baseline_wf["i"] = self.waveforms["i"].copy()
    self.baseline_wf["j"] = self.waveforms["j"].copy()

    self.baseline_integrated_data_eventwise["i"] = (
        self.waveforms["i"].copy()
    )
    self.baseline_integrated_data_eventwise["j"] = (
        self.waveforms["j"].copy()
    )
    # ----------------------------------------
    # Baseline correction + integration
    # ----------------------------------------
    chunk_size = 500  # Number of positions to process in each chunk
    for pm in PM:
        if "PMT" in pm:
                    int_window_min = self.int_window_min_PMT
                    int_window_max = self.int_window_max_PMT
                    sign = -1
        elif "SiPM" in pm:
                    int_window_min = self.int_window_min_SiPM
                    int_window_max = self.int_window_max_SiPM
                    sign = 1
        else:
                    raise ValueError(f"Unknown PM type: {pm}")
        waveforms = np.asarray(self.waveforms[pm], dtype=np.float32)

        if waveforms.ndim == 2:
            waveforms = waveforms[:, np.newaxis, :]

        n_positions = waveforms.shape[0]
        n_waveforms = waveforms.shape[1]

        corrected_all = np.empty_like(waveforms)
        integrals_all = np.empty(
            (n_positions, n_waveforms),
            dtype=np.float32
        )

        for start in range(0, n_positions, chunk_size):
            stop = min(start + chunk_size, n_positions)

            chunk = waveforms[start:stop]

            corrected, baselines, indices = self.correct_baseline_min(
                chunk,
                window=window,
                sigma=sigma,
                smooth_method=smooth_method
            )

            integrals = self.riemann_sum_peak(
                corrected,
                int_window_min,
                int_window_max
            )

            integrals *= sign

            corrected_all[start:stop] = corrected
            integrals_all[start:stop] = integrals

            del corrected
            del baselines
            del indices
            del integrals

        # ----------------------------------------
        # Store eventwise data
        # ----------------------------------------
        self.baseline_wf[pm] = corrected_all
        print(f"Stored baseline-corrected waveforms for {pm}.")
        self.baseline_integrated_data_eventwise[pm] = integrals_all
        print(f"Stored event-wise integrated data for {pm}.")
        # ----------------------------------------
        # Mean and standard deviation
        # over the m waveforms
        # ----------------------------------------
        # ----------------------------------------
        # Mean and standard deviation
        # over the m waveforms
        # ----------------------------------------

        valid = integrals_all != 0

        counts = valid.sum(axis=1)

        # Replace invalid (zero) values by NaN
        values = np.where(
            valid,
            integrals_all,
            np.nan
        )

        # Mean, ignoring zeros
        means = np.nanmean(values, axis=1)

        # Rows with no valid values should be NaN
        means[counts == 0] = np.nan

        # Standard deviation
        stds = np.full(
            means.shape,
            np.nan,
            dtype=np.float32
        )

        # Only calculate std for rows with at least 2 valid waveforms
        valid_std = counts > 1

        stds[valid_std] = np.nanstd(
            values[valid_std],
            axis=1,
            ddof=1
        )

        self.baseline_integrated_data[pm] = means
        print(f"Stored mean integrated data for {pm}.") 
        self.baseline_integrated_data[pm + "_std"] = stds
        print(f"Stored standard deviation of integrated data for {pm}.")

    # ----------------------------------------
    # Inspect before DataFrame
    # ----------------------------------------
    print("\nBefore DataFrame:")

    for key, value in self.baseline_integrated_data.items():
        print(
            key,
            type(value),
            np.shape(value),
            getattr(value, "dtype", None),
            getattr(value, "nbytes", 0) / 1e6,
            "MB"
        )

    print("Creating DataFrame...")

    self.baseline_integrated_data = pd.DataFrame(
        self.baseline_integrated_data
    )

    print("DataFrame created.")