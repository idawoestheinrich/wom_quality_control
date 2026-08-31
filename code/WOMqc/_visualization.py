# Importing Libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
from datetime import datetime, timedelta

#from moku.instruments import Oscilloscope
#to convert data from arduino to array
import ast
'''
Provides plot generation including live scopes, heatmaps, temperature histories, and waveform visualizations.
'''
'''
├── init_plotting()
├── update_example_waveform()
├── plot_temperature_over_time()
├── plot_integral_over_time()
├── heatmap_WOM()
├── light_yield_1dim()
├── charge_spectrum()
└── plot_waveforms_position()
'''
'''
dynamic plotting of example waveforms for each position during the scan.
'''  
"""
Initialize interactive waveform plotting window.
Creates a 1×3 subplot layout and prepares empty line objects
for real-time waveform updates.
Enables matplotlib interactive mode for live updates.
Stores:
    self.fig  : figure handle
    self.ax   : axes array (3 plots)
    self.lines: list of line objects
"""
def init_plotting(self):
    plt.ion()  # interactive mode ON
    self.fig, self.ax = plt.subplots(1, 3, figsize=(12, 4))
    self.lines = []
    for i in range(3):
        line, = self.ax[i].plot([], [], lw=1)
        self.lines.append(line)
        self.ax[i].set_xlabel("Time [samples]")
        self.ax[i].set_ylabel("Voltage [V]")
        self.ax[i].set_title(f"Example waveform {i+1}")
        self.ax[i].grid(True)
    self.fig.tight_layout()
"""
Update example waveform plots for a given scan position.
Updates 3 waveform channels simultaneously in an interactive plot.
Takes:
    waveforms : list[np.ndarray]
        List of 1D waveform arrays (length = 3)
    j : int
        Scan j-position
    i : int
        Scan i-position
    ch : list[str]
        Channel names used in plot titles
Notes:
    - Uses fixed time axis from acquisition settings
    - Rescales y-axis dynamically per update
    - Intended for live / iterative visualization
"""
def update_example_waveform(self, waveforms, j, i, ch):
    """
    waveforms: iterable of 3 1D numpy arrays
    """
    x = np.linspace(
        self.rec_time_min,
        self.rec_time_max,
        self.frame_length_max
    )
    xmin, xmax = self.rec_time_min, self.rec_time_max
    for idx in range(3):
        ymin, ymax = waveforms[idx].min() * 1.1, waveforms[idx].max() * 1.1        
        y = waveforms[idx]
        self.lines[idx].set_data(x, y)
        self.ax[idx].set_xlim(xmin, xmax)
        self.ax[idx].set_ylim(ymin, ymax)
        self.ax[idx].set_title(
            f"{ch[idx]} mean over {self.rep} wfs | j={j}, i={i}"
        )
    self.fig.canvas.draw_idle()
    plt.pause(0.1)
"""
Plot detector temperatures as a function of daytime.
Shows the inside and outside temperature measurements
including constant measurement uncertainties.
Returns:
    None
"""
def plot_temperature_over_time(self):
    timearr = []
    day = 0
    time0 = datetime.strptime(self.integrated_data["time"][0], "%H:%M:%S")
    prev_t = time0
    for i in range(len(self.integrated_data["time"])):
        t = datetime.strptime(self.integrated_data["time"][i], "%H:%M:%S")
        # detect midnight wrap
        if t < prev_t:
            day += 1
        # build full datetime (anchor to arbitrary date, e.g. Jan 1)
        t_full = datetime(2000, 1, 1) + timedelta(days=day,
                                                hours=t.hour,
                                                minutes=t.minute,
                                                seconds=t.second)
        timearr.append(t_full)
    prev_t = t
    plt.errorbar(np.array(timearr), np.array(self.integrated_data["temp_in"]), yerr = 0.0375, label="Temperature inside WOM")
    plt.errorbar(np.array(timearr), np.array(self.integrated_data["temp_out"]), yerr = 0.0375, label="Temperature outside WOM")        
    plt.xlabel("Daytime [HH:MM]")
    plt.ylabel("Temperature [°C]")
    plt.legend()
    # 👉 format x-axis to show only hours and minutes
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # optional: control tick density
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()        
"""
Plot integrated signal values over time for selected channels.
Creates a time evolution plot of the integrated charge
for the provided detector channels.
Takes:
    PM : list[str]
        List of channel names to plot.
Returns:
    None
""" 
def plot_integral_over_time(self, PM):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    for pm in PM: 
        ax.plot(np.array(self.integrated_data["time"]), np.array(self.integrated_data[pm]), label = pm)
    ax.set_xlabel("Time")
    ax.set_ylabel("Integral [Vxs]")
    plt.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=8))
"""
Plot heatmaps of the integrated data for one or two of the following options (ch1 = "...", ch2 = "..."):
    temp_in,
    temp_out,
    PMT_ratio_in,
    PMT_ratio_in_std,
    SiPM_ratio_in,
    SiPM_ratio_in_std,
    SiPM_ref_in,
    SiPM_ref_in_std,
    PMT_ratio_out,
    PMT_ratio_out_std,
    SiPM_ratio_out,
    SiPM_ratio_out_std,
    SiPM_ref_out,
    SiPM_ref_out_std
vmin, vmax: color scale limits as list vmin = [vmin], vmax = [vmax]/ vmin = [vmin_ch1, vmin_ch2], vmax = [vmax_ch1, vmax_ch2]
step_size = 6.55 - vertical step size - programmed into the motionsoft, software - note: it probably also makes sense to add a rotation step size that is programed into the arduino
colorlabel = None - Set to label of the colorbar (string) if you want to set the label
title = None  - Set to title string if you want to set the title 
baseline_subtract = True - set to False if you want to see the raw data without baseline correction
normalize = False,  - Set to True if you want to see the plot normalized to its mean
remove_rows = [0,0] : must be an array with [how many first rows should be removed, how many last rows should be removed]
"""
def heatmap_WOM(self, ch1= "PMT_ratio_in", ch2 = None, vmin=None, vmax=None,
                step_size = 6.55, colorlabel = None, title = None,  baseline_subtract = True, normalize = False, remove_rows = [0,0]):
    if baseline_subtract:
        if hasattr(self, "baseline_integrated_data"):
            integrated_data = self.baseline_integrated_data.copy()
            name_suffix = ""
        else:
            ValueError("self.baseline_integrated_data does not exist. Please run load()")    
    else: 
        integrated_data = self.integrated_data.copy()    
        name_suffix = "_wo_baseline_subtraction"
    # Extract values for heatmap x,y
    if self.integrated_data["j"][0] == 999:
        x, y = integrated_data["j"][1:], -integrated_data["i"][1:]
    else:
        x, y = integrated_data["j"], -integrated_data["i"]    
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    X = x.values.reshape(ny, nx)
    Y = y.values.reshape(ny, nx)        
    if colorlabel == None:
        if "ratio" in ch1 and "std" in ch1:
            colorbar_label = ["Standard deviation [a.u.]"]
        elif "std" in ch1:
            colorbar_label = ["Standard deviation [Vxs]"]    
        elif "err" in ch1:   
            colorbar_label = ["Error [Vxs]"]
        elif "pull" in ch1:
            colorbar_label = ["Difference/Error"]    
        elif "ref" in ch1 or "ratio" in ch1:
            colorbar_label = ["Integrated yield / Reference"]
        elif "temp" in ch1:
            colorbar_label = ["Temperature [°C]"]   
        if normalize:
            if "std" in ch1:
                colorbar_label = ["Standard deviation [a.u.]"]
            elif "err" in ch1:
                colorbar_label = ["Error [a.u.]"]   
            else:      
                colorbar_label = ["Normalized integrated yield [a.u.]"]    
        else:
            colorbar_label = ["Integral [Vxs]"]    
    else:
        colorbar_label = colorlabel
    if ch2: 
        # ---- Heatmap for two channels ----
        fig, ax = plt.subplots(1, 2, figsize=(12, 5),  facecolor='white')
        channels = [ch1, ch2]
        if title == None:
            title = [f"{ch1}", f"{ch2}"]
        dfname = self.foldername / f"{self.filename}_{ch1}_{ch2}{name_suffix}.pdf"
        if colorbar_label != colorlabel:
            if "ratio" in ch2 and "std" in ch2:
                colorbar_label.append("Standard deviation [a.u.]")
            elif "std" in ch2:
                colorbar_label.append("Standard deviation [Vxs]")   
            elif "err" in ch2:   
                colorbar_label.append("Error [Vxs]")
            elif "pull" in ch2:
                colorbar_label.append("Difference/Error")    
            elif "ref" in ch2 or "ratio" in ch2:
                colorbar_label.append("Integrated yield / Reference")
            elif "temp" in ch2:
                colorbar_label.append("Temperature [°C]")   
            if normalize:
                if "std" in ch2:
                    colorbar_label.append("Standard deviation [a.u.]")
                elif "err" in ch2:
                    colorbar_label.append("Error [a.u.]")   
                else:     
                    colorbar_label.append("Normalized integrated yield [a.u.]")    
            else:
                colorbar_label.append("Integral [Vxs]")           
    else:    
        # ---- Heatmap ----
        fig, ax = plt.subplots(1, 1, figsize=(6, 5),  facecolor='white')
        if title == None:
            title = [f"{ch1}"]
        dfname = self.foldername / f"{self.filename}_{ch1}{name_suffix}.pdf"
        ax = np.atleast_1d(ax)
        channels = [ch1]        
    for idx, ch in enumerate(channels):    
        # Extract values for heatmap z = integrated data[ch1]
        if self.integrated_data["j"][0] == 999:
            pivot = integrated_data.iloc[1:].pivot(index="i", columns="j", values=ch)
        else:
            pivot = integrated_data.pivot(index="i", columns="j", values=ch)    
        # remove rows? 
        #   
        if remove_rows != [0,0]:
            if remove_rows[0] != 0 and remove_rows[1] == 0:
                pivot = pivot.iloc[remove_rows[0]:]
            elif remove_rows[0] == 0 and remove_rows[1] != 0:  
                pivot = pivot.iloc[:-remove_rows[1]]
            else:    
                pivot = pivot.iloc[remove_rows[0]:-remove_rows[1]]                
        if normalize and "std" not in ch and "err" not in ch:
            overall_mean = pivot.mean().mean()   #[here one has to take the mean not the mean of the std]
        elif normalize and "std" in ch or "err" in ch:
            overall_mean = overall_mean
        else:
            overall_mean = 1            
        #X, Y = np.meshgrid(pivot.columns.values, -pivot.index.values)
        Z = pivot.values/overall_mean
        nx = Z.shape[1]
        ny = Z.shape[0]
        x_phys = np.linspace(0, 360, nx)
        if "_in" in ch or "ref_in" in ch or "Noise_in" in ch or "norm_in" in ch:
            y_phys = np.linspace(202-remove_rows[0]*step_size, 202 - ny*step_size, ny)  # inside spacing
            y_ticks = np.linspace(202-remove_rows[0]*step_size, 202 - ny*step_size, ny)
        elif "_out" in ch or "ref_out" in ch or "Noise_out" in ch or "norm_out" in ch:
            y_phys = np.linspace(195-remove_rows[0]*step_size, 195 - ny*step_size, ny)  # outside spacing
            y_ticks = np.linspace(195-remove_rows[0]*step_size, 195 - ny*step_size, ny)
        else:
            y_phys = np.linspace(199-remove_rows[0]*step_size, 199 - ny*step_size, ny)
            y_ticks = np.linspace(199-remove_rows[0]*step_size, 199 - ny*step_size, ny)
        X, Y = np.meshgrid(x_phys, y_phys)
        #z = self.integrated_data[ch]
        #Z = z.values.reshape(ny, nx)
        if vmin and vmax: 
            im = ax[idx].pcolormesh(X, Y, Z, cmap='plasma', vmin=vmin[idx], vmax=vmax[idx], shading='auto')  # heatmap
        else:
            im = ax[idx].pcolormesh(X, Y, Z, cmap='plasma', shading='auto')  # heatmap         
        cbar = fig.colorbar(im, ax=ax[idx], label=colorbar_label[idx])
        ax[idx].set_xlabel(r"Rotation $\phi$ [°]")
        ax[idx].set_ylabel("Distance to PMT [mm]")
        ax[idx].set_title(title[idx])
        ax[idx].set_ylim(190 - ny*step_size, 207)
        # Adjust y-axis ticks
        #ax[idx].set_yticks(y_phys)
        #ax[idx].set_yticklabels(y_ticks)
        # Adjust x-axis ticks
        #if nx < 10:
            #   nx_ticks = nx
        #else:
        #    nx_ticks = 10
        #ax[idx].set_xticks(np.linspace(0, nx, nx_ticks))
        #ax[idx].set_xticklabels(np.linspace(0, 360, nx_ticks))
    fig.tight_layout()
    # Save heatmap to folder
    plt.savefig(dfname)
    plt.show()
    return 0
"""
Plot integrated light yield or related quantities over time
for up to three detector channels.
Creates a 1D comparison plot showing the evolution of:
    - signal integrals
    - standard deviations
    - temperatures
for the selected channels.
Takes:
    ch1, ch2, ch3 : str
        Channel names to plot.
Returns:
    int
        Returns 0 after plotting and saving the figure.
Saves:
    PDF figure in the measurement folder.
"""
def light_yield_1dim(self, ch1= "PMT_in", ch2 = "SiPMin_in", ch3="SiPMout_in"):
    # Extract values for heatmap x,y
    x = self.integrated_data["time"]    
    if "std" in ch1:
        ylabel = "Standard deviation [Vxs]"
    elif "int" in ch1:
        ylabel = "Integral [Vxns]"
    elif "temp" in ch1:
        ylabel = "Temperature [°C]" 
    else:
        ylabel = "Unknown" 
    # ---- Plot ----
    fig, ax = plt.subplots(1, 3, figsize=(12, 4),  facecolor='white')        
    dfname = self.foldername / f"{self.filename}_Light_yield_over_distance_to_PMT_{ch1}.pdf"        
    channels = [ch1, ch2, ch3]
    for idx, ch in enumerate(channels):               
        y = self.integrated_data[ch]
        y = y[y != 0] * 1e9
        if "t_in" in ch1:
            x_scale = x[:len(y)]
        elif "t_out" in ch1:
            x_scale = x[len(y)-1:2*len(y)-1]
        else:
            x_scale = np.arange(0, len(y))    
        ax[idx].plot(x_scale, y, label=f"{round(np.mean(y),2)}+-{round(np.std(y),2)}")
        ax[idx].set_xlabel("time [h:m:s]")
        ax[idx].set_ylabel(ylabel)
        ax[idx].xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax[idx].legend()
        ax[idx].set_title(f"{self.WOMname} {ch}")
        # Adjust y-axis ticks
        #ax[idx].set_yticks(np.linspace(-(ny-1), 1, nymax))
        #ax[idx].set_yticklabels(y_ticks)
        # Adjust x-axis ticks
        #if nx < 10:
            #   nx_ticks = nx
        #else:
        #    nx_ticks = 10
        #ax[idx].set_xticks(np.linspace(0, nx, nx_ticks))
        #ax[idx].set_xticklabels(np.linspace(0, 360, nx_ticks))
    fig.tight_layout()
    # Save heatmap to folder
    plt.savefig(dfname)
    plt.show()
    return 0    
'''
Print the charge spectrum for one of the following options:
        PMT_ratio_in,
        SiPM_ratio_in, 
        SiPM_ref_in,
        PMT_out,
        SiPMin_out,
        SiPMout_out
    bins: number of bins in the histogram
    vmin, vmax: limits of the x-axis
    '''    
def charge_spectrum(self, ch="PMT_ratio_in", bins=90, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    data = self.integrated_data[self.integrated_data[ch] != 0]
    ax.hist(
        data,
        bins=bins,
        range=None if vmin is None or vmax is None else (vmin, vmax),
        histtype="step"
    )
    ax.set_xlabel("Integrated Signal [V·s]")
    ax.set_ylabel("# Entries")
    return fig, ax    
"""
    Plot all waveforms at a given scan position.    
    Takes:
        i : int or float
            Scan i-position    
        j : int or float
            Scan j-position    
        pm : str
            Channel name, e.g. "SiPMin_in"    
        baseline_corrected : bool
            If True, plots baseline-corrected waveforms.
            Otherwise plots raw waveforms.
    """
def plot_waveforms_position(
    self,
    i = 0,
    j = 0,
    PM=["PMT_in"],
    baseline_corrected = False,
    reps = None,
    plot_integration_window = False,
    plot_mean = False,
    legend = True,
    title = False
    ):
    if reps is None:
        reps = self.rep
    # ----------------------------------------
    # Find matching scan position
    # ----------------------------------------
    mask = (
        (self.integrated_data["i"] == i) &
        (self.integrated_data["j"] == j)
    )
    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise ValueError(f"No position found for i={i}, j={j}")
    idx = indices[0]
    # ----------------------------------------
    # Time axis
    # ----------------------------------------
    x = np.linspace(
        self.rec_time_min,
        self.rec_time_max,
        self.frame_length_max
    )*1e9 # convert to ns
    # ----------------------------------------
        # Plot
        # ----------------------------------------
    plt.figure(figsize=(8, 6))
    for pm in PM:
        # ----------------------------------------
        # Select waveform source
        # ----------------------------------------
        wf_sum = np.zeros_like(self.waveforms[pm][idx][0])
        if baseline_corrected:
            wf_data = self.baseline_wf[pm][idx]
        else:
            wf_data = self.waveforms[pm][idx]                
        if "in_in" in pm or "out_out" in pm:
                label = f"{pm} reference at i={i}, j={j}"
                color = 'green'
                color_line ='darkgreen'                    
        elif "in_out" in pm or "out_in" in pm:
                label = f"{pm} transmission at i={i}, j={j}" 
                color = 'blue'  
                color_line ='darkblue'                    
        elif "PMT" in pm:
                label = f"{pm} response at i={i}, j={j}"
                color = 'orange'
                color_line ='darkorange'
        else: 
                label = f"{pm} at i={i}, j={j}"   
        for wf in wf_data[0:reps]:  # plot all repetitions
            if plot_mean is False:
                alpha = max(1/reps, 0.1)   
                plt.plot(x, wf, alpha=alpha, color=color, label=label)
            elif plot_mean:
                wf_sum += wf/reps
            if plot_integration_window:
                if "PMT" in pm:
                    int_window_min = self.int_window_min_PMT
                    int_window_max = self.int_window_max_PMT
                    peak_idx = np.argmin(wf)
                    ymin = wf.min()
                    ymax = 0
                elif "SiPM" in pm:
                    int_window_min = self.int_window_min_SiPM
                    int_window_max = self.int_window_max_SiPM
                    peak_idx = np.argmax(wf)
                    ymin = 0 
                    ymax = wf.max()
                else:
                    raise ValueError(f"Unknown PM type: {pm}")                
                peak_time = x[peak_idx]                    
                start_time = peak_time - int_window_min*1e9
                end_time   = peak_time + int_window_max*1e9
                plt.vlines(
                    start_time,
                    ymin = ymin,
                    ymax = ymax,
                    color=color_line,
                    linestyle="--",
                    label=f"Integration start {pm}"
                )
                plt.vlines(
                    end_time,
                    ymin = ymin,
                    ymax = ymax,
                    color=color_line,
                    linestyle="--",
                    label=f"Integration end {pm}"
                )
        if plot_mean: 
            plt.plot(x, wf_sum, color=color, label=label)        
    plt.xlabel("Time difference to trigger time[ns]")
    plt.ylabel("Amplitude [V]")
    if legend:
        plt.legend()
    if title:
        plt.title(f"{pm} at i={i}, j={j}")
    #plt.grid(True)
    plt.xlim(self.rec_time_min*1e9, self.rec_time_max*1e9)
    plt.ylim(-1.1,1.5)
    plt.show()    