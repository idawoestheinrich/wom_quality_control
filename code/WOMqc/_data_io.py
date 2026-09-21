 # Importing Libraries
import numpy as np
import pandas as pd

from pathlib import Path
import logging

'''
Manages directory generation, logger setup,
metadata saving/loading, and binary/CSV persistence.
'''
'''
├── save_metadata()
├── savecsv()
├── savebin()
├── read_metadata()
├── readbin()
└── load()
'''

def _setup_logger(self):
    # 2️⃣ configure logger **here, inside the class**
    logger = logging.getLogger(self.__class__.__name__)
    logger.setLevel(logging.INFO)
        # Remove any default handlers to avoid duplicates
    if logger.hasHandlers():
            logger.handlers.clear()
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    # File handler — now self.foldername exists
    fh = logging.FileHandler(self.foldername / "logfile.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
'''
SAVING DATA / METADATA/ BASELINE CORRECTED DATA
'''
"""
Save experiment metadata to a text file.
Writes all class attributes (vars(self)) into a human-readable
key-value text file, excluding runtime or large data objects
that are not suitable for metadata storage.
Skipped keys:
    logger, i, j, device, arduino, waveforms, integrated_data
Output:
    metadata.txt in the experiment folder
Returns:
    None
"""
def save_metadata(self):
    metadata_file = self.foldername / "metadata.txt"
    with open(metadata_file, "w") as f:
        for key, value in vars(self).items():
            if key == "logger" or key == "i" or key == "j" or key == "device" or key == "arduino" or key == "waveforms" or key == "integrated_data" or key == "baseline_wf" or key == "baseline_integrated_data" or key == "baseline_integrated_data_eventwise":
                continue  # skip logger, i, j, device, arduino, waveforms, integrated_data
            f.write(f"{key}: {value}\n")
'''
    Used to read metadata from a previous measurement - should be used before loading bin or csv files
    md = WOMqc.read_metadata("foldername")
    qc = WOMqc(**md) 
'''
@classmethod
def read_metadata(cls, folder):
        metadata = {}
        metadata_file =  Path("/../data/")/ Path(folder) / "metadata.txt"
        with open(metadata_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue    
                key, value = line.split(":", 1)
                value = value.strip()    
                # type casting
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # keep string    
                metadata[key.strip()] = value
        return metadata
"""
Save integrated data to CSV file.
Supports saving either raw or baseline-corrected integrated data.
Takes:
    name : str
        - "integrated_data" → saves self.integrated_data
        - "baseline_integrated_data" → saves baseline version
Output:
    CSV file in experiment folder
Raises:
    ValueError if name is not recognized
Returns:
    None
"""
def savecsv(self, name = "integrated_data"):
    # Save integrated data into one csv file
    if name == "integrated_data":   
        filename = self.foldername / f"{self.filename}_integrated.csv"
        self.integrated_data.to_csv(filename, index=False)
    elif name == "baseline_integrated_data":    
        filename = self.foldername / f"{self.filename}_baseline_integrated.csv"
        self.baseline_integrated_data.to_csv(filename, index=False)
    else:
        raise ValueError("Invalid integrated data attribute")
    


def readcsv(self, name="integrated_data"):
    if name == "integrated_data":
        filename = self.foldername / f"{self.filename}_integrated.csv"
        self.integrated_data = pd.read_csv(filename)
        return self.integrated_data
    elif name == "baseline_integrated_data":
        filename = self.foldername / f"{self.filename}_baseline_integrated.csv"
        self.baseline_integrated_data = pd.read_csv(filename)
        return self.baseline_integrated_data
    else:
        raise ValueError("Invalid integrated data attribute")         
"""
Save waveform data to binary file.
Serializes full waveform dataset into a structured binary format
for fast reloading.
Takes:
    name : str
        - "waveforms" → saves raw waveforms
        - "baseline_waveforms" → saves baseline-corrected waveforms
Data layout:
    1. j indices (int32)
    2. i indices (int32)
    3. waveform arrays for all channels (float32):
        - PMT_in / SiPMin_in / SiPMout_in
        - PMT_out / SiPMin_out / SiPMout_out
Each waveform block has shape:
    (N_positions, repetitions, frame_length)
Output:
    .bin file in experiment folder
Raises:
    ValueError if name is invalid
Returns:
    None
"""
def savebin(self, name = "waveforms"):
    if name == "waveforms":
        waveforms = self.waveforms
    elif name == "baseline_waveforms":
        waveforms = self.baseline_wf
    elif name == "baseline_integrated_data_eventwise":    
        waveforms = self.baseline_integrated_data_eventwise # in this case not really waveforms, but the integrall over the individual waveforms                
    else:
        raise ValueError("Invalid waveforms attribute")
    # Save waveforms into one binary file
    j_arr = np.array(waveforms["j"], dtype=np.int32)     # shape (ni * nj,)
    i_arr = np.array(waveforms["i"], dtype=np.int32)      # shape (ni * nj,)
    PMT_LEDin_all = np.stack(waveforms["PMT_in"]).astype(np.float32) # shape (ni*nj, rep, frame_length_max)``
    SiPMin_LEDin_all = np.stack(waveforms["SiPMin_in"]).astype(np.float32)  # shape (ni*nj, rep, frame_length_max)
    SiPMout_LEDin_all = np.stack(waveforms["SiPMout_in"]).astype(np.float32)  # shape (ni*nj, rep, frame_length_max)
    PMT_LEDout_all = np.stack(waveforms["PMT_out"]).astype(np.float32) # shape (ni*nj, rep, frame_length_max)
    SiPMin_LEDout_all = np.stack(waveforms["SiPMin_out"]).astype(np.float32)  # shape (ni*nj, rep, frame_length_max)
    SiPMout_LEDout_all = np.stack(waveforms["SiPMout_out"]).astype(np.float32) # shape (ni*nj, rep, frame_length_max)
    #if waveforms["PMT_in_ref"]:  
    #    PMT_LEDin_ref = np.stack(waveforms["PMT_in_ref"]).astype(np.float32) # shape (ni*nj, rep, frame_length_max)``
    #    SiPMout_LEDin_ref = np.stack(waveforms["SiPMout_in_ref"]).astype(np.float32)  # shape (ni*nj, rep, frame_length_max)
    #    PMT_LEDout_all = np.stack(waveforms["PMT_out_ref"]).astype(np.float32) # shape (ni*nj, rep, frame_length_max)
    #    SiPMin_LEDout_all = np.stack(waveforms["SiPMin_out_ref"]).astype(np.float32)  # shape (ni*nj, rep, frame_length_max)            
    filename = self.foldername/ f"{self.filename}_{name}.bin"
    with open(filename, "wb") as f:
            j_arr.tofile(f)   # shape (ni*nj)
            i_arr.tofile(f)   # shape (ni*nj)
            PMT_LEDin_all.tofile(f)  # shape (ni*nj, rep, frame_length_max)
            SiPMin_LEDin_all.tofile(f)  # shape (ni*nj, rep, frame_length_max)
            SiPMout_LEDin_all.tofile(f)  # shape (ni*nj, rep, frame_length_max)
            PMT_LEDout_all.tofile(f)  # shape (ni*nj, rep, frame_length_max)
            SiPMin_LEDout_all.tofile(f)  # shape (ni*nj, rep, frame_length_max)
            SiPMout_LEDout_all.tofile(f)  # shape (ni*nj, rep, frame_length_max)
            #if waveforms["PMT_in_ref"]:   
            #    PMT_LEDin_ref.tofile(f)
            #    SiPMout_LEDin_ref.tofile(f)
            #    PMT_LEDout_all.tofile(f)
            #    SiPMin_LEDout_all.tofile(f)   
"""
Read waveform binary data from file.
Loads waveform arrays for all detector channels together with
their corresponding scan coordinates (i, j).
Takes:
    i : int
        Multiplier for number of scan positions.
    name : str
        Target waveform set:
            - "waveforms"
            - "baseline_waveforms"
Returns:
    dict
        Dictionary containing waveform arrays and scan indices.
Stores:
    self.waveforms
    or
    self.baseline_wf
"""
def readbin(self, i = 1, name="waveforms"):
        filename = self.foldername / f"{self.filename}_{name}.bin"
        if self.nj == 0:
            N = self.ni*i  
            if self.integrated_data["j"][0] == 999:
                N =  self.ni*i+1
            if self.integrated_data["j"][0] == 999 and self.integrated_data["j"].iloc[-1] == 9999:
                N =  self.ni*i+2
        else:
            N = self.nj*self.ni*i  
            if self.integrated_data["j"][0] == 999:
                N =  self.nj*self.ni*i+1
            if self.integrated_data["j"][0] == 999 and self.integrated_data["j"].iloc[-1] == 9999:
                N =  self.nj*self.ni*i+2    

        if name == "baseline_integrated_data_eventwise":    
            frame_length = 1    
        else: 
            frame_length = self.frame_length_max  

        rep = self.rep        

        shape = (N, rep, frame_length)
        with open(filename, "rb") as f:
            j_arr = np.fromfile(f, dtype=np.int32, count=N)
            i_arr = np.fromfile(f, dtype=np.int32, count=N)
            PMT_LEDin_all     = np.fromfile(f, dtype=np.float32, count=np.prod(shape)).reshape(shape)
            SiPMin_LEDin_all  = np.fromfile(f, dtype=np.float32, count=np.prod(shape)).reshape(shape)
            SiPMout_LEDin_all = np.fromfile(f, dtype=np.float32, count=np.prod(shape)).reshape(shape)
            PMT_LEDout_all    = np.fromfile(f, dtype=np.float32, count=np.prod(shape)).reshape(shape)
            SiPMin_LEDout_all = np.fromfile(f, dtype=np.float32, count=np.prod(shape)).reshape(shape)
            SiPMout_LEDout_all= np.fromfile(f, dtype=np.float32, count=np.prod(shape)).reshape(shape)
        assert PMT_LEDin_all.shape == (N, rep, frame_length)
        assert j_arr.dtype == np.int32
        assert PMT_LEDin_all.dtype == np.float32

        data = {
            "j": j_arr,
            "i": i_arr,
            "PMT_in": PMT_LEDin_all,
            "SiPMin_in": SiPMin_LEDin_all,
            "SiPMout_in": SiPMout_LEDin_all,
            "PMT_out": PMT_LEDout_all,
            "SiPMin_out": SiPMin_LEDout_all,
            "SiPMout_out": SiPMout_LEDout_all,
        }
        if name == "waveforms":
            self.waveforms = data
        elif name == "baseline_waveforms":
            self.baseline_wf = data
        elif name == "baseline_integrated_data_eventwise":  
            self.baseline_integrated_data_eventwise = data     
        else:
            raise ValueError("Invalid waveforms attribute")
        return data
"""
Load a WOMqc object including original and baseline corrected waveforms and corresponding integrated data.
Steps:
    1. Reads metadata using `read_metadata()`.
    2. Creates the WOMqc object.
    3. Loads waveform binary data using `readbin()`.
    4. Loads integrated CSV data using `readcsv()`.
Additionally attempts to load:
    - baseline-corrected waveforms
    - baseline-integrated data
If the baseline data files do not exist,
they are automatically generated using:
    create_baseline_data()
Takes:
    filename : str
        Base filename of the measurement.
And additional arguments for baseline data generation:
    window=(20, 0, 100), (integration window length, start index, end index)
    sigma= 0 (if > 0, applies smoothing before baseline correction),
    smooth_method=2 (running average), 2 (gaussian)            
Returns:
    WOMqc
        Fully initialized WOMqc object containing:
            - raw waveforms
            - integrated data
            - baseline-corrected waveforms
            - baseline-integrated data
"""
@classmethod
def load(cls, filename, redo=False, window=(20, 0, 100), sigma=0, smooth_method=2, bin_file = True, Vset = 40.7, dVset = 0.1, dT = 0.1):
    """
    Load a WOMqc object from metadata + stored data.
    """
    md = cls.read_metadata(filename)
    obj = cls(**md, load_existing=True)
    obj.readcsv()
    
    if bin_file:
        obj.readbin()
        # Try loading baseline data
        try:
            obj.readbin(name="baseline_waveforms")
            obj.readbin(name="baseline_integrated_data_eventwise")
            obj.readcsv(name="baseline_integrated_data")
            obj.apply_sipm_corrections(Vset = Vset, dVset=dVset, dT = dT)
            obj.calculate_eventwise_ratios()        
        # If files do not exist -> generate them
        except Exception:            
            print("Baseline data not found. Generating...")
            obj.create_baseline_data(window=window, sigma=sigma, smooth_method=smooth_method)
            obj.calculate_eventwise_ratios()
            obj.apply_sipm_corrections(Vset = Vset, dVset=dVset, dT = dT)
            obj.calculate_eventwise_ratios()
        if redo:
            print("Redoing baseline correction with new parameters...")
            obj.create_baseline_data(window=window, sigma=sigma, smooth_method=smooth_method)
    return obj
