import pandas as pd
import numpy as np
import os
import math

def process_io_vnbd(s_file_path, v_file_path, output_path, outage_start_s=None, outage_duration_s=None):
    """
    Loads and parses S-S1.csv and V-S1.csv, aligns them, converts GPS to local ENU,
    applies GNSS outage, and saves a SUMARO format dataset.
    """
    # Load files, specify encoding since it's latin1
    s_df = pd.read_csv(s_file_path, encoding='latin1')
    v_df = pd.read_csv(v_file_path, encoding='latin1')
    
    # Strip leading/trailing spaces from column names
    s_df.columns = s_df.columns.str.strip()
    v_df.columns = v_df.columns.str.strip()
    
    # Simple alignment: trim to shorter length and align row-by-row
    min_len = min(len(s_df), len(v_df))
    s_df = s_df.iloc[:min_len].reset_index(drop=True)
    v_df = v_df.iloc[:min_len].reset_index(drop=True)
    
    # Unified time axis (assume 10Hz -> 0.1s dt)
    time_s = np.arange(min_len) * 0.1
    
    # Extract origin from first valid vehicle GPS point (or smartphone)
    # Using smartphone for consistency
    valid_gps = s_df[['GPS LATITUDE (degrees)', 'GPS LONGITUDE (degrees)']].dropna()
    if len(valid_gps) == 0:
        raise ValueError("No valid GPS data found in smartphone file.")
    lat0 = valid_gps.iloc[0]['GPS LATITUDE (degrees)']
    lon0 = valid_gps.iloc[0]['GPS LONGITUDE (degrees)']
    lat0_rad = math.radians(lat0)
    
    # Function for simple flat-earth ENU conversion
    def to_enu(lat, lon):
        east = (lon - lon0) * math.cos(lat0_rad) * 111320
        north = (lat - lat0) * 111320
        return east, north
    
    # Apply ENU conversion to smartphone GPS
    gnss_e, gnss_n = to_enu(s_df['GPS LATITUDE (degrees)'], s_df['GPS LONGITUDE (degrees)'])
    
    # Apply ENU conversion to vehicle GPS
    true_e, true_n = to_enu(v_df['Latitude (degrees)'], v_df['Longitude (degrees)'])
    
    # Create the output dataframe
    out_df = pd.DataFrame()
    out_df['time'] = time_s
    
    # Helper to find column containing a string (to handle encoding artifacts)
    def find_col(df, prefix):
        return [c for c in df.columns if prefix in c][0]

    # Raw Accelerometer (includes gravity)
    # The EKF expects this.
    out_df['accel_x'] = s_df[find_col(s_df, 'ACCELEROMETER X')]
    out_df['accel_y'] = s_df[find_col(s_df, 'ACCELEROMETER Y')]
    out_df['accel_z'] = s_df[find_col(s_df, 'ACCELEROMETER Z')]
    
    # Gyroscope mapping
    # Smartphone axes: Roll -> x, Pitch -> y, Yaw -> z
    out_df['gyro_x'] = s_df['GYROSCOPE Roll (rad/s)']
    out_df['gyro_y'] = s_df['GYROSCOPE Pitch (rad/s)']
    out_df['gyro_z'] = s_df['GYROSCOPE Yaw (rad/s)']
    
    # Smartphone GNSS ENU
    out_df['gnss_x'] = gnss_e
    out_df['gnss_y'] = gnss_n
    
    # Vehicle true ENU
    out_df['true_x'] = true_e
    out_df['true_y'] = true_n
    
    # Vehicle indicated speed (km/h to m/s)
    out_df['true_speed'] = v_df['Indicated Vehicle Speed (km/hr)'] / 3.6
    
    # Vehicle heading (degrees to radians)
    out_df['true_heading'] = np.radians(v_df['Heading (degrees)'])
    
    # Vehicle yaw rate (deg/s to rad/s)
    out_df['true_yaw_rate'] = np.radians(v_df['Yaw Rate (deg/sec)'])
    
    # Vehicle longitudinal acceleration (g to m/s^2)
    # 1 g = 9.80665 m/s^2
    out_df['true_accel_x_vehicle'] = v_df['Indicated Longitudinal Acceleration (g)'] * 9.80665
    
    # Simulated GNSS outage
    if outage_start_s is not None and outage_duration_s is not None:
        outage_end_s = outage_start_s + outage_duration_s
        outage_mask = (out_df['time'] >= outage_start_s) & (out_df['time'] <= outage_end_s)
        out_df.loc[outage_mask, 'gnss_x'] = np.nan
        out_df.loc[outage_mask, 'gnss_y'] = np.nan
        
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    
    return out_df
