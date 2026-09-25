import os
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from data_adapters.io_vnbd_adapter import process_io_vnbd

def main():
    s_file = 'data/io-vnbd/raw/S1/S-S1.csv'
    v_file = 'data/io-vnbd/raw/S1/V-S1.csv'
    out_file = 'data/io-vnbd/processed/S1_sumaro_format.csv'
    
    # Run adapter
    print(f"Processing data from {s_file} and {v_file}...")
    df = process_io_vnbd(s_file, v_file, out_file, outage_start_s=2000.0, outage_duration_s=60.0)
    
    # Summary stats
    duration = df['time'].max()
    distance = (df['true_speed'] * 0.1).sum()  # roughly
    speed_range = (df['true_speed'].min(), df['true_speed'].max())
    gnss_fixes = df['gnss_x'].notna().sum()
    
    print(f"Duration: {duration:.2f} s")
    print(f"Distance: {distance:.2f} m")
    print(f"Speed Range: {speed_range[0]:.2f} to {speed_range[1]:.2f} m/s")
    print(f"GNSS Fixes: {gnss_fixes}")
    
    # Plot trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(df['true_x'], df['true_y'], label='Vehicle Truth', color='black')
    plt.scatter(df['gnss_x'], df['gnss_y'], s=2, label='Smartphone GNSS', color='blue', alpha=0.5)
    plt.title('IO-VNBD S1 Trajectory')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    fig_dir = 'reports/figures'
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'io_vnbd_s1_trajectory.png')
    plt.savefig(fig_path)
    print(f"Plot saved to {fig_path}")

if __name__ == '__main__':
    main()
