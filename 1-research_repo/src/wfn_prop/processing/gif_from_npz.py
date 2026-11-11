import os
import argparse
import numpy as np
import h5py
from wfn_prop.analysis import export_diffusion_gif_updated2, export_multistate_gif

def find_file(filename):
    for candidate in [filename, os.path.join("..", filename), os.path.join(os.path.dirname(__file__), "..", filename)]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot find {filename}")

def main():
    parser = argparse.ArgumentParser(description="Render animated diffusion GIFs from HDF5 simulation output.")
    parser.add_argument("--h5", type=str, default="naph_8x8x4_2state_backscat_S2init_40fs_frames.h5", help="HDF5 frames file")
    parser.add_argument("--vel", type=str, default="final_velocity_payload_8x8x4-naph.npz", help="Velocity / physics npz payload")
    parser.add_argument("--mode", type=str, choices=["coupled", "multistate", "all"], default="all",
                        help="Visualization mode: 'coupled' (2-panel), 'multistate' (4-panel), or 'all'")
    parser.add_argument("--outdir", type=str, default="gifs", help="Output directory for generated GIFs")
    args = parser.parse_args()

    h5_path = find_file(args.h5)
    vel_path = find_file(args.vel)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reading HDF5 dataset: {h5_path}")
    print(f"Reading velocity payload: {vel_path}")

    with h5py.File(h5_path, 'r') as h5_file:
        frames = h5_file['frames']
        dt = float(h5_file.attrs['dt'])
        physics = np.load(vel_path)
        q_vectors = physics['Qpts']

        base_name = os.path.splitext(os.path.basename(h5_path))[0]

        if args.mode in ("coupled", "all"):
            out_coupled = os.path.join(args.outdir, f"{base_name}_coupled.gif")
            print(f"\n--- Generating Coupled 2-Panel GIF: {out_coupled} ---")
            export_diffusion_gif_updated2(
                frames=frames,
                dt=dt,
                save_interval=2,
                length_x=100,
                length_y=100,
                grid_x=250,
                grid_y=250,
                right_panel_mode='qgrid',
                q_vectors=q_vectors,
                projection=('x', 'y'),
                filename=out_coupled
            )

        if args.mode in ("multistate", "all"):
            out_multi = os.path.join(args.outdir, f"{base_name}_multistate.gif")
            print(f"\n--- Generating State-Resolved 4-Panel GIF: {out_multi} ---")
            export_multistate_gif(
                frames=frames,
                dt=dt,
                save_interval=2,
                length_x=100,
                length_y=100,
                grid_x=250,
                grid_y=250,
                q_vectors=q_vectors,
                state_labels=['S2', 'S1'],
                projection=('x', 'y'),
                filename=out_multi,
                fps=12
            )

    print("\nGIF rendering complete.")

if __name__ == "__main__":
    main()
