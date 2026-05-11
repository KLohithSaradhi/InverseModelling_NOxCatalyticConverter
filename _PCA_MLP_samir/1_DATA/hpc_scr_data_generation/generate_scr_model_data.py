import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from scr_model import run_scr_model


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sample_path = ROOT / "lhs_sampling.npy"
mat_path = ROOT / "NEDC_lite.mat"

samples = np.load(sample_path)

n_samples = len(samples)
max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 7))

out_data_path = RESULTS_DIR / "scr_model_data.npy"
out_time_path = RESULTS_DIR / "scr_time.npy"

save_every = 100


def run_one_sample(i):
    k_ads, k_des, k_std, k_fst, k_slw, k_oxi = samples[i]

    time_s, model_sensor_ppm = run_scr_model(
        mat_path=str(mat_path),
        k_std_0=k_std,
        k_fst_0=k_fst,
        k_slw_0=k_slw,
        k_ads_0=k_ads,
        k_des_0=k_des,
        k_nh3ox_0=k_oxi,
    )

    return i, np.asarray(time_s), np.asarray(model_sensor_ppm)


if __name__ == "__main__":
    print("samples shape:", samples.shape, flush=True)
    print("mat_path:", mat_path, flush=True)
    print("max_workers:", max_workers, flush=True)

    data = None
    time_ref = None
    n_done = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one_sample, i) for i in range(n_samples)]

        for future in tqdm(as_completed(futures), total=n_samples, desc="Running SCR model"):
            i, time_s, model_sensor_ppm = future.result()

            if data is None:
                time_ref = time_s
                n_time = len(model_sensor_ppm)
                data = np.zeros((n_samples, n_time), dtype=float)
                np.save(out_time_path, time_ref)
                print("n_time:", n_time, flush=True)

            data[i, :] = model_sensor_ppm
            n_done += 1

            if n_done % save_every == 0:
                np.save(out_data_path, data)
                print(f"Saved {n_done}/{n_samples}", flush=True)

    np.save(out_data_path, data)
    np.save(out_time_path, time_ref)

    print("Done.", flush=True)
    print("saved:", out_data_path, flush=True)
    print("saved:", out_time_path, flush=True)
    print("data shape:", data.shape, flush=True)
