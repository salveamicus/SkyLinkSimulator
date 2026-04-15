import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

from cosmicbeats import CosmicBeats

CONFIG_FILE = "CosmicBeats/configs/oneweb/config.json"
TIME_INTERVAL_SEC = 15
MAX_TIMEPOINTS_PER_FILE = 1000
DEFAULT_TOTAL_DAYS = 7


def calculate_positions_for_step(satellites, current_time):
    return np.array([s.get_Position(current_time).to_tuple() for s in satellites], dtype=np.float64)


def build_file_indices(total_timepoints, start_file_index=0, end_file_index=None):
    total_files = (total_timepoints + MAX_TIMEPOINTS_PER_FILE - 1) // MAX_TIMEPOINTS_PER_FILE

    if end_file_index is None:
        end_file_index = total_files - 1

    if start_file_index < 0 or end_file_index < 0:
        raise ValueError("start_file_index and end_file_index must be >= 0")
    if start_file_index > end_file_index:
        raise ValueError("start_file_index must be <= end_file_index")
    if end_file_index >= total_files:
        raise ValueError(
            f"end_file_index={end_file_index} exceeds max available index {total_files - 1} "
            f"for total_timepoints={total_timepoints}"
        )

    return list(range(start_file_index, end_file_index + 1))


def write_file(file_index, total_timepoints, output_dir, overwrite=False):
    output_path = output_dir / f"satellite_positions_{file_index}.h5"
    if output_path.exists() and not overwrite:
        return f"[SKIP] {output_path} already exists"

    cosmicbeats = CosmicBeats(CONFIG_FILE)
    satellites = cosmicbeats.get_satellite_list()
    num_satellites = len(satellites)

    start_step = file_index * MAX_TIMEPOINTS_PER_FILE
    if start_step >= total_timepoints:
        return f"[SKIP] file_index={file_index} starts outside requested range"

    steps_in_file = min(MAX_TIMEPOINTS_PER_FILE, total_timepoints - start_step)

    current_time = copy.deepcopy(cosmicbeats.start_time)
    current_time = current_time.add_seconds(cosmicbeats.time_delta * start_step)

    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        dset = f.create_dataset(
            "positions",
            shape=(steps_in_file, num_satellites, 3),
            dtype="float64",
            compression="gzip",
        )

        for t in range(steps_in_file):
            dset[t, :, :] = calculate_positions_for_step(satellites, current_time)
            global_step = start_step + t + 1
            if global_step % 10 == 0:
                print(f"Progress: {global_step}/{total_timepoints} time points processed.")
            current_time = current_time.add_seconds(cosmicbeats.time_delta)

    return f"[OK] wrote {output_path} (steps {start_step}..{start_step + steps_in_file - 1})"


def worker(slot, workers, file_indices, total_timepoints, output_dir, overwrite=False):
    assigned = [idx for idx in file_indices if idx % workers == slot]
    messages = []
    for idx in assigned:
        messages.append(write_file(idx, total_timepoints, output_dir, overwrite=overwrite))
    return slot, assigned, messages


def parse_args():
    parser = argparse.ArgumentParser(description="Multithreaded satellite position precompute generator.")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads.")
    parser.add_argument("--total_days", type=int, default=DEFAULT_TOTAL_DAYS, help="Total days to generate.")
    parser.add_argument(
        "--total_timepoints",
        type=int,
        default=None,
        help="Optional override for exact number of timepoints. If omitted, derives from total_days.",
    )
    parser.add_argument("--start_file_index", type=int, default=0, help="First chunk index to generate.")
    parser.add_argument(
        "--end_file_index",
        type=int,
        default=None,
        help="Last chunk index to generate. Defaults to max index needed for total_timepoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../data/positions/satellite_positions",
        help="Directory for satellite_positions_{index}.h5 files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.workers <= 0:
        raise ValueError("--workers must be >= 1")

    total_timepoints = args.total_timepoints
    if total_timepoints is None:
        total_timepoints = (args.total_days * 24 * 60 * 60) // TIME_INTERVAL_SEC

    file_indices = build_file_indices(
        total_timepoints,
        start_file_index=args.start_file_index,
        end_file_index=args.end_file_index,
    )

    script_dir = Path(__file__).resolve().parent
    output_dir = (script_dir / args.output_dir).resolve()

    print(f"Generating {len(file_indices)} file(s) with {args.workers} threads")
    print(f"Output directory: {output_dir}")

    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for slot in range(args.workers):
            futures.append(
                executor.submit(
                    worker,
                    slot,
                    args.workers,
                    file_indices,
                    total_timepoints,
                    output_dir,
                    args.overwrite,
                )
            )

        for future in as_completed(futures):
            slot, assigned, messages = future.result()
            print(f"Thread {slot}: assigned file indices {assigned}")
            for msg in messages:
                print(msg)

    print("Done.")


if __name__ == "__main__":
    main()
