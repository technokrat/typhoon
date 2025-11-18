import numpy as np

from typhoon import rainflow, goodman_transform, summed_histogram


def main() -> None:
    # Simple waveform with a few cycles
    waveform = np.array([-20.0, 20.0, -10.0, 10.0, -20.0], dtype=np.float32)

    cycles, peaks = rainflow(waveform)
    print("Rainflow cycles:", cycles)
    print("Peaks:", peaks)

    # Example Goodman parameters
    M = 0.2
    M2 = M / 3.0

    ersatz = goodman_transform(cycles, M, M2)
    print("Goodman equivalent amplitudes (Ersatzamplitudenkollektiv):")
    for amp, count in sorted(ersatz.items()):
        print(f"  S_a,ers = {amp:.3f}: count = {count}")

    hist = summed_histogram(ersatz)
    print("\nSummed histogram (descending S_a,ers):")
    for s_a_ers, cumulative in hist:
        print(f"  S_a,ers = {s_a_ers:.3f}: cumulative count = {cumulative:.1f}")


if __name__ == "__main__":
    main()
