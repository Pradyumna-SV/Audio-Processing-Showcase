import time

import numpy as np

from app import ALGORITHMS, CATEGORIES, PACKAGES, PROC_MAP, SUBCATEGORIES, technique_available


def noisy_speech_like_signal(sr: int = 16000, seconds: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    times = np.arange(int(sr * seconds), dtype=np.float32) / sr
    envelope = 0.45 + 0.4 * np.sin(2 * np.pi * 2.2 * times) ** 2
    voiced = (
        0.42 * np.sin(2 * np.pi * 140 * times)
        + 0.20 * np.sin(2 * np.pi * 280 * times)
        + 0.10 * np.sin(2 * np.pi * 720 * times)
    )
    noise = 0.12 * rng.standard_normal(times.shape).astype(np.float32)
    return np.clip(envelope * voiced + noise, -1.0, 1.0).astype(np.float32)


def main() -> None:
    sr = 16000
    audio = noisy_speech_like_signal(sr)
    assert len(ALGORITHMS) >= 15, f"expected at least 15 algorithms, found {len(ALGORITHMS)}"
    assert {item["id"] for item in ALGORITHMS} == set(PROC_MAP)
    assert set(CATEGORIES) == {"DSP", "ML", "DL"}
    assert set(SUBCATEGORIES) == {"Established", "Advanced"}
    assert {
        (algorithm["category"], algorithm["subcategory"]) for algorithm in ALGORITHMS
    } == {(category, subcategory) for category in CATEGORIES for subcategory in SUBCATEGORIES}
    assert set(PACKAGES["Baseline DSP"]["ids"]) == {"highpass", "bandpass", "spectral_sub", "wiener"}

    for algorithm in ALGORITHMS:
        if not technique_available(algorithm):
            print(f"SKIP {algorithm['name']}: optional local dependency not installed")
            continue
        start = time.perf_counter()
        output = PROC_MAP[algorithm["id"]](audio, sr)
        elapsed = time.perf_counter() - start
        assert output.shape == audio.shape, f"{algorithm['name']} returned shape {output.shape}"
        assert output.dtype == np.float32, f"{algorithm['name']} returned dtype {output.dtype}"
        assert np.all(np.isfinite(output)), f"{algorithm['name']} returned non-finite values"
        assert np.max(np.abs(output)) <= 1.00001, f"{algorithm['name']} exceeded audio bounds"
        print(f"PASS {algorithm['name']}: {elapsed:.2f}s")

    short_audio = audio[:32]
    for algorithm in ALGORITHMS:
        if not technique_available(algorithm):
            continue
        output = PROC_MAP[algorithm["id"]](short_audio, sr)
        assert output.shape == short_audio.shape, f"{algorithm['name']} failed short-input shape"
        assert output.dtype == np.float32, f"{algorithm['name']} failed short-input dtype"
        assert np.all(np.isfinite(output)), f"{algorithm['name']} failed short-input finiteness"
    print("PASS short recording handling for all algorithms")


if __name__ == "__main__":
    main()
