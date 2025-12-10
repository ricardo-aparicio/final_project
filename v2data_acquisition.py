#!/usr/bin/env python3
import numpy as np
import adi  # pyadi-iio
import time


sdr = adi.Pluto("ip:192.168.201.203")


sdr.rx_lo = int(2.442e9)       
sdr.sample_rate = int(50e6)   
sdr.rx_rf_bandwidth = int(45e6)
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = 30  


OUT_FILE = "autel_noisy01.bin"

DURATION_SEC   = 3          
CHUNK_SAMPLES  = 1_000_000   

# ----------------------------------------

def capture_samples(filename, duration_sec):
    fs = sdr.sample_rate
    target_samples = int(fs * duration_sec)

    print(f"Capturando ~{duration_sec:.2f} s "
          f"({target_samples/1e6:.2f} M muestras) en {filename}")

    samples_written = 0

    with open(filename, "wb") as f:
        t0 = time.time()
        while samples_written < target_samples:
           
            this_block = min(CHUNK_SAMPLES, target_samples - samples_written)
            sdr.rx_buffer_size = this_block
            iq = sdr.rx()

            iq_c64 = np.asarray(iq, dtype=np.complex64)
            iq_c64.tofile(f)

            samples_written += len(iq_c64)
            print(f"\r  -> {samples_written/1e6:7.2f} M muestras", end="")

        t1 = time.time()

    real_sec = samples_written / fs
    print(f"\nListo. Guardadas {samples_written} muestras "
          f"(~{real_sec:.3f} s) en {filename}. "
          f"Tiempo real: {t1-t0:.1f} s")

if __name__ == "__main__":
    capture_samples(OUT_FILE, DURATION_SEC)
