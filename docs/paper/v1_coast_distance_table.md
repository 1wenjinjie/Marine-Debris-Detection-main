# Table 3: Marine_Debris detection by distance-to-coast bucket

**Source:** docs\paper\v1_test_per_tile.csv (BiFPN @ imgsz=1024, seed 0; deployment conf τ = 0.25; IoU ≥ 0.5)

**Coastline reference:** Natural Earth 1:50M physical/coastline

**Caveat:** 73 test tiles split into 3 buckets — per-bucket samples are small. Numbers are descriptive; do **not** claim statistical significance between buckets.

| Bucket | n tiles | n GT | TP | FP | FN | P | R | F1 | dist median (km) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coastal (<5 km) | 41 | 63 | 43 | 25 | 20 | 0.632 | 0.683 | 0.656 | 2.0 |
| nearshore (5-20 km) | 25 | 34 | 23 | 22 | 11 | 0.511 | 0.676 | 0.582 | 11.0 |
| offshore (>20 km) | 7 | 11 | 2 | 2 | 9 | 0.500 | 0.182 | 0.267 | 21.7 |
| **all (reference)** | **73** | **108** | **68** | **49** | **40** | **0.581** | **0.630** | **0.604** | 4.3 |