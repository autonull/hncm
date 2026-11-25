# H-NCM v9: the end-to-end proof (enwik8_100m)

The model is fitted on 81 MB of the 90 MB train; the held-out 10000000 B val is encoded and decoded by the concept arm. The bitstream round-trips: **True**.

- payload: 3300098 B = 2.640 bpb
- model: 300.9 kB = 0.024 bpb over 100 MB
- recorded shared: **2.664 bpb**
- calls: 2393110; tokens: 3733841
- encode/decode: 0.02 / 0.02 MB/s
- model: 2.696; entropy 2.640; gap -0.056
