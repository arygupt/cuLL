# Lab log

One entry per kernel version. Predict before you run. Peak: 192 GB/s, ~5000 GFLOP/s FP32.

<!--
## gemv v2: warp per row, shuffle reduction   (ch 6.2)
predicted: memory-bound, ~150 GB/s
measured:  151 GB/s (79%)   [M=4096 K=4096]
ncu:       memory throughput 81%, achieved occupancy 62%
why:       coalesced loads vs v1's strided ones
next:      float4 loads
-->
