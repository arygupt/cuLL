# Quantization

Reference code (read *after* writing yours): `../book.cu/7_quant`

Start a kernel: `cp ../common/template.cu name.cu`  ·  run it: `book make run-name`

## To write

- [ ] fp32_int8.cu, int8_uint8.cu, fp32_int4.cu: quantize/dequantize
- [ ] tensorwise.cu, channelwise.cu, groupwise.cu, blockwise.cu: scale granularity
- [ ] dynamic_static.cu, calibrator.cu
- [ ] awq.cu (9.3)
- [ ] Bonus: an INT8 GEMV using `__dp4a`. Your card supports it
