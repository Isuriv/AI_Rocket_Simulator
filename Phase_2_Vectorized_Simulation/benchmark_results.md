| Rockets | Plain loop | `torch.vmap` | Speedup | Peak memory (loop) | Peak memory (vmap) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.0 ms | 12.7 ms | 0.5x | <0.1 MB | <0.1 MB |
| 10 | 63.3 ms | 17.7 ms | 3.6x | <0.1 MB | <0.1 MB |
| 100 | 655.9 ms | 13.9 ms | 47.3x | <0.1 MB | <0.1 MB |
| 1,000 | 6.54 s | 14.4 ms | 454.7x | <0.1 MB | <0.1 MB |
| 10,000 | 66.00 s | 24.0 ms | 2755.0x | 5.2 MB | 1.8 MB |

Hardware: Intel(R) Xeon(R) Processor (CPU, 4 torch threads), PyTorch 2.15.0+cpu, Python 3.12.3, Linux 6.12.113+.
Times are medians of repeated forward runs (180 physics steps per rocket); memory is peak process RSS growth during the timed runs, on top of a ~262 MB baseline for Python, PyTorch and the inputs. Max difference between loop and vmap final states: 0.0e+00.
