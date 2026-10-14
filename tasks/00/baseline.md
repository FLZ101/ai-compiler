# Task 00: Eager vs compiled baseline

## Files

- `tasks/00/baseline.py`
- `tasks/00/baseline.md`

## Run

```bash
make run SCRIPT=tasks/00/baseline.py
```

## Run 1

- `eager: 4.969 ms/iter`
- `compiled: 2.829 ms/iter`
- `speedup: 1.76x`
- First run included TorchInductor compilation.

## Run 2

- `eager: 4.892 ms/iter`
- `compiled: 2.794 ms/iter`
- `speedup: 1.75x`
- The compiled result was reused from the Inductor cache.

## Check

- The first compiled call was slower than the warmed-up calls because it included compilation.
- The final speedup was greater than `1.0x`.
- `torch.compile` used `backend="inductor"`.
