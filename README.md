# Temporal Convolutional Network (TCN)

Python implementaion of a Temporal Convolutional Network using deep learning framework pytorch.

## Quick start

```python
import src

looking_back_window = 7
forecast_horizon = 3
input_features = 1

net = src.TCN(
    looking_back_window,
    forecast_horizon,
    input_features
)
net
```

```txt
TCN(
  (m): Sequential(
    (0): ResidualBlock(
      (res): Conv1d(1, 3, kernel_size=(1,), stride=(1,))
      (m): Sequential(
        (0): ConstantPad1d(padding=(2, 0), value=0)
        (1): Conv1d(1, 3, kernel_size=(3,), stride=(1,))
        (2): BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU()
        (4): Dropout(p=0, inplace=False)
        (5): ConstantPad1d(padding=(2, 0), value=0)
        (6): Conv1d(3, 3, kernel_size=(3,), stride=(1,))
        (7): BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU()
        (9): Dropout(p=0, inplace=False)
      )
    )
    (1): ResidualBlock(
      (res): Conv1d(3, 1, kernel_size=(1,), stride=(1,))
      (m): Sequential(
        (0): ConstantPad1d(padding=(4, 0), value=0)
        (1): Conv1d(3, 3, kernel_size=(3,), stride=(1,), dilation=(2,))
        (2): BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU()
        (4): Dropout(p=0, inplace=False)
        (5): ConstantPad1d(padding=(4, 0), value=0)
        (6): Conv1d(3, 1, kernel_size=(3,), stride=(1,), dilation=(2,))
        (7): BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
)
```

## TODO

- [ ] Add technical description about TCNs to README

## Papers

- [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)
- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)
- [Temporal Convolutional Networks Applied to Energy-Related Time Series Forecasting](https://www.mdpi.com/2076-3417/10/7/2322)
- [Temporal Convolutional Networks for Anomaly Detection in Time Series](https://iopscience.iop.org/article/10.1088/1742-6596/1213/4/042050)
- [Temporal Convolutional Networks for the Advance Prediction of ENSO](https://www.nature.com/articles/s41598-020-65070-5)
- [Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction](https://arxiv.org/abs/2106.09305)
- [Trellis Networks for Sequence Modeling](https://arxiv.org/abs/1810.06682)

## References

- https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
- https://timeseriesai.github.io/tsai/models.TCN.html
- https://paperswithcode.com/sota/time-series-forecasting-on-etth1-24
