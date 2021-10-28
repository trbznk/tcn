import src
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from .data import Timeseries


LOOKING_BACK_WINDOW = 48
FORECAST_HORIZON = 24
BATCH_SIZE = 64
DEV_SIZE = 0.1
LR = 0.001
DROPOUT = 0
HIDDEN_CHANNELS_FACTOR = 1


def get_ett(looking_back_window=None, forecast_horizon=None, num_workers=None, batch_size=None):
    assert looking_back_window
    assert forecast_horizon
    assert num_workers is not None
    assert batch_size

    df = pd.read_csv("data/ETTh1.csv", parse_dates=["date"])
    df.index = df["date"]
    df.index.name = "t"
    df = df.drop(["date"], axis=1)
    df = df[["OT"]]
    df["OT"] = (df["OT"]-df["OT"].mean())/df["OT"].std()
    split_idx = int(len(df)*0.9)
    df_train, df_dev = df[:split_idx], df[split_idx:]
    df_train.shape, df_dev.shape

    dataset = {
    "train": Timeseries(df_train, "OT", looking_back_window=LOOKING_BACK_WINDOW, forecast_horizon=FORECAST_HORIZON),
    "dev": Timeseries(df_dev, "OT", looking_back_window=LOOKING_BACK_WINDOW, forecast_horizon=FORECAST_HORIZON)
    }
    data_loader = {
        "train": DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True, pin_memory=True),
        "dev": DataLoader(dataset["dev"], batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    }

    return dataset, data_loader


dataset, data_loader = get_ett()
net = src.TCN( 
    LOOKING_BACK_WINDOW,
    FORECAST_HORIZON,
    dataset["train"].features, 
    dropout=DROPOUT,
    hidden_channels_factor=HIDDEN_CHANNELS_FACTOR
)

print(net.amount_parameters)
print(net)

trainer = src.Trainer()
trainer.fit(net, data_loader, epochs=100)

y_true, y_pred = trainer.predict(net, data_loader)

fig, axs = plt.subplots(2, figsize=(20, 12))
zoom_idx, zoom_size = 500, 24*7

axs[0].plot(y_true, label="y_true")
axs[0].plot(y_pred, label="y_pred")
axs[0].legend()
axs[1].plot(y_true[zoom_idx:zoom_idx+zoom_size], label="y_true")
axs[1].plot(y_pred[zoom_idx:zoom_idx+zoom_size], label="y_pred")
axs[1].legend()
plt.show()
