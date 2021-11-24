import src
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

from torch.utils.data import DataLoader

EPOCHS = 100
LOOKING_BACK_WINDOW = 30
FORECAST_HORIZON = 7
BATCH_SIZE = 64
DEV_SIZE = 0.1
LR = 0.001
DROPOUT = 0
HIDDEN_CHANNELS_FACTOR = 1
FORECAST_COL = "temperature"
INDEX_COL = "date"
DATA_PATH = "./data/temperatures.csv"


def get_data():
    df = pd.read_csv(DATA_PATH, parse_dates=[INDEX_COL])
    df.index = df[INDEX_COL]
    df = df.drop([INDEX_COL], axis=1)

    df[FORECAST_COL] = (df[FORECAST_COL]-df[FORECAST_COL].mean())/df[FORECAST_COL].std()
    split_idx = int(len(df)*0.9)
    df_train, df_dev = df[:split_idx], df[split_idx:]
    df_train.shape, df_dev.shape

    dataset = {
        "train": src.Timeseries(df_train, FORECAST_COL, looking_back_window=LOOKING_BACK_WINDOW, forecast_horizon=FORECAST_HORIZON),
        "dev": src.Timeseries(df_dev, FORECAST_COL, looking_back_window=LOOKING_BACK_WINDOW, forecast_horizon=FORECAST_HORIZON)
    }
    data_loader = {
        "train": DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True, pin_memory=True),
        "dev": DataLoader(dataset["dev"], batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    }

    return dataset, data_loader


dataset, data_loader = get_data()

net = src.TCN(
    LOOKING_BACK_WINDOW,
    FORECAST_HORIZON,
    dataset["train"].features,
    dropout=DROPOUT,
    hidden_channels_factor=HIDDEN_CHANNELS_FACTOR
)
print(f"Parameters: {net.amount_parameters}")
print(net)

trainer = src.Trainer()
trainer.fit(net, data_loader, epochs=EPOCHS, lr=LR)

all_inputs, all_targets, all_outputs, y_true, y_pred = trainer.predict(net, data_loader)

zoom_idx, zoom_size = 500, 24*7

examples = []
random_idx = random.randrange(len(all_inputs))
random_inputs = all_inputs[random_idx]
random_targets = all_targets[random_idx]
random_outputs = all_outputs[random_idx]

for i in range(0, 64, 16):
    examples.append((
        np.array(random_inputs[i]).reshape(-1)[-FORECAST_HORIZON:],
        np.array(random_targets[i]),
        np.array(random_outputs[i])
    ))

fig = plt.figure(constrained_layout=True, figsize=(20, 16))
gs = fig.add_gridspec(7, 2)

ax1 = fig.add_subplot(gs[:2, :])
ax1.plot(y_true, label="y_true")
ax1.plot(y_pred, label="y_pred")
ax1.legend()

ax2 = fig.add_subplot(gs[2, :])
ax2.plot(y_true[zoom_idx:zoom_idx+zoom_size], label="y_true")
ax2.plot(y_pred[zoom_idx:zoom_idx+zoom_size], label="y_pred")
ax2.legend()

ax31 = fig.add_subplot(gs[3, 0])
ax31.plot(examples[0][0], label="inputs")
ax31.legend()

ax32 = fig.add_subplot(gs[3, 1])
ax32.plot(examples[0][1], label="y_true")
ax32.plot(examples[0][2], label="y_pred")
ax32.legend()

ax41 = fig.add_subplot(gs[4, 0])
ax41.plot(examples[1][0], label="inputs")
ax41.legend()

ax42 = fig.add_subplot(gs[4, 1])
ax42.plot(examples[1][1], label="y_true")
ax42.plot(examples[1][2], label="y_pred")
ax42.legend()

ax51 = fig.add_subplot(gs[5, 0])
ax51.plot(examples[2][0], label="inputs")
ax51.legend()

ax52 = fig.add_subplot(gs[5, 1])
ax52.plot(examples[2][1], label="y_true")
ax52.plot(examples[2][2], label="y_pred")
ax52.legend()

ax61 = fig.add_subplot(gs[6, 0])
ax61.plot(examples[3][0], label="inputs")
ax61.legend()

ax62 = fig.add_subplot(gs[6, 1])
ax62.plot(examples[3][1], label="y_true")
ax62.plot(examples[3][2], label="y_pred")
ax62.legend()

plt.show()
