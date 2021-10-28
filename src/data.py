import torch

from torch.utils.data import Dataset
from statsmodels.tsa.seasonal import seasonal_decompose


class Timeseries(Dataset):
    def __init__(self, df, y, looking_back_window=None, forecast_horizon=None, decompose=False, differentiate=False):
        assert forecast_horizon > 0
        
        self.df = df.dropna()

        if decompose:
            self.decompose()
        if differentiate:
            self.differentiate()

        self.y = y
        self.looking_back_window = looking_back_window
        self.forecast_horizon = forecast_horizon

    @property
    def features(self):
        return self.df.shape[1]
        
    def decompose(self):
        for col in self.df.columns:
            result = seasonal_decompose(self.df[col])
            self.df.loc[:, f"{col}_trend"] = result.trend
            self.df.loc[:, f"{col}_seasonal"] = result.seasonal
            self.df.loc[:, f"{col}_resid"] = result.resid

    def differentiate(self):
        for col in self.df.columns:
            self.df.loc[:, f"{col}_diff"] = self.df[col].diff()
    
    def __len__(self):
        return len(self.df)-self.looking_back_window-self.forecast_horizon+1
    
    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError(i)
        
        start, stop = i, i+self.looking_back_window
        inputs = self.df[start:stop]
        
        start, stop = stop, stop+self.forecast_horizon
        targets = self.df[start:stop]
        
        inputs, targets = inputs.T.to_numpy(), targets[self.y].to_numpy()
        inputs, targets = torch.tensor(inputs).float(), torch.tensor(targets).float() 
        
        return inputs, targets
