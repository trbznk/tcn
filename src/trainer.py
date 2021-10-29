import time
import torch
import torch.optim as optim
import torch.nn as nn


class Trainer:
    def __init__(self, device="auto"):
        if device == "auto":
            self.device = self.available_device()
        else:
            self.device = device

    def available_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, model, data_loader, epochs=10, lr=0.001):
        model = model.to(self.device)
        criterion_1 = nn.MSELoss()
        criterion_2 = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        start_time = time.time()
        for epoch in range(epochs):
            for phase in ["train", "dev"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                    
                running_loss = 0.0

                for inputs, targets in data_loader[phase]:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion_1(outputs, targets)
                    l1 = criterion_2(outputs, targets)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                    running_loss += l1.item()

                mean_loss = running_loss/len(data_loader[phase])
                print(f"({epoch+1}) {phase} \t l1={mean_loss:.2f}")
   
        end_time = time.time()
        duration = int(end_time-start_time)
        print(f"trained in {duration} seconds")

    def predict(self, model, data_loader):
        model.eval()

        all_inputs, all_targets, all_outputs, y_true, y_pred = [], [], [], [], []

        for inputs, targets in data_loader["dev"]:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = model(inputs)

            all_inputs.append(inputs.tolist())
            all_targets.append(targets.tolist())
            all_outputs.append(outputs.tolist())

            y_true += targets[:, [0]].view(-1).tolist()
            y_pred += outputs[:, [0]].view(-1).tolist()

        return all_inputs, all_targets, all_outputs, y_true, y_pred
