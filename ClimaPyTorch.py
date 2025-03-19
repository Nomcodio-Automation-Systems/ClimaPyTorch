import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
import time
import pandas as pd
from Options import Options
from ClimaNet import ClimaNet
from ClimaDataSet import ClimateDataset
from RTools import RTools


options = Options()

# Load CSV Data
""" 
Load the data from a CSV file and return a list of tuples, where each tuple contains
an input tensor and a target tensor.
@param file_path: The path to the CSV file.
"""
def load_csv(file_path):
    data = []
    
    # Read the CSV file and skip the first row (header)
    df = pd.read_csv(file_path, skiprows=1, header=None)  # No header assumed after skipping
    
    # Ensure all columns are numeric and handle missing values
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, replace non-numeric with NaN
    df = df.fillna(0)  # Replace NaNs with 0 (or you could use df.dropna())
    
    # Iterate over rows to extract inputs and targets
    for _, row in df.iterrows():
        # Extract target (3rd column, index 2) and inputs (columns 4-7, indices 3-6)
        targets = torch.tensor(row.iloc[2], dtype=torch.float32).unsqueeze(0)  # Target is in column 3
        inputs = torch.tensor(row.iloc[3:7].values, dtype=torch.float32).unsqueeze(0)  # Inputs are in columns 4-7
        
        data.append((inputs, targets))
    
    return data

# Runtime Logger
"""
Log the runtime of the script to a file and print the elapsed time.
@param start_time: The start time of the script.
@param output_file: The file to write the runtime to.
"""
def log_runtime(start_time, output_file="runtime.txt"):
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"Runtime: {hours} hours, {minutes} minutes, {seconds} seconds")
    with open(output_file, "w") as f:
        f.write(f"Runtime: {hours} hours, {minutes} minutes, {seconds} seconds")

# Main Training Loop
def main():
    start_time = time.time()
    torch.manual_seed(1337)
    env = os.getcwd()
    options.update_device()
    print("Starting training...")
    print(f"Using {'CUDA' if options.device == torch.device('cuda') else 'CPU'}")
    print(f"Current working directory: {os.getcwd()}")

    net = ClimaNet(4, 50, 50, 50, 1).to(options.device)
   
    options.info_file_path = "temp-co2csv8.csv"
    options.learning_rate = 0.001

    data = load_csv(options.info_file_path)
    train_set = ClimateDataset(data)
    train_loader = DataLoader(train_set, batch_size=options.train_batch_size, shuffle=True, num_workers=options.num_workers)

    optimizer = Adam(net.parameters(), lr=options.learning_rate)
    stop_training = False

    for epoch in range(1, options.iterations + 1 if options.dont_stop_training else options.iterations + 1):
        r2_total, div_total, loss_total = 0.0, 0.0, 0.0
        for batch_index, (inputs, targets) in enumerate(train_loader, start=1):
            inputs, targets = inputs.to(options.device), targets.to(options.device)
            optimizer.zero_grad()

            predictions = net(inputs)
            loss = RTools.r2_dd_loss(predictions, targets, dynamic=2, size_average=True)
            r2 = RTools.r2_score(predictions, targets).item()
            div = RTools.divergence(predictions, targets).item()

            loss_total += loss.item()
            r2_total += r2
            div_total += div

            loss.backward()
            optimizer.step()

            if epoch % options.log_interval == 0  and batch_index == 2:
                print(f"Epoch: {epoch} | Batch: {batch_index} | Loss: {loss_total / batch_index} | R2: {r2_total / batch_index} | Div: {div_total / batch_index}%")
                

                loss_total = loss_total / batch_index
                r2_total = r2_total / batch_index
                div_total = div_total / batch_index
                
                if loss_total < 0.2 and div_total > 20.0:
                    print("Divergence too high! Stopping training.")
                    stop_training = True
                    break
                if r2_total > 0.90 and div_total < 10.0:
                    print("R2 is high enough! Saving model.")
                    if options.save_model:
                        torch.save(net.state_dict(), "net.pt")
                    stop_training = True
                    break

        if stop_training:
            break

    print("Training finished!")
    log_runtime(start_time)

if __name__ == "__main__":
    main()