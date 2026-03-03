from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from data_transformation import calculate_labels, create_extended_features
from models.model import AnomalyModel

# https://realpython.com/generative-adversarial-networks/#what-are-generative-adversarial-networks

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        output = self.model(x)
        return output

 
class GANModel(AnomalyModel):
    """ Class for GAN model"""
    
    def run_model(self, X_train : torch.Tensor, X_test : torch.Tensor, epochs: int) :
        """ 
        Trains the GAN on the training data and returns the anomaly scores for the test data.
        
        Parameters:
        - X_train: the training data (clean data)
        - X_test: the test data (contaminated data)
        - epochs: the number of epochs to train the model 
        """
        torch.manual_seed(42)
        latent_dim = 16
        batch_size = 1024

        generator = Generator(latent_dim, X_train.shape[1])
        discriminator = Discriminator(X_train.shape[1])
        
        criterion = nn.BCELoss()
        optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.0001, weight_decay=1e-8)
        optimizer_generator = optim.Adam(generator.parameters(), lr=0.001, weight_decay=1e-8)

        for epoch in range(epochs):
            for i in range (0, X_train.size(0), batch_size):
                train_batch = X_train[i:i+batch_size]

                real_labels = torch.ones(train_batch.size(0), 1)   
                fake_labels = torch.zeros(train_batch.size(0), 1)

                latent_space = torch.randn(train_batch.size(0), latent_dim)
                fake_samples = generator(latent_space).detach()

                all_samples = torch.cat((train_batch, fake_samples), dim=0)  
                all_labels = torch.cat((real_labels, fake_labels))

                # Training the discriminator
                discriminator.zero_grad()
                discriminated = discriminator(all_samples)
                loss_discriminator = criterion(discriminated, all_labels)
                loss_discriminator.backward()
                optimizer_discriminator.step()

                # Training the generator
                latent_space = torch.randn(train_batch.size(0), latent_dim)
                generator.zero_grad()
                generated = generator(latent_space)
                discriminated_generated = discriminator(generated)
                loss_generator = criterion(discriminated_generated, real_labels)
                loss_generator.backward()
                optimizer_generator.step()

            print(f'Training: Epoch {epoch+1}, Loss discriminator: {loss_discriminator}, Loss generator: {loss_generator}')

            
    def get_results(self):
        results = {}
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()

        for node, clean_dfs in all_clean_dfs.items():
            clean_df = pd.concat(clean_dfs)
            contaminated_dfs = all_contaminated_dfs[node]
            contaminated_df = pd.concat(contaminated_dfs)

            X_train = create_extended_features(clean_df, self.config.disinfectant.value, self.config.window_size)
            X_test = create_extended_features(contaminated_df, self.config.disinfectant.value, self.config.window_size)

            # Normalize the data 
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0)
            std[std == 0] = 1  

            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std
    
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)

            # # TODO : handle multiple contaminants, for now only one contaminant is handled
            # y_true = calculate_labels(contaminated_df, self.config.contaminants[0].value, self.config.window_size-1)
            # y_true = y_true[:len(X_test)]   
            self.run_model(X_train, X_test, 100)
            # y_pred = np.where(anomalies, -1, 1)  
            # results[node] = {"y_true": y_true, "y_pred": y_pred}

            # test_timestamps = contaminated_df.iloc[self.config.window_size:]["timestep"].values
            # self.get_plots(node, test_timestamps, X_test, reconstructions, anomalies, y_true, threshold, test_error)

        return results
    