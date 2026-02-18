import torch
import torch.nn as nn
import torch.optim as optim
from data_transformation import calculate_labels, create_features_2
from models.model import AnomalyModel

# Source: https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/
# Source: https://www.datacamp.com/tutorial/introduction-to-autoencoders

class Autoencoder(nn.Module):
    """ Class for the autoencoder module"""
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
class AutoencoderModel(AnomalyModel):
    """ Class for Autoencoder model"""
    
    def run_model(self, X_train : torch.Tensor, X_test : torch.Tensor, epochs: int) :
        """ 
        Trains the autoencoder on the training data and returns the anomaly scores for the test data.
        
        Parameters:
        - X_train: the training data (clean data)
        - X_test: the test data (contaminated data)
        - epochs: the number of epochs to train the model 

        Returns:
        - anomaly scores for the test data 
        """
        torch.manual_seed(42)

        model = Autoencoder(X_train.shape[1])
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-8)

        # Training
        for epoch in range(epochs):
            reconstruction = model(X_train)
            train_loss = criterion(reconstruction, X_train)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            print(f'Training: Epoch {epoch+1}, Loss: {train_loss}')


        # # Testing
        # reconstruction = model(X_test)
        # loss = criterion(reconstruction, X_test)
        # error = torch.mean((reconstruction - X_test) ** 2, dim=1)
        # return error



    def get_results(self):
        results = {}
        clean_dfs, contaminated_dfs = self.load_datasets()

        for i in range(len(contaminated_dfs)):
            node = contaminated_dfs[i]['node'].iloc[0] # get node number (should be the same for all rows inside one dataframe)
            node = str(node)

            X_train = create_features_2(clean_dfs[i], self.config.disinfectant.value, self.config.window_size)
            X_test = create_features_2(contaminated_dfs[i], self.config.disinfectant.value, self.config.window_size)

            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)

            # TODO : handle multiple contaminants, for now only one contaminant is handled
            y_true = calculate_labels(contaminated_dfs[i], self.config.contaminants[0].value, self.config.window_size)

            self.run_model(X_train, X_test, 20)
            # results[node] = {"y_true": y_true,"y_pred": anomaly_scores }
            results[node] = {"y_true": "1", "y_pred": "1"}
        
        return results