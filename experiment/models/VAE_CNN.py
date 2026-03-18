# L'encodeur du VAE crée un embedding dans l'espace latent (juste partie encode)
# A partir de l'embedding, un CNN fully connected prédit pour chaque point de la série temporelle d'entrée si c'est une anomalie ou pas 
# Objectifs: opti (pas de decoder), pas de threshold arbitraire, meilleurs résultats ? 

from VAE import VAE, VAEModel

class VAEEncoder(VAE):
    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        return z  
    