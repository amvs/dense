#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#%%

class ImprovedGaussianMixModel(nn.Module):
    def __init__(self, n_features, n_components=2):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components

        self.log_weights = nn.Parameter(torch.zeros(n_components))
        self.means = nn.Parameter(torch.randn(n_components, n_features))
        self.log_stdevs = nn.Parameter(torch.zeros(n_components, n_features))

    def forward(self, x):
        weights = F.softmax(self.log_weights, dim=0)
        stdevs = F.softplus(self.log_stdevs)

        # Compute log probabilities for each component
        log_probs = []
        for i in range(self.n_components):
            dist = torch.distributions.Normal(self.means[i], stdevs[i])
            log_prob = dist.log_prob(x.unsqueeze(1)).sum(-1) + torch.log(weights[i])
            log_probs.append(log_prob)

        # Stack log probabilities
        log_probs = torch.stack(log_probs, dim=-1)

        # Apply log-sum-exp trick
        max_log_prob = torch.max(log_probs, dim=-1, keepdim=True)[0]
        log_sum_exp = max_log_prob + torch.log(torch.sum(torch.exp(log_probs - max_log_prob), dim=-1, keepdim=True))

        return log_sum_exp.squeeze(-1)

    def sample(self, n_samples):
        weights = F.softmax(self.log_weights, dim=0)
        stdevs = F.softplus(self.log_stdevs)

        # Sample component indices
        component_indices = torch.multinomial(weights, n_samples, replacement=True)

        # Sample from selected components
        samples = torch.normal(
            self.means[component_indices],
            stdevs[component_indices]
        )

        return samples

#%%
if __name__ == "__main__":
    n_components = 1
    n_features = 2
    gmm = ImprovedGaussianMixModel( n_features=n_features, n_components=n_components)
    train_means = torch.randn( (n_components,n_features))
    train_stdevs = (torch.rand( (n_components,n_features)) + 1.0) * 0.25
    train_weights = torch.rand( n_components)
    ind_dists = torch.distributions.Independent( torch.distributions.Normal( train_means, train_stdevs), 1)
    mix_weight = torch.distributions.Categorical( train_weights)
    train_dist = torch.distributions.MixtureSameFamily( mix_weight, ind_dists)

    train_samp = train_dist.sample( [2000])
    valid_samp = torch.rand( (4000, 2)) * 8 - 4.0

    #%%

    max_iter = 2000
    features = train_samp #.to( 'cuda')

    optim = torch.optim.Adam( gmm.parameters(),  lr=5e-4)
    metrics = {'loss':[]}

    for i in range( max_iter):
        optim.zero_grad()
        loss = - gmm(  features)
        loss.mean().backward()
        optim.step()
        metrics[ 'loss'].append( loss.mean().item())
        print( f"{i} ) \t {metrics[ 'loss'][-1]:0.5f}", end=f"{' '*20}\r")
        if metrics[ 'loss'][-1] < 0.1:
            print( "---- Close enough")
            break
        if len( metrics[ 'loss']) > 300 and np.std( metrics[ 'loss'][-300:]) < 0.0005:
            print( "---- Giving up")
            break
        if i % 100 == 0:
            print(i, metrics['loss'][-1])
    print( f"Min Loss: {np.min( metrics[ 'loss']):0.5f}")


    #%% 

    # Usage example
    n_features = 2
    n_components = 3
    model = ImprovedGaussianMixModel(n_features, n_components).to('cuda')

    # Generate some random data
    x = torch.randn(1000, n_features).to('cuda')

    # Compute log probabilities
    log_probs = model(x)

    # Sample from the model
    samples = model.sample(1000)