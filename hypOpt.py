import torch
import gpytorch
from share import privacy_preserving_avg_consensus
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math
import tikzplotlib
from matplotlib.lines import Line2D
import time

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def consensus_based_gradient(X_parts, Y_parts, G, config, Topt, eta, decay):
    """
    Runs Protocol 4

    Parameters
    ----------
    X_Parts : 
        Training input data
    Y_Parts :
        Training output data   
    G : Graph
    config: class
        Scale factors, modulus, and weight matrix
    Topt : int
        Maximum number of iterations for hyperparameter optimization
    eta : float
        Learning rate
    decay : float
        Decay rate of the learning rate

    Returns
    -------
        GP Models for each agent with the obtained hyperparameters
    """
    # Number of agents
    M = len(X_parts)
    # Hyperparameters = [length scale, signal variance, noise varaince]
    scales = torch.tensor([1.0, 1.0, 0.2])  # different magnitudes for each of the 3 elements
    theta_estimates = [torch.abs(torch.randn(3)) * scales +0.01 for _ in range(M)] # Randomly initialize

    history = []
    loss_sum = []

    models = []
    likelihoods = []

    # Create GP Models with the initial hyperparameters
    for i in range(M):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(X_parts[i], Y_parts[i], likelihood)
        models.append(model)
        likelihoods.append(likelihood)

    start = time.perf_counter()

    for _ in range(Topt):
        eta *= decay # decay learning rate
        intermediates = [] 
        loss_sum_tmp = 0

        for i in range(M):
            model = models[i]
            likelihood = likelihoods[i]

            model.train()
            likelihood.train()
            # Assing current hyperparameter estimate
            with torch.no_grad():
                model.covar_module.base_kernel.raw_lengthscale.copy_(
                    model.covar_module.base_kernel.raw_lengthscale_constraint.inverse_transform(theta_estimates[i][0].detach())
                )
                model.covar_module.raw_outputscale.copy_(
                    model.covar_module.raw_outputscale_constraint.inverse_transform(theta_estimates[i][1].detach())
                )
                likelihood.noise_covar.raw_noise.copy_(
                    likelihood.noise_covar.raw_noise_constraint.inverse_transform(theta_estimates[i][2].detach())
                )
            model.zero_grad()
            likelihood.zero_grad()

            # Compute log marginal likelihood 
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            output = model(X_parts[i])
            loss = mll(output, Y_parts[i])
            loss_sum_tmp += loss.item()
            loss.backward()

            # Get gradients
            grad_l = model.covar_module.base_kernel.raw_lengthscale.grad.item()
            grad_f = model.covar_module.raw_outputscale.grad.item()
            grad_n = likelihood.raw_noise.grad.item()

            # take gradient step
            grad = torch.tensor([grad_l, grad_f, grad_n])
            update = theta_estimates[i] + eta * grad
            intermediates.append(update)


        # Take consensus step for each hyperparameter
        intermediates_np = np.stack([param.detach().numpy() for param in intermediates])  # shape (M, 3)

        consensus_result = []
        for dim in range(intermediates_np.shape[1]):
            z_ini = intermediates_np[:, dim]
            z_history = privacy_preserving_avg_consensus(z_ini, G, config, 1) # T=1
            consensus_final = z_history[-1]
            consensus_result.append(consensus_final)

        consensus_result = np.stack(consensus_result, axis=1)  # shape (M, 3)

        # Update hyperparameter updates
        theta_estimates = [
            torch.tensor(consensus_result[i, :]).clone().detach().requires_grad_()
            for i in range(M)
        ]

        current_estimates = np.array([theta_estimates[i].detach().numpy() for i in range(M)])  # (M, 3)
        history.append(current_estimates)
        loss_sum.append(loss_sum_tmp)

    end = time.perf_counter()   
    elapsed = (end - start) * 1000  

    print(f"Total elapsed time for Protocol~4: {elapsed:.0f} ms")


    plot_consensus_based_gradient(history, Topt, loss_sum, M)

    
    models = [model.eval() for model in models]
    likelihoods = [likelihood.eval() for likelihood in likelihoods]

    return models, likelihoods

def plot_consensus_based_gradient(history, Topt, loss_sum, M):
    colors = plt.cm.get_cmap('tab10', M)
    markers = ['o', 's', 'D', '^', 'v']

    history = np.array(history)  # shape: (Topt, M, 2)
    iterations = range(Topt)
    loss_values = [loss.item() if hasattr(loss, 'item') else loss for loss in loss_sum]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Lengthscale subplot
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel(r"$\theta_\ell$")
    ax1.set_title(r'$\theta_l$ Estimates')
    for agent in range(M):
        ax1.plot(iterations, history[:, agent, 0], color=colors(agent),
                 marker=markers[agent % len(markers)], alpha=0.7,
                 markevery=5, 
                 linestyle='-', linewidth=2, label=f'Agent {agent+1}')
    ax1.grid(True)

    # Outputscale subplot
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel(r"$\theta_s$")
    ax2.set_title(r'$\theta_s$ Estimates')
    for agent in range(M):
        ax2.plot(iterations, history[:, agent, 1], color=colors(agent),
                 marker=markers[agent % len(markers)], alpha=0.7,
                 markevery=5, 
                 linestyle='-', linewidth=2, label=f'Agent {agent+1}')
    ax2.grid(True)

    # Log Marginal Likelihood subplot
    ax3.plot(loss_values, color='black', linewidth=2)
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel(r"$\widehat{LMM}$")
    ax3.set_title('Log Marginal Likelihood Estimates')
    ax3.grid(True)
    # Export as TikZ
    tikzplotlib.save("hyperparameter.tex")
    plt.tight_layout()
    plt.show()

