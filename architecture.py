import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import parameters_to_vector
from torch.nn.utils.stateless import functional_call
from torch.autograd.functional import hessian
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, depth, input_size, output_size, width=48):
        super().__init__()
        layers = [nn.Linear(input_size, width), nn.ReLU()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(width, width), nn.ReLU()])
        layers.append(nn.Linear(width, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def get_parameters(self):
        return parameters_to_vector(self.parameters())

class Architecture:
    def __init__(self, depth, input_size=784, output_size=10, width=48):
        self.depth = depth
        self.input_size = input_size
        self.output_size = output_size
        self.width = width

    def generate_initial_guesses(self, num_guesses):
        # clone N freshly-initialized networks
        return [
            [p.clone().detach()
             for p in MLP(self.depth, self.input_size, self.output_size, self.width).parameters()]
            for _ in range(num_guesses)
        ]
    
    def train(
            self, initial_guess, x, y,
            lr=0.001, tolerance=1e-8, 
            max_iters=100000,
            hutchinson_probes=45
            ):
        # 1) Reinitialize & seed from initial_guess
        self.model = MLP(self.depth,
                        self.input_size,
                        self.output_size,
                        self.width)
        self.model.train()
        with torch.no_grad():
            for p, init in zip(self.model.parameters(), initial_guess):
                p.copy_(init)

        optimizer = Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        losses = []
        for iteration in range(max_iters):
            optimizer.zero_grad()
            y_pred = self.model(x_t)
            loss = criterion(y_pred, y_t)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)
            if len(losses) > 1 and abs(losses[-1] - losses[-2]) < tolerance:
                break

        # 2) Approximate Frobenius norm of Hessian via Hutchinson
        self.model.eval()
        # get flat parameter vector with grad enabled
        p0 = parameters_to_vector(self.model.parameters())\
                .detach().requires_grad_(True)

        # define loss as a function of the flat parameters
        names, params = zip(*self.model.named_parameters())
        shapes = [p.shape    for p in params]
        numels = [p.numel()  for p in params]
        def loss_fn(p_vec):
            splits = torch.split(p_vec, numels)
            param_dict = {
                name: split.view(shape)
                for name, split, shape in zip(names, splits, shapes)
            }
            y_pred = functional_call(self.model, param_dict, x_t)
            return F.mse_loss(y_pred, y_t)

        # compute gradient at p0
        L = loss_fn(p0)
        grad = torch.autograd.grad(L, p0, create_graph=True)[0]

        # Hutchinson: E[||H v||^2] = tr(H^2)
        sum_sq = 0.0
        for _ in range(hutchinson_probes):
            v = torch.randn_like(p0)
            # directional second derivative: H v
            hv = torch.autograd.grad((grad * v).sum(), p0, retain_graph=True)[0]
            sum_sq = sum_sq + (hv * hv).sum()
        tr_H2 = sum_sq / float(hutchinson_probes)
        frob_est = torch.sqrt(tr_H2).item()

        # 3) Return model, losses, and estimated frobenius norm
        return self.model, losses, frob_est

    def get_validation_error(self, test_x, test_y):
        self.model.eval()
        with torch.no_grad():
            test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
            test_y_tensor = torch.tensor(test_y, dtype=torch.float32)
            y_pred = self.model(test_x_tensor)
            loss = F.mse_loss(y_pred, test_y_tensor)
        return loss.item()

    def run_experiment(self, num_guesses, x, y, test_x, test_y):
        initial_guesses = self.generate_initial_guesses(num_guesses)
        model_data = []

        for init in initial_guesses:
            model, losses, hessian_eigenvalues = self.train(init, x, y)
            params = parameters_to_vector(model.parameters()).detach()
            val_error = self.get_validation_error(test_x, test_y)
            model_data.append([params, losses, hessian_eigenvalues, val_error])

        return model_data
