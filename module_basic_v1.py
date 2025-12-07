"""
03/02/2025,
basic module to setup the model.
   1. the NNs for policy and value functions;
   2. sampling from the state space;
   3. plot equm functions.
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
from torch.distributions.log_normal import LogNormal
from scipy.interpolate import griddata

# Load configuration from the JSON file
class Config:
    def __init__(self, config_file):
        with open(config_file, "r") as json_file:
            self.config = json.load(json_file)

        # extract all parameters
        for key, value in self.config.items():
            setattr(self, key, value)

    def get_z_bounds(self):
        z_bounds = self.bounds.get("z", {})
        return z_bounds.get("min"), z_bounds.get("max")


class MyModel(nn.Module):
    """
    define the neural networks:
    """
    class MyReLU(nn.Module):
        def forward(self, x):
            return nn.functional.relu(x) + 1

    def __init__(self, n_input, n_p_output, n_v_output, n1_p, n2_p, n1_v, n2_v):
        super(MyModel, self).__init__()

        # Define MyReLU as a Module
        self.my_relu = self.MyReLU()

        # Define policy function
        self.policy_func = nn.Sequential(
            nn.Linear(n_input, n1_p),
            nn.ReLU(),
            nn.Linear(n1_p, n2_p),
            nn.ReLU(),
            nn.Linear(n2_p, n_p_output),
        )

        # Define value function
        self.value_func = nn.Sequential(
            nn.Linear(n_input, n1_v),
            nn.ReLU(),
            nn.Linear(n1_v, n2_v),
            nn.ReLU(),
            nn.Linear(n2_v, n_v_output)
        )

    def f_policy(self, x):
        return self.policy_func(x)

    def f_value(self, x):
        return self.value_func(x)


class DomainSampling:
    def __init__(self, ranges, device=None):
        self.ranges = ranges
        self.device = device
        self.config = Config("config_v1.json")

    def generate_samples(self, num_samples, num_k):
        '''num_k is the dim of the last columns needs to be normalized as they represent the distribution'''
        keys = ["z", "a"]
        ranges = [(self.config.bounds[key]["min"], self.config.bounds[key]["max"]) for key in keys]
        # Extend the ranges for dist_a_pdf
        for dist_a_pdf in self.config.dist_a_pdf:
            extended_min = dist_a_pdf * (1 - self.config.dist_a_band)
            extended_max = dist_a_pdf * (1 + self.config.dist_a_band)
            ranges.append((extended_min, extended_max))

        n_states = len(ranges)
        samples_tensor = torch.empty(num_samples, n_states, dtype=torch.float32, device=self.device)

        # Parameters for the lognormal distribution
        mu, sigma = self.config.mu_z, self.config.sigma_z
        all_variables = ["z", "a", "dist_a"]
        z_index = all_variables.index("z")
        z_min, z_max = self.ranges[z_index]
        log_normal = LogNormal(mu, sigma)

        # Efficient generation of the first column (z)
        z_samples = torch.zeros(num_samples, dtype=torch.float32, device=self.device)
        for i in range(num_samples):
            sample_accepted = False
            attempts = 0
            while not sample_accepted and attempts < 1000:  # Limit the number of attempts
                sample = log_normal.sample()
                if z_min <= sample <= z_max:
                    z_samples[i] = sample
                    sample_accepted = True
                attempts += 1
            if not sample_accepted:
                raise ValueError(f"Unable to generate a valid sample in {attempts} attempts.")

        samples_tensor[:, 0] = z_samples

        # Generate other columns
        for i, (lower, upper) in enumerate(ranges[1:], start=1):
            samples_tensor[:, i] = torch.rand(num_samples, dtype=torch.float32, device=self.device) * (
                        upper - lower) + lower

        # Normalize the last num_k columns
        sum_last_k = samples_tensor[:, -num_k:].sum(dim=1, keepdim=True)
        samples_tensor[:, -num_k:] /= sum_last_k

        return samples_tensor

    def generate_samples_a_pdf(self, n_batch, num_k):
        # Extract the range for the last num_k columns from the configuration
        ranges = []
        for dist_a_pdf in self.config.dist_a_pdf[-num_k:]:  # Get the last num_k distributions
            extended_min = dist_a_pdf * (1 - self.config.dist_a_band_path)
            extended_max = dist_a_pdf * (1 + self.config.dist_a_band_path)
            ranges.append((extended_min, extended_max))

        # Generate samples for the last num_k columns
        samples_tensor = torch.empty(n_batch, num_k, dtype=torch.float32, device=self.device)
        for i, (lower, upper) in enumerate(ranges):
            samples_tensor[:, i] = torch.rand(n_batch, dtype=torch.float32, device=self.device) * (
                        upper - lower) + lower

        # Normalize the samples
        sum_last_k = samples_tensor.sum(dim=1, keepdim=True)
        normalized_samples = samples_tensor / sum_last_k

        return normalized_samples

    def dist_enforce_boundaries(self, x_dist1, a_pdf_penalty):
        n_batch, n_dim = x_dist1.shape

        # Check if dist_a_pdf has the same number of elements as the columns of x_dist1
        if len(self.config.dist_a_pdf) != n_dim:
            raise ValueError("Length of dist_a_pdf must be equal to the number of columns in x_dist1")

        # Convert dist_a_pdf to a tensor and prepare for broadcasting
        dist_a_pdf_tensor = torch.tensor(self.config.dist_a_pdf, device=self.device)

        # Calculate extended_min and extended_max
        extended_min = dist_a_pdf_tensor * (1 - self.config.dist_a_band)
        extended_max = dist_a_pdf_tensor * (1 + self.config.dist_a_band)

        # Enforce boundaries for each element
        penalty_below = a_pdf_penalty * (extended_min[None, :] - x_dist1).clamp(min=0)
        penalty_above = a_pdf_penalty * (x_dist1 - extended_max[None, :]).clamp(min=0)
        total_penalty = a_pdf_penalty * (penalty_below + penalty_above).sum(dim=1).unsqueeze(-1)
        x_dist1_clamped = x_dist1.clamp(min=extended_min[None, :], max=extended_max[None, :])

        return x_dist1_clamped, total_penalty


class plot_equm_funcs:
    def __init__(self, num_samples, num_k, dist_a_mid, model, device=None):
        self.num_samples = num_samples
        self.num_k = num_k
        self.dist_a_mid = dist_a_mid
        self.model = model
        self.device = device
        self.config = Config("config_v1.json")  # Assuming Config is defined elsewhere
        self.initialize_ranges()

    def initialize_ranges(self):
        # Initialize the ranges based on the config
        keys = ["z", "a"]
        self.ranges = [(self.config.bounds[key]["min"], self.config.bounds[key]["max"]) for key in keys]

    def generate_samples_fixed_k(self):
        config = Config("config_v1.json")
        keys = ["z", "a"]
        ranges = [(config.bounds[key]["min"], config.bounds[key]["max"]) for key in keys]
        for dist_a_pdf in config.dist_a_pdf:
            extended_min = dist_a_pdf * (1 - config.dist_a_band)
            extended_max = dist_a_pdf * (1 + config.dist_a_band)
            ranges.append((extended_min, extended_max))

        n_states = len(ranges)
        samples_tensor = torch.empty(self.num_samples, n_states, dtype=torch.float32, device=self.device)

        mu, sigma = config.mu_z, config.sigma_z
        z_lower_bound, z_upper_bound = config.get_z_bounds()
        log_normal = LogNormal(mu, sigma)

        # Generate samples for each row
        for i in range(self.num_samples):
            # Generate z within bounds
            z_sample = log_normal.sample()
            while not (z_lower_bound <= z_sample <= z_upper_bound):
                z_sample = log_normal.sample()
            samples_tensor[i, 0] = z_sample

            # Generate a and first row of the remaining columns
            for j, (lower, upper) in enumerate(ranges[1:], start=1):
                samples_tensor[i, j] = torch.rand(1, dtype=torch.float32, device=self.device) * (upper - lower) + lower

        # Normalize and replicate last num_k columns
        sum_last_k = samples_tensor[0, -self.num_k:].sum()
        samples_tensor[:, -self.num_k:] = samples_tensor[0, -self.num_k:] / sum_last_k

        return samples_tensor

    def create_plot(self):
        # 1. 准备绘图数据 (Fixed Grid)
        a_min, a_max = self.config.bounds["a"]["min"], self.config.bounds["a"]["max"]
        x_a0 = torch.linspace(a_min, a_max, 100).unsqueeze(1)
        x_z0 = torch.zeros_like(x_a0) + 1.0  # 假设 z=1 (mean productivity)
        
        # 构造输入向量 [a, z, z_agg_idx, mean_a, var_a]
        # 注意：这里必须保证 aggregate state 与训练时一致，否则策略会飘
        mean_a_val = 1.0  # 假设稳态资本约为 1.0
        x_mean_a = torch.zeros_like(x_a0) + mean_a_val
        x_var_a = torch.zeros_like(x_a0) + 0.1 # 假设方差
        x_z_agg_idx = torch.zeros_like(x_a0) # 假设 z_agg index = 0
        
        # 归一化输入 (调用模型自带的归一化函数)
        # 注意：需确保 normalized_input 逻辑与训练一致
        # 这里手动拼装一个简单的输入用于定性观察
        # 如果您有封装好的 prep_input 函数更好，这里模拟最基础的输入构造
        x_input = torch.cat([x_a0, x_z0, x_z_agg_idx, x_mean_a, x_var_a], dim=1)
        
        # 归一化 (假设简单的 min-max 归一化，请根据您的 normalize_input 修改)
        x_input_norm = (x_input - a_min) / (a_max - a_min) 
        # 修正：通常只归一化资产，这里简化处理，直接传入模型预测
        # 严谨做法应调用: self.model.normalize_input(...) 如果存在
        
        # 2. 获取模型预测
        self.model.eval() # 切换到评估模式
        with torch.no_grad():
            if isinstance(self.model, torch.nn.DataParallel):
                raw_logits = self.model.module.f_policy(x_input) # 假设输入已适配
            else:
                raw_logits = self.model.f_policy(x_input)

        # 3. 计算物理量 (关键修正步骤!)
        if hasattr(self.config, 'solver_method') and self.config.solver_method == 'euler':
            # === Euler 方法逻辑 (新) ===
            # A. 计算储蓄率 s (0, 1)
            s = torch.sigmoid(raw_logits[:, 0].unsqueeze(1))
            
            # B. 计算相关价格 (r, w)
            # 使用上面设定的 mean_a_val 作为 K
            K = mean_a_val 
            L = self.config.l_bar
            Z = 1.0 # 假设 Aggregate Z = 1
            
            r = self.config.alpha * Z * (K/L)**(self.config.alpha - 1) - self.config.delta
            w = (1 - self.config.alpha) * Z * (K/L)**self.config.alpha
            
            # C. 计算总财富 Wealth
            # Wealth = (1+r)a + w*l*z
            wealth = (1 + r) * x_a0 + w * x_z0 * 1.0 # 假设 l=1
            
            # D. 计算下一期资产 a'
            x_a1 = s * wealth
            
            # E. 计算消费 c
            x_c0 = (1 - s) * wealth
            
        else:
            # === Bellman 方法逻辑 (旧) ===
            # 兼容旧模型：假设输出直接是归一化的 a'
            # 如果旧模型也没了 sigmoid，这里需补上
            pred_norm = torch.sigmoid(raw_logits[:, 0].unsqueeze(1))
            x_a1 = pred_norm * (a_max - a_min) + a_min
            
            # 估算消费 (Bellman下通常不直接算c，或者是倒算的)
            # 这里简单用 budget constraint
            K = mean_a_val
            r = self.config.alpha * (K/self.config.l_bar)**(self.config.alpha - 1) - self.config.delta
            w = (1 - self.config.alpha) * (K/self.config.l_bar)**self.config.alpha
            wealth = (1 + r) * x_a0 + w * x_z0
            x_c0 = wealth - x_a1

        # 4. 绘图 (保持不变)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # 策略函数 a'(a)
        ax[0].plot(x_a0.numpy(), x_a1.numpy(), label='Policy a\'(a)')
        ax[0].plot(x_a0.numpy(), x_a0.numpy(), 'k--', alpha=0.3, label='45 degree')
        ax[0].set_title(f"Epoch {epoch}: Asset Policy")
        ax[0].set_xlabel("Current Asset a")
        ax[0].legend()
        
        # 消费函数 c(a)
        ax[1].plot(x_a0.numpy(), x_c0.numpy(), color='orange', label='Consumption c(a)')
        ax[1].set_title(f"Epoch {epoch}: Consumption Policy")
        ax[1].set_xlabel("Current Asset a")
        ax[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'figures/scatter_policy_c.png')
        plt.close()

    def extract_state_variables(self, x_sample):
        """
        Extract state variables from the input batch.
        """
        device = self.device
        x_z = x_sample[:, 0].unsqueeze(1).to(device)
        x_a = x_sample[:, 1].unsqueeze(1).to(device)
        x_dist = x_sample[:, 2:].to(device)  # Extracts all columns from the 3rd to the last
        return x_z, x_a, x_dist

def normalize_inputs(inputs, normalization_bounds):
    '''
    Normalize the first two columns of the input by their min and max values.
    Normalize the last k columns so that their sum equals 1.
    '''
    n, total_columns = inputs.shape
    outputs = torch.zeros_like(inputs)

    # Normalize first two columns
    for i in range(2):
        key = list(normalization_bounds.keys())[i]
        min_val = normalization_bounds[key]["min"]
        max_val = normalization_bounds[key]["max"]
        outputs[:, i] = (inputs[:, i] - min_val) / (max_val - min_val)

    # Normalize last k columns by their row sum
    k_columns = inputs[:, 2:]
    row_sums = k_columns.sum(dim=1, keepdim=True)
    normalized_k_columns = k_columns / row_sums
    outputs[:, 2:] = normalized_k_columns

    return outputs


def bounded_log_normal_samples(mu, sigma, lower_bound, upper_bound, num_samples):
    """
    Draw n_samples from a log-normal distribution bounded by lower_bound and upper_bound.
    """
    log_normal = LogNormal(mu, sigma)
    samples = torch.empty(num_samples)

    for i in range(num_samples):
        sample_accepted = False
        while not sample_accepted:
            sample = log_normal.sample().item()
            if lower_bound <= sample <= upper_bound:
                samples[i] = sample
                sample_accepted = True
    return samples

def enforce_boundary(x, min_val, max_val, lambda_penalty):
    '''
    reset the variable if it exceeds its boundary
    '''
    penalty = 0
    # Calculate penalties for values below the minimum and above the maximum
    penalty_below = lambda_penalty * (min_val - x).clamp(min=0)
    penalty_above = lambda_penalty * (x - max_val).clamp(min=0)

    # Total penalty for the tensor
    total_penalty = lambda_penalty * penalty_below + (1-lambda_penalty) * penalty_above

    # Clamp the tensor values within the boundary
    x_clamped = x.clamp(min=min_val, max=max_val)

    return x_clamped, total_penalty


def contains_nan(tensor):
    return torch.isnan(tensor).any().item()

# module_basic_v1.py (在文件末尾添加以下函数)

def fischer_burmeister(a, b):
    """
    Fischer-Burmeister function: psi(a, b) = a + b - sqrt(a^2 + b^2)
    It satisfies psi(a, b) = 0 <=> a >= 0, b >= 0, ab = 0
    Used for KKT conditions in Euler equation method.
    """
    # 1e-8 added for numerical stability to prevent NaN gradients
    return a + b - torch.sqrt(a**2 + b**2 + 1e-8)

