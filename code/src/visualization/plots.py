import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_attention_pattern(attention_weights, title="Attention Pattern"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_weights.cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    return plt.gcf()

def plot_residual_decomposition(contributions, direction_name=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(range(len(contributions['attention'])), contributions['attention'])
    axes[0].set_title('Attention Contributions')
    axes[0].set_xlabel('Layer')
    axes[1].bar(range(len(contributions['mlp'])), contributions['mlp'])
    axes[1].set_title('MLP Contributions')
    axes[1].set_xlabel('Layer')
    axes[2].bar(['Embedding', 'Attn Total', 'MLP Total'],
                [contributions['embedding'], sum(contributions['attention']), sum(contributions['mlp'])])
    axes[2].set_title('Total Contributions')
    plt.suptitle(f'Residual Stream Decomposition: {direction_name}')
    plt.tight_layout()
    return fig
