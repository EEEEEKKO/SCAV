import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm.auto import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_embeddings(safe_embeds, malicious_embeds, perturbed_embeds, n_layers = 32, method = 'pca'):
    safe_keys = [k for k in safe_embeds.keys()]
    malicious_keys = [k for k in malicious_embeds.keys()]
    perturbed_keys = [k for k in perturbed_embeds.keys()]

    fig = make_subplots(rows = 6, cols = 6,
                        subplot_titles = [f'Layer {i}' for i in range(1,n_layers+1)],
                        vertical_spacing=0.05,
                        horizontal_spacing=0.05)
    
    for layer_idx in tqdm(range(n_layers), desc = "Processing layers"):
        safe_layer_embeds = np.vstack([safe_embeds[k][layer_idx].cpu().numpy() for k in safe_keys])
        malicious_layer_embeds = np.vstack([malicious_embeds[k][layer_idx].cpu().numpy() for k in malicious_keys])
        perturbed_layer_embeds = np.vstack([perturbed_embeds[k][layer_idx].cpu().numpy() for k in perturbed_keys])

        all_embeds = np.concatenate([safe_layer_embeds, malicious_layer_embeds, perturbed_layer_embeds])

        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        
        reduced_embeds = reducer.fit_transform(all_embeds)

        n_safe = len(safe_keys)
        n_malicious = len(malicious_keys)
        n_perturbed = len(perturbed_keys)

        safe_reduced = reduced_embeds[:n_safe]
        malicious_reduced = reduced_embeds[n_safe:n_safe+n_malicious]
        perturbed_reduced = reduced_embeds[n_safe+n_malicious:]
        
        row = (layer_idx // 6) + 1
        col = (layer_idx % 6) + 1

        # 只在第一个子图显示图例
        show_legend = (layer_idx == 0)

        fig.add_trace(
            go.Scatter(x=safe_reduced[:, 0], y=safe_reduced[:, 1],
                      mode='markers', name='Safe',
                      marker=dict(color='blue', size=8),
                      showlegend=show_legend),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(x=malicious_reduced[:, 0], y=malicious_reduced[:, 1],
                      mode='markers', name='Malicious',
                      marker=dict(color='red', size=8),
                      showlegend=show_legend),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(x=perturbed_reduced[:, 0], y=perturbed_reduced[:, 1],
                      mode='markers', name='Perturbed',
                      marker=dict(color='purple', size=8),
                      showlegend=show_legend),
            row=row, col=col
        )
    
    # 更新布局，将图例放在图像上方
    fig.update_layout(
        height=2400,
        width=2400,
        title_text=f"Embedding Distributions Across Layers ({method.upper()})",
        showlegend=True,
        legend=dict(
            orientation="h",  # 水平放置图例
            yanchor="bottom",
            y=1.02,  # 将图例放在图像上方
            xanchor="center",
            x=0.5,   # 图例居中
            bgcolor='rgba(255,255,255,0.8)',  # 半透明白色背景
            bordercolor='rgba(0,0,0,0.2)',    # 浅灰色边框
            borderwidth=1
        )
    )

    fig.update_xaxes(title_text="Dimension 1")
    fig.update_yaxes(title_text="Dimension 2")

    return fig

    