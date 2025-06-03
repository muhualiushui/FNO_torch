
import torch
import matplotlib.pyplot as plt
import numpy as np




def tensor_to_display(tensor, is_label=False):
    """
    将一个 tensor (B, C, D, H, W) 或 (B, C, H, W) 转换为可视化的 numpy array。
    对于 label 多通道，取 argmax。
    """
    tensor = tensor[0]  # 取 batch 第一个

    if is_label:
        if tensor.shape[0] > 1:  # 多通道 → 单通道标签
            tensor = torch.argmax(tensor, dim=0, keepdim=False)
        else:
            tensor = tensor[0]  # 单通道
    else:
        if tensor.dim() == 4:  # 非标签图像 [C, D, H, W]
            tensor = tensor[0]  # 只取第一个通道
        elif tensor.dim() == 3: # 非标签图像 [C, H, W]
            tensor = tensor[0]  # 只取第一个通道
            
    return tensor.cpu().float().numpy()


def select_slice(tensor, is_label=False):
    """
    选择中心 z-slice，假设输入 shape 是 (B, C, D, H, W)
    """
    img = tensor_to_display(tensor, is_label=is_label)

    if img.ndim == 3:  # [D, H, W]
        z_index = int(img.shape[0] *0.4)
        return img[z_index]
    elif img.ndim == 2:  # 已经是 [H, W]
        return img
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")



def visualize(visual_dict, save_path, num_classes):

    titles_row1 = ['image', 'image']
    titles_row2 = ['label', 'pred']


    rows = 1
    if any(k in visual_dict for k in titles_row2):
        rows = 2

    total_cols = len(titles_row1)
    # fig, axes = plt.subplots(2, total_cols, figsize=(4 * total_cols, 8))
    fig, axes = plt.subplots(rows, total_cols, figsize=(4 * total_cols, 4 * rows))

    def render_slice(ax, slice_img, is_label=False):
        if not is_label:
            norm_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
            rgb = np.stack([norm_img]*3, axis=-1)
        else:
            rgb = np.zeros(slice_img.shape + (3,))
            colors = np.array([
                [0, 0, 0], [1, 0, 0], [0, 1, 0],
                [0, 0, 1], [1, 1, 0], [1, 0, 1],
            ])
            for cls in range(1, min(num_classes, len(colors))):
                rgb[slice_img == cls] = colors[cls]
        ax.imshow(rgb)
        ax.axis('off')

    # for row_idx, titles in enumerate([titles_row1, titles_row2]):
    #     for col_idx, title in enumerate(titles):
    #         ax = axes[row_idx, col_idx]
    #         img = visual_dict[title]  # 已是 numpy array, [H, W]
    #         render_slice(ax, img, is_label=('label' in title or 'pred' in title))
    #         ax.set_title(title)

    # === Row 1: Image content (always exists) ===
    for col_idx, key in enumerate(titles_row1):
        if key in visual_dict:
            ax = axes[col_idx] if rows == 1 else axes[0, col_idx]
            render_slice(ax, visual_dict[key], is_label=False)
            ax.set_title(key)
    # === Row 2: Labels or predictions (optional) ===
    if rows == 2:
        for col_idx, key in enumerate(titles_row2):
            if key in visual_dict:
                ax = axes[1, col_idx]
                render_slice(ax, visual_dict[key], is_label=True)
                ax.set_title(key)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


