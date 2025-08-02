import torch
import einx

def demonstrate_einx_get_at():
    """演示 einx.get_at 的各种用法"""
    
    print("=== einx.get_at 使用示例 ===\n")
    
    # ===== 基本用法 =====
    print("1. 基本索引操作:")
    
    # 创建一个简单的2D tensor
    x = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])
    print(f"原始tensor x:\n{x}")
    print(f"形状: {x.shape}")
    
    # 获取特定位置的元素
    indices = torch.tensor([0, 2, 1])  # 要获取的行索引
    result1 = einx.get_at("[index] features -> selected features", x, index=indices)
    print(f"\n获取行 [0, 2, 1]:\n{result1}")
    
    # ===== 多维索引 =====
    print("\n2. 多维索引操作:")
    
    # 3D tensor示例
    x_3d = torch.randn(2, 3, 4)  # (batch, seq, features)
    print(f"3D tensor形状: {x_3d.shape}")
    
    # 从每个batch中获取特定序列位置
    seq_indices = torch.tensor([[0, 2], [1, 0]])  # 每个batch选择2个位置
    result2 = einx.get_at("batch [seq_idx] features -> batch new_seq features", 
                          x_3d, seq_idx=seq_indices)
    print(f"选择特定序列位置后的形状: {result2.shape}")
    
    # ===== 在注意力机制中的应用 =====
    print("\n3. 注意力机制中的top-k选择:")
    
    # 模拟注意力分数
    attention_scores = torch.tensor([[0.1, 0.8, 0.3, 0.9, 0.2],
                                     [0.7, 0.1, 0.9, 0.4, 0.6]])
    values = torch.randn(2, 5, 64)  # (batch, seq, hidden)
    
    # 获取top-k注意力位置
    k = 3
    _, top_indices = torch.topk(attention_scores, k, dim=1)
    print(f"Top-{k} 注意力位置: {top_indices}")
    
    # 使用get_at获取对应的values
    top_values = einx.get_at("batch [seq_idx] hidden -> batch topk hidden", 
                             values, seq_idx=top_indices)
    print(f"Top-k values形状: {top_values.shape}")
    
    # ===== 词汇表索引 =====
    print("\n4. 词汇表embedding查找:")
    
    # 模拟embedding表
    vocab_size, embed_dim = 1000, 128
    embeddings = torch.randn(vocab_size, embed_dim)
    
    # 输入token ids
    token_ids = torch.tensor([[1, 15, 234, 56],
                              [789, 2, 444, 123]])
    
    # 获取对应的embeddings
    token_embeddings = einx.get_at("[vocab] embed -> batch seq embed", 
                                   embeddings, vocab=token_ids)
    print(f"Token embeddings形状: {token_embeddings.shape}")
    
    # ===== 高级用法：条件索引 =====
    print("\n5. 条件索引 - 根据mask选择:")
    
    # 创建数据和mask
    data = torch.randn(2, 6, 32)  # (batch, seq, features)
    mask = torch.tensor([[True, False, True, True, False, True],
                         [False, True, True, False, True, True]])
    
    # 获取mask为True的位置的索引
    valid_indices = []
    for i in range(mask.shape[0]):
        valid_idx = torch.where(mask[i])[0]
        valid_indices.append(valid_idx)
    
    # 由于每个batch的有效长度不同，我们需要处理变长序列
    max_valid = max(len(idx) for idx in valid_indices)
    padded_indices = torch.full((mask.shape[0], max_valid), -1, dtype=torch.long)
    
    for i, idx in enumerate(valid_indices):
        padded_indices[i, :len(idx)] = idx
    
    print(f"有效位置索引:\n{padded_indices}")
    
    # 使用get_at获取有效位置的数据（注意：-1索引需要特殊处理）
    # 在实际使用中，通常会先过滤掉-1或使用其他方法
    
    # ===== 比较传统索引方法 =====
    print("\n6. 与传统索引方法对比:")
    
    x_compare = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices_compare = torch.tensor([0, 2])
    
    # 传统方法
    traditional = x_compare[indices_compare]
    print(f"传统索引结果:\n{traditional}")
    
    # einx方法
    einx_result = einx.get_at("[index] features -> selected features", 
                              x_compare, index=indices_compare)
    print(f"einx.get_at结果:\n{einx_result}")
    
    print(f"结果相同: {torch.equal(traditional, einx_result)}")


def rope_example_with_get_at():
    """在RoPE中使用get_at的示例"""
    print("\n=== RoPE中使用get_at的示例 ===")
    
    # 假设我们有预计算的cos/sin缓存
    max_seq_len = 100
    d_k = 64
    
    # 预计算的cos/sin表 (max_seq_len, d_k//2)
    cos_cache = torch.randn(max_seq_len, d_k // 2)
    sin_cache = torch.randn(max_seq_len, d_k // 2)
    
    # 输入token位置
    token_positions = torch.tensor([[0, 1, 5, 10], [2, 3, 7, 8]])
    
    # 使用get_at根据位置获取对应的cos/sin值
    cos_vals = einx.get_at("[pos] dim -> batch seq dim", 
                           cos_cache, pos=token_positions)
    sin_vals = einx.get_at("[pos] dim -> batch seq dim", 
                           sin_cache, pos=token_positions)
    
    print(f"Token positions: {token_positions}")
    print(f"Retrieved cos shape: {cos_vals.shape}")
    print(f"Retrieved sin shape: {sin_vals.shape}")


if __name__ == "__main__":
    demonstrate_einx_get_at()
    rope_example_with_get_at()