import torch
import einx
from einops import rearrange

def compare_causal_mask_methods():
    sequence_length = 4
    batch_size = 2
    num_heads = 3
    device = torch.device('cpu')
    
    # 模拟批次维度
    b = [batch_size]  # 假设只有一个批次维度
    
    print("=== 方法1：原生PyTorch (第一个代码) ===")
    seq1 = torch.arange(sequence_length)  # 注意：没有指定device
    qi1 = seq1.view(-1, 1)  # (seq_len, 1)
    ki1 = seq1.view(1, -1)  # (1, seq_len)
    causal_mask1 = qi1 >= ki1  # (seq_len, seq_len)
    
    # 手动添加维度
    for i in range(len(b) + 1):  # +1 for heads
        causal_mask1 = causal_mask1.unsqueeze(0)
    
    print(f"掩码1形状: {causal_mask1.shape}")
    print("掩码1内容 (去除批次和头维度):")
    print(causal_mask1.squeeze().int())
    
    print("\n=== 方法2：einx (第二个代码) ===")
    seq2 = torch.arange(sequence_length, device=device)
    qi2 = einx.rearrange('query -> b... 1 query 1', seq2, b=[1] * len(b))
    kj2 = einx.rearrange('key   -> b... 1 1   key', seq2, b=[1] * len(b))
    causal_mask2 = qi2 >= kj2
    
    print(f"掩码2形状: {causal_mask2.shape}")
    print("掩码2内容 (去除批次和头维度):")
    print(causal_mask2.squeeze().int())
    
    print("\n=== 结果对比 ===")
    # 为了公平比较，将mask1也移到相同设备
    causal_mask1 = causal_mask1.to(device)
    
    # 调整形状使其一致（第一种方法可能需要广播到正确的形状）
    if causal_mask1.shape != causal_mask2.shape:
        print(f"形状不同: {causal_mask1.shape} vs {causal_mask2.shape}")
        # 可能需要手动调整形状
    
    # 检查内容是否相同
    if torch.equal(causal_mask1.squeeze(), causal_mask2.squeeze()):
        print("✅ 两种方法生成的掩码内容相同")
    else:
        print("❌ 两种方法生成的掩码内容不同")
    
    return causal_mask1, causal_mask2

def show_detailed_differences():
    print("\n" + "="*60)
    print("详细差异分析:")
    print("="*60)
    
    differences = [
        {
            "方面": "设备处理",
            "方法1": "❌ 未指定device，可能导致设备不匹配",
            "方法2": "✅ 正确指定device=x.device"
        },
        {
            "方面": "代码简洁性",
            "方法1": "❌ 需要手动循环添加维度",
            "方法2": "✅ 使用einx一步到位"
        },
        {
            "方面": "可读性",
            "方法1": "✅ 逻辑清晰，容易理解",
            "方法2": "⚠️ 需要了解einx语法"
        },
        {
            "方面": "性能",
            "方法1": "✅ 原生操作，无额外依赖",
            "方法2": "⚠️ 需要einx库，但操作更高效"
        },
        {
            "方面": "维度处理",
            "方法1": "❌ 手动处理，容易出错",
            "方法2": "✅ 自动处理批次维度"
        },
        {
            "方面": "错误风险",
            "方法1": "❌ 容易忘记设备或维度处理",
            "方法2": "✅ 更加安全"
        }
    ]
    
    for diff in differences:
        print(f"\n{diff['方面']}:")
        print(f"  方法1: {diff['方法1']}")
        print(f"  方法2: {diff['方法2']}")

if __name__ == "__main__":
    compare_causal_mask_methods()
    show_detailed_differences()
    
    print("\n" + "="*60)
    print("推荐做法:")
    print("="*60)
    print("1. 修复第一个代码中的语法错误")
    print("2. 添加proper设备处理")
    print("3. 考虑使用torch.tril()作为更简洁的替代方案")
    print("4. 确保位置编码函数调用参数一致")
    print("5. 统一变量命名（qi/ki vs qi/kj）")