import torch
from torch.nn import functional as F


def cosine_similarity(vector1, vector2):
    # 计算两个向量的余弦相似度
    dot_product = torch.dot(vector1.flatten(), vector2.flatten())
    norm1 = torch.norm(vector1)
    norm2 = torch.norm(vector2)
    return dot_product / (norm1 * norm2)


def separation_func(vector, N=19):
    # 计算19个向量之间的余弦相似度之和
    similarity_sum = 0
    for i in range(N):
        for j in range(i + 1, N):
            similarity_sum += cosine_similarity(vector[:, i], vector[:, j])
    return similarity_sum


def gradient(vector, class_list, ignore_id=255):
    if ignore_id in class_list:
        class_list.remove(ignore_id)
    device = vector.device
    # 计算目标函数关于参数的梯度
    grad = torch.zeros(vector.shape).to(device)
    length = len(class_list)
    for i in range(length):
        for j in range(i + 1, length):
            dot_product = torch.dot(vector[:, class_list[i]].flatten(), vector[:, class_list[j]].flatten())
            norm1 = torch.norm(vector[:, class_list[i]])
            norm2 = torch.norm(vector[:, class_list[j]])
            gradient_i = (vector[:, class_list[j]] - dot_product * vector[:, class_list[i]] / (norm1 ** 2)) / norm2
            gradient_j = (vector[:, class_list[i]] - dot_product * vector[:, class_list[j]] / (norm2 ** 2)) / norm1
            grad[:, class_list[i]] += gradient_i
            grad[:, class_list[j]] += gradient_j
    return grad


def gradient_descent(vector, class_list, learning_rate, num_iterations):
    vector_new = vector.clone()
    for iteration in range(num_iterations):
        grad = gradient(vector_new, class_list)
        vector_new -= learning_rate * grad
    return vector_new


def norm(vector):  # 计算每个向量的模长
    norms = torch.norm(vector, dim=1)
    # 对每个向量按照维度1进行归一化
    normalized_vector = vector / norms.unsqueeze(1)
    return normalized_vector


def separation(seg_queue_new, seg_queue_old=None, n=19 * 18, best_ratio=0):
    if seg_queue_old is not None:
        feats_new = torch.cat([torch.mean(f, dim=1, keepdim=True) for f in seg_queue_new], dim=1)
        feats_new = F.normalize(feats_new, dim=0)
        separate_ratio_new = (torch.matmul(torch.transpose(feats_new, 0, 1), feats_new)).sum() + n
        if separate_ratio_new < best_ratio + 1:
            return True, separate_ratio_new
        feats_old = torch.cat([torch.mean(f, dim=1, keepdim=True) for f in seg_queue_old], dim=1)
        feats_old = F.normalize(feats_old, dim=0)
        separate_ratio = (torch.matmul(torch.transpose(feats_old, 0, 1), feats_old)).sum() + n
        if separate_ratio_new < separate_ratio:
            return True, separate_ratio_new, feats_new
        else:
            return False, separate_ratio, feats_old
    else:
        feats_new = torch.cat([torch.mean(f, dim=1, keepdim=True) for f in seg_queue_new], dim=1)
        feats_new = F.normalize(feats_new, dim=0)
        separate_ratio_new = (torch.matmul(torch.transpose(feats_new, 0, 1), feats_new)).sum() + n
        return True, separate_ratio_new, feats_new


def main():
    # 假设有一些输入数据，这里用随机生成的数据作为示例
    num_classes = 19
    num_features = 256
    num_segments = 100
    seg_queue_new = [torch.randn(num_features, num_segments) for _ in range(num_classes)]

    # 初始化向量
    vector_init = torch.randn(num_features, num_classes)

    # 设置学习率和迭代次数
    learning_rate = 0.01
    num_iterations = 1000

    # 运行梯度下降优化过程
    optimized_vector = gradient_descent(vector_init, list(range(num_classes)), learning_rate, num_iterations)

    separ = separation_func(optimized_vector)
    # 输出优化后的向量
    print("优化后的向量:")
    print(optimized_vector)
    print('separ', separ)

if __name__ == "__main__":
    main()