import sys
import os
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_file(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as file:
            return file.read()


def preprocess_text(text):
    """文本预处理：分词并去除停用词"""
    words = jieba.lcut(text)
    print("分词结果：", words)  # 调试信息
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    filtered_words = [word for word in words if word not in stop_words]
    print("过滤后结果：", filtered_words)  # 调试信息
    return ' '.join(filtered_words)


def calculate_similarity(text1, text2):
    """计算两篇文本的余弦相似度"""
    if not text1 or not text2:
        print("警告：文本为空，无法计算相似度")
        return 0.0
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]


def is_writable(path):
    """检查路径是否可写"""
    try:
        with open(path, 'w') as f:
            f.write('test')
        os.remove(path)
        return True
    except IOError:
        return False


def main():
    # 检查命令行参数是否正确
    if len(sys.argv) != 4:
        print("Usage: python paper_check.py <original_file_path> <plagiarized_file_path> <output_file_path>")
        sys.exit(1)

    # 从命令行参数获取文件路径
    original_file_path = sys.argv[1]
    plagiarized_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    # 检查文件是否存在
    if not os.path.exists(original_file_path):
        print(f"文件不存在：{original_file_path}")
        sys.exit(1)
    if not os.path.exists(plagiarized_file_path):
        print(f"文件不存在：{plagiarized_file_path}")
        sys.exit(1)

    # 检查输出路径是否可写
    if not is_writable(output_file_path):
        print(f"无法写入文件：{output_file_path}")
        sys.exit(1)

    # 读取文件内容
    original_text = read_file(original_file_path)
    plagiarized_text = read_file(plagiarized_file_path)

    # 检查文件内容是否为空
    if not original_text or not plagiarized_text:
        print("警告：输入文件为空")
        sys.exit(1)

    # 文本预处理
    original_processed = preprocess_text(original_text)
    plagiarized_processed = preprocess_text(plagiarized_text)

    # 计算相似度
    similarity = calculate_similarity(original_processed, plagiarized_processed)

    # 输出结果到文件，保留两位小数
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(f"{similarity:.2f}")


if __name__ == "__main__":
    main()

# 文件末尾添加一个空行
