import sys
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(file_path):
    """读取文件内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    """文本预处理：分词并去除停用词"""
    # 使用jieba进行分词
    words = jieba.lcut(text)
    # 简单的停用词过滤（可以根据需求扩展）
    stop_words = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def calculate_similarity(text1, text2):
    """计算两篇文本的余弦相似度"""
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

def main():
    if len(sys.argv) != 4:
        print("Usage: python paper_check.py <original_file_path> <plagiarized_file_path> <output_file_path>")
        sys.exit(1)

    original_file_path = sys.argv[1]
    plagiarized_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    # 读取文件内容
    original_text = read_file(original_file_path)
    plagiarized_text = read_file(plagiarized_file_path)

    # 文本预处理
    original_processed = preprocess_text(original_text)
    plagiarized_processed = preprocess_text(plagiarized_text)

    # 计算相似度
    similarity = calculate_similarity(original_processed, plagiarized_processed)

    # 输出结果到文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(f"{similarity:.2f}")

if __name__ == "__main__":
    main()