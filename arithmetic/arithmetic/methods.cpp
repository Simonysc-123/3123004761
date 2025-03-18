#include"common.h"
#include"calculate.h"

// 生成随机数
int getRandomNumber(int min, int max) {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_int_distribution<> distrib(min, max);
	return distrib(gen);
}

// 生成随机运算符
char getRandomOperator() {
	char operators[] = { '+', '-', '*', '/' };
	return operators[getRandomNumber(0, 3)];
}

// 生成随机自然数或真分数
std::string getRandomNumberOrFraction(int range) {
	if (getRandomNumber(0, 1) == 0) {
		return std::to_string(getRandomNumber(1, range - 1));
	}
	else {
		int numerator = getRandomNumber(1, range - 1);
		int denominator = getRandomNumber(1, range - 1);
		if (numerator >= denominator) {
			std::swap(numerator, denominator);
		}
		return std::to_string(numerator) + "/" + std::to_string(denominator);
	}
}

// 生成表达式
std::string generateExpression(int range) {
	int numOperators = getRandomNumber(1, 3);
	std::string expression = getRandomNumberOrFraction(range);
	for (int i = 0; i < numOperators; ++i) {
		char op = getRandomOperator();
		std::string nextNumber = getRandomNumberOrFraction(range);
		expression += " " + std::string(1, op) + " " + nextNumber;
	}
	return expression;
}

// 计算表达式的值
std::string calculateExpression(const std::string& expression) {
	// 这里可以使用更复杂的表达式解析和计算逻辑
	// 为了简化，这里只实现简单的加减乘除
	std::istringstream iss(expression);
	double result = 0;
	char op;
	std::string token;
	iss >> result;
	while (iss >> op >> token) {
		double num = std::stod(token);
		switch (op) {
		case '+': result += num; break;
		case '-': result -= num; break;
		case '*': result *= num; break;
		case '/': result /= num; break;
		}
	}
	return std::to_string(result);
}

// 生成题目和答案
void generateQuestions(int numQuestions, int range, const std::string& questionFile, const std::string& answerFile) {
	std::ofstream qFile(questionFile);
	std::ofstream aFile(answerFile);
	std::set<std::string> uniqueExpressions;

	for (int i = 1; i <= numQuestions; ++i) {
		std::string expression;
		do {
			expression = generateExpression(range);
		} while (uniqueExpressions.count(expression) > 0);

		uniqueExpressions.insert(expression);
		qFile << expression << " =\n";
		aFile << calculateExpression(expression) << "\n";
	}
}

// 评分功能
void gradeAnswers(const std::string& questionFile, const std::string& answerFile, const std::string& gradeFile) {
	std::ifstream qFile(questionFile);
	std::ifstream aFile(answerFile);
	std::ofstream gFile(gradeFile);

	std::vector<int> correct, wrong;
	std::string question, answer, expected;
	int index = 1;

	while (std::getline(qFile, question) && std::getline(aFile, answer)) {
		expected = calculateExpression(question.substr(0, question.find(" =")));
		if (answer == expected) {
			correct.push_back(index);
		}
		else {
			wrong.push_back(index);
		}
		++index;
	}

	gFile << "Correct: " << correct.size() << " (";
	for (int i = 0; i < correct.size(); ++i) {
		gFile << correct[i] << (i < correct.size() - 1 ? ", " : "");
	}
	gFile << ")\n";

	gFile << "Wrong: " << wrong.size() << " (";
	for (int i = 0; i < wrong.size(); ++i) {
		gFile << wrong[i] << (i < wrong.size() - 1 ? ", " : "");
	}
	gFile << ")\n";
}