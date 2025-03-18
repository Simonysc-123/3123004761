#include"common.h"

std::string generateExpression(int range);
std::string calculateExpression(const std::string& expression);
void generateQuestions(int numQuestions, int range, const std::string& questionFile, const std::string& answerFile);
void gradeAnswers(const std::string& questionFile, const std::string& answerFile, const std::string& gradeFile);

int getRandomNumber(int min, int max);
char getRandomOperator();
std::string getRandomNumberOrFraction(int range); 

