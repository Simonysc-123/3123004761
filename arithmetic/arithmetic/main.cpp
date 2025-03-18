#include"common.h"
#include"calculate.h"



int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " -n <numQuestions> -r <range> | -e <exerciseFile> -a <answerFile>\n";
		return 1;
	}

	std::string mode = argv[1];
	if (mode == "-n" && argc == 5) {
		int numQuestions = std::stoi(argv[2]);
		int range = std::stoi(argv[4]);
		generateQuestions(numQuestions, range, "Exercises.txt", "Answers.txt");
	}
	else if (mode == "-e" && argc == 5) {
		std::string exerciseFile = argv[2];
		std::string answerFile = argv[4];
		gradeAnswers(exerciseFile, answerFile, "Grade.txt");
	}
	else {
		std::cerr << "Invalid arguments.\n";
		return 1;
	}

	return 0;
}
