#include <iostream>
#include <vector>
#include <omp.h>
#include <random>

using namespace std;

const int p = 5;
const int m = 4;
const int one = 1;
const int q = 6;

const int processorElements = 1;

vector < vector < double >> matrixA;
vector < vector < double >> matrixB;
vector < vector < double >> matrixE;
vector < vector < double >> matrixG;

vector < vector < double >> matrixC;

uniform_real_distribution<double> unif(-1.0, 1.0);
default_random_engine re;

// Первичное заполнение матриц
void config(vector<vector<double >> &matrix) {
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++)
		{
			matrix[i][j] = unif(re);
		}
	}
}

void configMatrix() {
	matrixA.resize(p, vector<double>(m));
	matrixB.resize(m, vector<double>(q));
	matrixE.resize(one, vector<double>(m));
	matrixG.resize(p, vector<double>(q));

	matrixC.resize(p, vector<double>(q));

	config(matrixA);
	config(matrixB);
	config(matrixE);
	config(matrixG);
}

// Функции обработки
double dDeltaFunction(double aIK, double bKJ) {
	if (aIK > bKJ)
		return bKJ;
	else
		return aIK;
}

double deltaDFunction(int currentI, int currentJ) {
	double tempResult = 1;
	for (int f = 0; f < m; f++) {
		tempResult *= (1 - dDeltaFunction(matrixA[currentI][f], matrixB[f][currentJ]));
	}
	return 1 - tempResult;
}

double abDeltaFunction(double aIK, double bKJ) {
	double tempAIK = 1 - aIK;
	if (tempAIK > bKJ)
		return tempAIK;
	else
		return bKJ;
}

double baDeltaFunction(double aIK, double bKJ) {
	double tempBKJ = 1 - aIK;
	if (aIK > tempBKJ)
		return aIK;
	else
		return tempBKJ;
}

double simpleFFunction(int currentI, int currentJ) {
	double result = 0;
	for (int f = 0; f < m; f++) {
		result += abDeltaFunction(matrixA[currentI][f],
			matrixB[f][currentJ]) * (2 * matrixE[0][f] - 1) * matrixE[0][f] +
			baDeltaFunction(matrixA[currentI][f],
				matrixB[f][currentJ]) * (1 + (4 * abDeltaFunction(matrixA[currentI][f],
					matrixB[f][currentJ]) - 2)
					* matrixE[0][f]) * (1 - matrixE[0][f]);
	}

	return result;
}

double deltaFFunction(int currentI, int currentJ) {
	double result = 1;
	for (int f = 0; f < m; f++) {
		result *= simpleFFunction(currentI, currentJ);
	}
	return result;
}

double deltaFDFunction(int currentI, int currentJ) {
	double tempResult = (deltaFFunction(currentI, currentJ) + deltaDFunction(currentI, currentJ) - 1);
	if (tempResult > 0)
		return tempResult;
	else
		return 0;
}

// Обработка
void computeMatrixC() {
//#pragma omp parallel for private(i)
	for (int i = 0; i < matrixC.size(); i++) {
//#pragma omp parallel for shared(i) private(j)
		for (int j = 0; j < matrixC[i].size(); j++) {
			matrixC[i][j] = deltaFFunction(i, j) * (3 * matrixG[i][j] - 2) * matrixG[i][j] +
				(deltaDFunction(i, j) + (4 * deltaFDFunction(i, j) - 3 * deltaDFunction(i, j)) * matrixG[i][j]) * (1 - matrixG[i][j]);
		}
	}
}

// Вывод
void printMatrix(vector < vector < double >> matrix) {
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

int main()
{
	configMatrix();

	omp_set_dynamic(false);
	omp_set_num_threads(processorElements);

	computeMatrixC();
	cout << "matrix A" << endl;
	printMatrix(matrixA);
	cout << endl << "matrix B" << endl;
	printMatrix(matrixB);
	cout << endl << "matrix E" << endl;
	printMatrix(matrixE);
	cout << endl << "matrix G" << endl;
	printMatrix(matrixG);
	cout << endl << "matrix C" << endl;
	printMatrix(matrixC);
	system("pause");
}
