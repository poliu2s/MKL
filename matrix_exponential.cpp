#include <iostream>
#include <mkl.h>
#include <math.h>

// Matrix exponential function calculation
// Implementation by Po Liu using Intel's Math Kernel Library in C++
// Algorithm based on paper published by Arsigny (in same folder)

using namespace std;

double factorial(int n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

double* matrix_exponential(double* matrix)
{
	int accuracy = 10;
	int one = 1;
	int zero = 0;
	int sxtn = 16;
	int four = 4;

	// Scaling
	int N = 4;

	//M_small = M/(2^N);
	double* M_small = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	for(int i = 0; i < 16; i++) M_small[i] = matrix[i] / pow(2.0, (double)N);

	// Exp part
	double* m_exp = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* m_exp1 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	for(int i = 0; i < 16; i++) m_exp[i] = 0.0;
	for(int i = 0; i < 16; i+=5) m_exp[i] = 1.0;

	double* M_power = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* M_power1 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	for(int i = 0; i < 16; i++) M_power[i] = M_small[i];

	double* tmpM1 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);

	for(int i = 1; i < accuracy; i++) {

		//m_exp = m_exp + M_power/factorial(i);
		for(int x = 0; x < 16; x++) tmpM1[x] = M_power[x] / factorial(i);
		vdAdd(sxtn, m_exp, tmpM1, m_exp);

		//M_power = M_power * M_small;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, 1.0, M_power, 4, M_small, 4, 0.0, M_power1, 4);
		for(int x = 0; x < 16; x++) M_power[x] = M_power1[x];

	}

	// Squaring step
	for(int i = 0; i < N; i++) {
		// m_exp = m_exp*m_exp;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, 1.0, m_exp, 4, m_exp, 4, 0.0, m_exp1, 4);
		for(int x = 0; x < 16; x++) m_exp[x] = m_exp1[x];
	}
	return m_exp;
}


int main()
{	
	// Testing matrix exponential calculation
	int temp_array_size = 9;

	double* A = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* B = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);

	A[0] = 1.0;
	A[1] = 1.0;
	A[2] = 0.0;
	A[3] = 0.0;
	A[4] = 0.0;
	A[5] = 2.0;
	A[6] = 0.0;
	A[7] = 0.0;
	A[8] =-1.0;
	A[9] = 0.0;
	A[10] = 0.0;
	A[11] = 0.0;
	A[12] = 0.0;
	A[13] = 0.0;
	A[14] = 0.0;
	A[15] = 0.0;

	B = matrix_exponential(A);

	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++)
			cout << B[i*4+j] << " ";
		cout << endl;
	}
	getchar();

	return 0;

}