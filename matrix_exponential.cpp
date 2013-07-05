#include <iostream>
#include <mkl.h>
#include <math.h>

// Matrix exponential function calculation
// Implementation by Po Liu using Intel's Math Kernel Library 11.0.4 in C++
// Algorithm based Arsigny's PhD thesis (PDF of specific page in same folder)

using namespace std;

double* matrix_exponential(double* matrix, double* result)
{
	int accuracy = 10;

	// Scaling
	int N = 4;

	//M_small = M/(2^N);
	double* M_small = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	for(int i = 0; i < 16; i++) M_small[i] = matrix[i] / pow(2.0, (double)N);

	// Exp part
	double* m_exp1 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* m_exp2 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	for(int i = 0; i < 16; i++) result[i] = 0.0;
	for(int i = 0; i < 16; i+=5) result[i] = 1.0;

	double* M_power = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* M_power1 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	cblas_dcopy(16, M_small, 1, M_power, 1);

	double* tmpM1 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);

	double factorial_i = 1.0;
	for(int i = 1; i < accuracy; i++) {
		factorial_i = factorial_i * i;

		//m_exp = m_exp + M_power/factorial(i);
		for(int x = 0; x < 16; x++) tmpM1[x] = M_power[x] / factorial_i;
		
		vdAdd(sxtn, result, tmpM1, result);

		//M_power = M_power * M_small;
		cblas_dcopy(16, M_power, 1, M_power1, 1);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, 1.0, M_power1, 4, M_small, 4, 0.0, M_power, 4);
	}

	
	// Squaring step
	const MKL_INT oneb = 1;
	for(int i = 0; i < N; i++) {
		// m_exp = m_exp*m_exp;
		cblas_dcopy(16, result, oneb, m_exp1, 1);
		cblas_dcopy(16, result, 1, m_exp2, 1);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, 1.0, m_exp1, 4, m_exp2, 4, 0.0, result, 4);
	}
	

	mkl_free(M_small);
	mkl_free(m_exp1);
	//mkl_free(M_power);
	//mkl_free(M_power1);
	mkl_free(tmpM1);
	mkl_free(m_exp2);
	
	return;
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

	matrix_exponential(A, B);

	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++)
			cout << B[i*4+j] << " ";
		cout << endl;
	}
	getchar();

	return 0;

}
