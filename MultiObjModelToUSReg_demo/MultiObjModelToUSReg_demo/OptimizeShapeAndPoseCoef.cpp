#include "Registration.h"
#include <mkl.h>
#include "stdafx.h"




//=================================================computeMultiTransformationValueAndDerivative==//
// COMPUTEMULTITRANSFORMATIONVALUEANDDERIVATIVE computes the value and the
// derivative of transformations given a basis of transformation and weights
// associated to them.
// INPUTS: Basis is 7xL, where each column is a similarity transformation presented in log-euclidean space
//		   Theta is Lx1, and is weights associated to each basis
//		   poseMean is 4x4, representing the pose mean of these basis
// OUTPUT: deriv is 4x4xL
//		   value is 4x4
void Registration::computeMultiTransformationValueAndDerivative(double* theta, double* basis, int L, 
																double* poseMean, double* deriv, double* value)
{
	int temp_array_size;
	int onei = 1;
	double zero = 0.0;
	double one = 1.0;

	//cout << "computeMultiTransformationValueAndDerivative" << endl;

	double* Tr = (double*)mkl_malloc(4 * 4 * L * sizeof(double), 64);
	double* transform_tmp = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* transform_tmp2 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	
	// value = eye(4);
	//value = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* value2 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	for(int i = 0; i < 16; i++) value[i] = 0.0;
	for(int i = 0; i < 16; i+=5) value[i] = 1.0;
	

	for(int i = 0; i < L; i++) {
		
		//for(int p = 0; p < 16; p++) cout << basis[i+p] << " ";
		/*
		cout << L << endl;		
		cout << "basis" << endl;
		for(int p = 0; p < 7; p++) {
			for(int o = 0; o < 2; o++)
				cout << basis[i+ p*L + o] << "," << i+ p*L + o << "  ";
			cout << endl;
		}
		
		getchar();
		*/


		// Tr(:,:,l) = real(expm(theta(l)*getTransformationMatrixFromVector(basis(:,l))));
		//MultiObjModelMKL::getTransformationMatrixFromVector(&basis[i], L, transform_tmp);
		MultiObjModelMKL::getTransformationMatrixFromVector(&basis[i], 31, transform_tmp);

		//cout << "inner1" << endl;
		/*
		cout << "transform_tmp" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << transform_tmp[p*4 + o] << " ";
			cout << endl;
		}
		getchar();
		*/

		//cout << "theta[i]: " << theta[i] << endl;
		//getchar();

		cblas_dscal(16, theta[i], transform_tmp, onei);
		
		//cout << "inner2" << endl;
		/*
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << transform_tmp[p*4 + o] << "  ";
			cout << endl;
		}
		cout << "transform_tmp" << endl;
		getchar();
		*/

		// fails here
		MultiObjModelMKL::matrix_exponential(transform_tmp, transform_tmp2);		
		cblas_dcopy(16, transform_tmp2, 1, &Tr[i*16], 1);
		

		//cout << "inner3" << endl;
		/*
		cout << "value" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << value[p*4 + o] << " ";
		cout << endl;
		}
		getchar();
		*/
		
		// value = value * Tr(:,:,l);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, one, value, 4, &Tr[i*16], 4, zero, value2, 4);
		
		//cout << "inner4" << endl;
		/*
		cout << "value2" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << value2[p*4 + o] << " ";
		cout << endl;
		}
		getchar();
		*/
		cblas_dcopy(16, value2, 1, value, 1);
		//cout << "inner5" << endl;
	}
	
	
	
	/* completely correct
	cout << "TR: " << endl;
	for(int i = 0; i < 4*2; i++) {
		for(int j = 0; j < 4; j++)
			cout << Tr[i*4 + j] << " ";
		cout << endl;

	}
	getchar();
	*/

	/*
	cout << "value" << endl;
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++)
			cout << value[i*4 + j] << " ";
		cout << endl;

	}
	getchar();
	*/
	//cout << "inner5a" << endl;

	// value = value * poseMean;	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				4, 4, 4, one, value, 4, poseMean, 4, zero, value2, 4);
	//cout << "inner5b" << endl;
	cblas_dcopy(16, value2, 1, value, 1);
	
	//cout << "inner6"  << endl;
	/*
	cout << "value" << endl;
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++)
			cout << value[i*4 + j] << " ";
		cout << endl;

	}
	getchar();
	*/
	

	//deriv = (double*)mkl_malloc(4 * 4 * L * sizeof(double), 64);
	for(int i = 0; i < 4 * 4 * L; i++) deriv[i] = 0.0;
	
	
	double* deriv_intermd = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	for(int i = 0; i < 4 * 4; i++) deriv_intermd[i] = 0.0;

	
	for(int i = 0; i < L; i++) {

		// deriv(:,:,l) = eye(4);
		for(int x = 0; x < 4 * 4; x++) deriv[i*16 + x] = 0.0;
		for(int x = 0; x < 4 * 4; x+=5) deriv[i*16 + x] = 1.0;
		
		for(int p = 0; p < i; p++) {
			// deriv(:,:,l) = deriv(:,:,l) * Tr(:,:,p);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						4, 4, 4, one, &deriv[i*16], 4, &Tr[p*16], 4, zero, deriv_intermd, 4);
			cblas_dcopy(16, deriv_intermd, 1, &deriv[i*16], 1);
			
		}
		
		/*
		cout << "deriv-i" << endl;
		for(int p = 0; p < 4 * 2; p++) {
			for(int o = 0; o < 4; o++)
				cout << deriv[p*4 + o] << " ";
			cout << endl;

		}
		getchar();
		*/
		
		

		//MultiObjModelMKL::getTransformationMatrixFromVector(&basis[i], L, transform_tmp);
		MultiObjModelMKL::getTransformationMatrixFromVector(&basis[i], 31, transform_tmp);
 
		/* Correct
		cout << "transform_tmp" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << transform_tmp[p*4 + o] << " ";
			cout << endl;

		}
		getchar();
		*/

		// deriv(:,:,l) = deriv(:,:,l) * getTransformationMatrixFromVector(basis(:,l)) * Tr(:,:,l);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, one, &deriv[i*16], 4, transform_tmp, 4, zero, deriv_intermd, 4);
		
		/*
		cout << "deriv_intermd" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << deriv_intermd[p*4 + o] << " ";
			cout << endl;

		}
		getchar();
		*/

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, one, deriv_intermd, 4, &Tr[i*16], 4, zero, &deriv[i*16], 4);

		/*
		cout << "deriv-i2" << endl;
		for(int p = 0; p < 4 * 2; p++) {
			for(int o = 0; o < 4; o++)
				cout << deriv[p*4 + o] << " ";
			cout << endl;

		}
		getchar();
		*/
		

		cblas_dcopy(16, &deriv[i*16], 1, deriv_intermd, 1);
		for(int q = 1; q < L; q++) {
			// deriv(:,:,l) = deriv(:,:,l) * Tr(:,:,q);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						4, 4, 4, one, deriv_intermd, 4, &Tr[q*16], 4, zero, &deriv[i*16], 4);

			/* correct
			cout << "deriv-i3" << endl;
			for(int p = 0; p < 4 * 2; p++) {
				for(int o = 0; o < 4; o++)
					cout << deriv[p*4 + o] << " ";
				cout << endl;

			}
			getchar();
			*/

			cblas_dcopy(16, &deriv[i*16], 1, deriv_intermd, 1);
			
		}

		// deriv(:,:,l) = deriv(:,:,l) * poseMean;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, one, deriv_intermd, 4, poseMean, 4, zero, &deriv[i*16], 4);

		/*
		cout << "poseMean" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << poseMean[p*4 + o] << " ";
			cout << endl;

		}
		getchar();

		cout << "deriv_intermd" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << deriv_intermd[p*4 + o] << " ";
			cout << endl;

		}
		getchar();
		*/

		/*
		cout << "deriv-final" << endl;
		for(int p = 0; p < 4 * 2; p++) {
			for(int o = 0; o < 4; o++)
				cout << deriv[p*4 + o] << " ";
			cout << endl;

		}
		getchar();
		*/
	}

	//cout << "inner7" << endl;
	/*
	cout << "deriv" << endl;
	for(int i = 0; i < 4 * 2; i++) {
		for(int j = 0; j < 4; j++)
			cout << deriv[i*4 + j] << " ";
		cout << endl;

	}
	getchar();
	*/
		mkl_free(deriv_intermd);
	
	mkl_free_buffers();
	//cout << "inner7a" << endl;
	mkl_free(Tr);
	//cout << "inner7b" << endl;
	mkl_free(transform_tmp);
	//cout << "inner7c" << endl;
	mkl_free(transform_tmp2);
	//cout << "inner7d" << endl;
	mkl_free(value2);
	//cout << "inner7e" << endl;

	
	//cout << "inner8" << endl;
	
}

//==========================================================computePointCorrValueAndDerivative===//
// COMPUTEPOINTCORRVALUEANDDERIVATIVE computes the value of the cost function
// and the derivative of it with respect to the weights given to pose
// variations.
// INPUT: BASIS is a 7xL
//		  POSECOEF is a Lx1
//        POSEMEAN is a 4x4
//        PX is a Nx3
//		  Y is Nx3
//        P1 is Nx1
// OUTPUT:DERIV is a Lx1
// 
void Registration::computePointCorrValueAndDerivative(double* poseCoef, int L, double* PX, double* P1,
													  double* poseMean, double* y, int N, double* basis,
													  int pose_modes_nr_, double* deriv)
{

	//cout << "computePointCorrValueAndDerivative begin" << endl;

	int temp_array_size;
	int onei = 1;
	double zero = 0.0;
	double one = 1.0;

	// [Tr TrDeriv] = computeMultiTransformationValueAndDerivative(poseCoef, basis(:,1:coefNr), poseMean);
	double* TrDeriv = (double*)mkl_malloc(4 * 4 * 2 * sizeof(double), 64);
	double* Tr = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	
	
	// Need to copy the arrays
	double* basis_trim = (double*)mkl_malloc(7 * pose_modes_nr_ * sizeof(double), 64); // have to change 31 later
	temp_array_size = 7 * 2;
	int two = 2;
	/*
	dcopy(&temp_array_size, basis, &basis_cols, basis_trim, &two);
	dcopy(&temp_array_size, &basis[1], &basis_cols, &basis_trim[1], &two);
	cout << "trim: " << basis_trim[3] << endl;
	computeMultiTransformationValueAndDerivative(poseCoef, basis_trim, L, poseMean, TrDeriv, Tr);
	*/
	//cout << "compute1a" << endl;
	

	// Checking input to most inner funciton
	/*
	for(int p = 0; p < 2; p++) cout << poseCoef[p] << " ";
	cout << endl;
	cout << "poseCoef" << endl;
	getchar();


	for(int p = 0; p < 7; p++) {
		for(int o = 0; o < 2; o++) 
			cout << basis[p*31 + o] << " ";
		cout << endl;
	}
	cout << "basis" << endl;
	getchar();
	

	
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++)
			cout << poseMean[i*4 + j] << " ";
		cout << endl;

	}
	cout << "poseMean" << endl;
	getchar();
	*/
	computeMultiTransformationValueAndDerivative(poseCoef, basis, L, poseMean, TrDeriv, Tr);
	cout << "poedit" << endl;

	// Need to add in the calculation in order to return TrY's values
	
	//printMatrix(Tr, 4, 4, "Tr");
	//printMatrix(y, N, 3, "y");
	
	// TrY = bsxfun(@plus, y*Tr(1:3,1:3)', Tr(1:3,4)');
	double* TrY = (double*)mkl_malloc(N * 3 * sizeof(double), 64);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, 3, 3, one, y, 3, Tr, 4, zero, TrY, 3);
	
	//cout << "poedit1a" << endl;

	double* Nx1 = (double*)mkl_malloc(N * 1 * sizeof(double), 64);
	for(int i = 0; i < N * 1; i++) Nx1[i] = 1.0;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, 3, 1, one, Nx1, 1, &Tr[3], 4, one, TrY, 3);
	
	//cout << "poedit2" << endl;

	/*
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++)
			cout << TrDeriv[i*4 + j] << " ";
		cout << endl;

	}
	cout << "TrDeriv" << endl;
	getchar();

	
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++)
			cout << Tr[i*4 + j] << " ";
		cout << endl;

	}
	cout << "Tr" << endl;
	getchar();
	

	
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < 3; j++)
			cout << y[i*3 + j] << " ";
		cout << endl;

	}
	cout << "y " << y[0] << endl;
	getchar();
	
	

	
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < 3; j++)
			cout << TrY[i*3 + j] << " ";
		cout << endl;

	}
	cout << "TrY " << TrY[0] << endl;
	getchar();
	*/
	
	
	
	// deriv = zeros(L, 1);
	//double* deriv = (double*)mkl_malloc(L * 1 * sizeof(double), 64);
	for(int i = 0; i < L; i++) deriv[i] = 0.0;

	double* TrDerivY = (double*)mkl_malloc(N * 3 * sizeof(double), 64);
	double* tmp1 = (double*)mkl_malloc(N * 3 * sizeof(double), 64);
	double* tmp2 = (double*)mkl_malloc(N * 3 * sizeof(double), 64);
	double* tmp3 = (double*)mkl_malloc(3 * 1 * sizeof(double), 64);
	double* sum1 = (double*)mkl_malloc(N * 1 * sizeof(double), 64);
	double* sum2 = (double*)mkl_malloc(N * 1 * sizeof(double), 64);
	
	//cout << "compute5" << endl;
	
	for(int i = 0; i < L; i++) {
		//cout << "poedit3" << endl;
		// TrDerivY = bsxfun(@plus, y*TrDeriv(1:3,1:3,l)', TrDeriv(1:3,4,l)'); 
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
					N, 3, 3, one, y, 3, &TrDeriv[i*16], 4, zero, TrDerivY, 3);

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, 3, 1, one, Nx1, 1, &TrDeriv[i*16 + 3], 4, one, TrDerivY, 3);
		
		/*
		for(int p = 0; p < N; p++) {
			for(int o = 0; o < 3; o++)
				cout << y[p*3 + o] << " ";
			cout << endl;

		}
		cout << "y " << y[0] << endl;
		getchar();
		

		for(int p = 0; p < 4 * 2; p++) {
			for(int o = 0; o < 4; o++)
				cout << TrDeriv[p*4 + o] << " ";
			cout << endl;

		}
		cout << "TrDeriv " << endl;
		getchar();
		
	
		for(int p = 0; p < N; p++) {
			for(int o = 0; o < 3; o++)
				cout << TrDerivY[p*3 + o] << " ";
			cout << endl;

		}
		cout << "TrDerivY " << TrDerivY[0] << endl;
		getchar();
		*/
		

		// deriv(l) = 2*sum(P1.*sum(TrY.*TrDerivY,2)) - 2*sum(sum(PX.*TrDerivY));
		temp_array_size = N * 3;
		vdMul(temp_array_size, TrY, TrDerivY, tmp1);

		/*
		for(int p = 0; p < N; p++) {
			for(int o = 0; o < 3; o++) {
				cout << tmp1[p*3 + o] << " ";
			}
			cout << endl;
		}
		cout << "tmp1" << endl;
		getchar();
		*/
		
		/*
		//optimize later po!
		temp_array_size = 3;
		for(int x = 0; x < N; x++) {
			sum1[x] = tmp1[x*3] + tmp1[x*3 + 1] + tmp1[x*3 + 2];
			//cout << sum1[x] << endl;
			
		}
		//getchar();
		
		
		temp_array_size = N * 1;
		dcopy(&temp_array_size, sum1, &onei, sum2, &onei);
		vdMul(temp_array_size, P1, sum2, sum1);
		for(int p = 0; p < N; p++) deriv[i] += 2 * sum1[p]; // optimize again po!
		*/

		//cout << "poedit4" << endl;
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
					1, 3, N, one, P1, 1, tmp1, 3, zero, tmp3, 3);
		deriv[i] = 2 * (tmp3[0] + tmp3[1] + tmp3[2]);
		

		
		/*
		for(int p = 0; p < N; p++) {
			cout << sum2[p] << endl;
		}
		cout << "sum2 " << endl;
		getchar();

	
		for(int p = 0; p < N; p++) {
			cout << P1[p] << endl;
		}
		cout << "P1 " << endl;					// P1 is different!!!
		getchar();

		for(int p = 0; p < N; p++) {
			cout << sum1[p] << endl;
		}
		cout << "sum1 " << endl;
		getchar();
		*/
		
		
		
		// correct up to here

		temp_array_size = N * 3;	
		vdMul(temp_array_size, PX, TrDerivY, tmp2);

		
		for(int p = 0; p < N * 3; p++)    /// sum of a vector
			deriv[i] -= 2 * tmp2[p];
	
		/*
		for(int p = 0; p < N; p++) {
			for(int o = 0; o < 3; o++) 
				cout << PX[p*3 + o] << " ";
			cout << endl;
		}
		cout << "PX " << endl;			// PX is different!
		getchar();
		*/

		/*
		for(int p = 0; p < N; p++) {
			for(int o = 0; o < 3; o++) 
				cout << TrDerivY[p*3 + o] << " ";
			cout << endl;
		}
		cout << "TrDerivY " << endl;		// TrDerivY is off in the 2nd it
		getchar();

		for(int p = 0; p < N; p++) {
			for(int o = 0; o < 3; o++) 
				cout << tmp2[p*3 + o] << " ";
			cout << endl;
		}
		cout << "tmp2 " << endl;
		getchar();
		

		cout << "derivhere: " << deriv[i] << endl;
		getchar();
		*/
	}
	
	/*
	for(int p = 0; p < 2; p++) {
		cout << deriv[p] << " ";
		
	}
	cout << endl;
	cout << "deriv " << endl;
	getchar();
	*/
	
	//cout << "middle1" << endl;
	mkl_free_buffers();
	//cout << "middle1a" << endl;
	//mkl_thread_free_buffers();
	mkl_free(TrDeriv);
	//cout << "middle1b" << endl;
	mkl_free(Tr);
	//cout << "middle1c" << endl;
	mkl_free(basis_trim);
	//cout << "middle1d" << endl;
	mkl_free(TrY);
	//cout << "middle1e" << endl;
	mkl_free(Nx1);
	//cout << "middle1f" << endl;
	mkl_free(TrDerivY);
	//cout << "middle1g" << endl;
	mkl_free(tmp1);
	//cout << "middle1h" << endl;
	mkl_free(tmp2);
	//cout << "middle1i" << endl;
	mkl_free(sum1);
	//cout << "middle1j" << endl;
	mkl_free(sum2);
	//cout << "middle2" << endl;
}

//==============================================multiObjectAtlasPoseCoefCostFunction=============//
// INPUT:  POSECOEF is a Lx1 vector
//		   PX is a Nx3
//		   GAMMA is the regularization matrix
// OUTPUT: VALUE is a scalar  --> not complete
//		   DERIV is a Lx1
void Registration::multiObjectAtlasPoseCoefCostFunction(double* poseCoef, int L, MultiObjModelMKL* model,
														double* shapeDeformed, double* PX,
														double* P1, double Gamma,
														double* deriv)
{
	int onei = 1;

	//mkl_disable_fast_mm();

	double value = 0;
	double* derivTmp = (double*)mkl_malloc(L * 1 * sizeof(double), 64);
	double* derivTmp2 = (double*)mkl_malloc(L * 1 * sizeof(double), 64);
	for(int i = 0; i < L * 1; i++) derivTmp2[i] = 0.0;

	int temp_array_size = L*1;	
	
	cout << "multiObjectAtlasPoseCoefCostFunction begin" << endl;
	for(int i = 0; i < model->get_obj_nr_(); i++) {
		/*
		[valueTmp derivTmp] = computePointCorrValueAndDerivative( 
				poseCoef, 
				PX(model.objInd(l,1):model.objInd(l,2),:),
				P1(model.objInd(l,1):model.objInd(l,2)), 
				model.poseMean(:,:,l), 
				shapeDeformed(model.objInd(l,1):model.objInd(l,2),:), 
				model.poseFeature(l*7-6:l*7,:));
		*/
		/*
		computePointCorrValueAndDerivative(double* poseCoef,
										   int L, 
										   double* PX, 
										   double* P1,
										   double* poseMean, 
										   double* y, 
										   int N, 
										   double* basis
										   double* deriv)
													  */
		/*
		cout << "basis" << endl;
		for(int p = 0; p < 7; p++) {
			for(int o = 0; o < 2; o++)
				cout << (model->get_pose_feature_())[i * model->get_pose_sigma_rows() + p*2 + o] << " ";
			cout << endl;
		}
		
		getchar();
		*/

		/*
		for(int p = 0; p < 1500; p++) {
			for(int o = 0; o < 1; o++)
				cout << P1[(model->get_obj_ind_())[i*2] - 1 + p + o] << " ";
			cout << endl;
		}
		*/
		
		/*
		cout << "PX " << (model->get_obj_ind_())[i*2+1] - (model->get_obj_ind_())[i*2] + 1 << endl;
		cout << "P1 " << P1[(model->get_obj_ind_())[i*2] - 1] << endl;
		getchar();
		*/

		Registration::computePointCorrValueAndDerivative(poseCoef,
														 L,
														 &PX[((model->get_obj_ind_())[i*2] - 1) * 3],
														 &P1[(model->get_obj_ind_())[i*2] - 1],
														 &(model->get_pose_mean_())[i*16],
														 &shapeDeformed[((model->get_obj_ind_())[i*2] - 1) * 3], 
														 (model->get_obj_ind_())[i*2+1] - (model->get_obj_ind_())[i*2] + 1,
														 &(model->get_pose_feature_())[i*7 * model->get_pose_modes_nr_()], 
														 model->get_pose_modes_nr_(),
														 derivTmp);
		
		// &(model->get_pose_feature_())[i * model->get_pose_sigma_rows()],
		cblas_dcopy(L, deriv, 1, derivTmp2, 1);
		vdAdd(temp_array_size, derivTmp, derivTmp2, deriv);
	}
	
	/*
	cout << "outside deriv: " << deriv[0] << " " << deriv[1] << endl;
	getchar();
	*/

	// normalize the cost function by the number of points

	// deriv = deriv / model.objInd(end, 1);
	for(int i = 0; i < L; i++) {
		deriv[i] /= (model->get_obj_ind_())[(model->get_obj_nr_()-1)*2];
	}

	
	// deriv = deriv + Gamma*poseCoef;
	double* Gamma_x_poseCoef = (double*)mkl_malloc(2 * 1 * sizeof(double), 64);
	cblas_dcopy(L, poseCoef, 1, Gamma_x_poseCoef, 1);
	for(int i = 0; i < L; i++) Gamma_x_poseCoef[i] *= Gamma;
	for(int i = 0; i < L; i++) deriv[i] += Gamma_x_poseCoef[i];
	

	
	mkl_free_buffers();
	mkl_thread_free_buffers();

	mkl_free(Gamma_x_poseCoef);
	mkl_free(derivTmp);
	mkl_free(derivTmp2);
}

//==============================================multiObjectAtlasPoseCoefCostFunction=============//
// INPUT:  POSECOEF is a Lx1 vector
//		   PX is a Nx3
//		   GAMMA is the regularization matrix
// OUTPUT: VALUE is a scalar  --> not complete
//		   DERIV is a Lx1
void Registration::multiObjectAtlasShapeCoefFunction(double* poseCoef, double* shapeCoef, int L, int* objInd,
													 double* shapeFeature, double* poseFeature,
													 double* poseDeformed, double* movedShape,
													 int obj_nr, double* poseMean, int shapeFeatureCols,
													 double* PX, double* P1, double Gamma,
													 double* deriv)
{
	
	// Initialize some variables
	double* Tr = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* Tr2 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* pose_feature_transform = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* pose_feature_transform2 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* part1 = (double*)mkl_malloc(1 * L * sizeof(double), 64);
	double* part2 = (double*)mkl_malloc(1 * L * sizeof(double), 64);
	double* part1_2 = (double*)mkl_malloc(L * sizeof(double), 64);

	double* ones = (double*)mkl_malloc(1 * 3 * sizeof(double), 64);
		for(int x = 0; x < 3; x++) ones[x] = 1.0;
	double* two = (double*)mkl_malloc(1 * sizeof(double), 64);
		two[0] = 2.0;

	// Main Loop
	for(int i = 0; i < obj_nr; i++) {
		int num_elements = (objInd[i*2 + 1] - objInd[i*2]) + 1;

		// Tr = eye(4)
		for(int x = 0; x < 16; x++) {
			Tr[x] = 0.0;
			Tr2[x] = 0.0;
		}
		for(int x = 0; x < 16; x += 5) {
			Tr[x] = 1.0;
			Tr2[x] = 1.0;
		}

		// Clear transformation matrix output
		for(int x = 0; x < 16; x++) {
			pose_feature_transform[x] = 0.0;
			pose_feature_transform2[x] = 0.0;
		}

		for(int j = 0; j < L; j++) {
			// getTransformationMatrixFromVector( model.poseFeature((l-1)*7+1: l*7, i) );
			MultiObjModelMKL::getTransformationMatrixFromVector( &poseFeature[i*7*31 + j], 31, pose_feature_transform );
			
			// Tr = Tr * real(expm(poseCoef(i)*getTransformationMatrixFromVector( ... )));
			for(int x = 0; x < 16; x++) 
				pose_feature_transform[x] *= poseCoef[j];
			MultiObjModelMKL::matrix_exponential(pose_feature_transform, pose_feature_transform2);

			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						4, 4, 4, 1, Tr, 4, pose_feature_transform2, 4, 0, Tr2, 4);
			cblas_dcopy(16, Tr2, 1, Tr, 1);
		}


		
		

		// Tr = real(Tr * model.poseMean(:,:,l));
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, 1, Tr2, 4, &poseMean[i * 16], 4, 0, Tr, 4);

		

		// MTr = movedShape(model.objInd(l,1):model.objInd(l,2),:)*Tr(1:3,1:3);
		double* MTr = (double*)mkl_malloc(num_elements * 3 * sizeof(double), 64);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					num_elements, 3, 3, 1, &movedShape[(objInd[i*2]-1) * 3], 3, Tr, 4, 0, MTr, 3);
		
		/*
		for(int p = 0; p < num_elements; p++) {
			for(int o = 0; o < 3; o++) {
				cout << movedShape[objInd[i*2]-1 + p*3 + o] << " ";
			}
			cout << endl;
		}
		cout << "movedshape, l=" << i << endl;
		getchar();
		*/
		/*
		for(int p = 0; p < 2; p++) {
			for(int o = 0; o < 2; o++) {
				cout << objInd[p*2 + o] << " ";
			}
			cout << endl;
		}
		cout << "objInd" << i << endl;
		getchar();
		*/

		// PMTr = bsxfun(@times, MTr, P1(model.objInd(l,1):model.objInd(l,2)))';
		double* PMTr_temp = (double*)mkl_malloc(num_elements * 3 * sizeof(double), 64);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					num_elements, 3, 1, 1, &P1[(int)objInd[i*2]-1], 1, ones, 3, 0, PMTr_temp, 3);

		double* PMTr = (double*)mkl_malloc(num_elements * 3 * sizeof(double), 64);
		vdMul(num_elements*3, MTr, PMTr_temp, PMTr);


		// featurePart = model.shapeFeature(model.objInd(l,1)*3-2:model.objInd(l,2)*3,1:featureNr);
		// part1 = PMTr(:)'*featurePart;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					1, L, num_elements * 3, 1, PMTr, num_elements * 3, &shapeFeature[(objInd[i*2]-1)*3*shapeFeatureCols], shapeFeatureCols, 0, part1, L);
		

		// PXTr = PX(model.objInd(l,1):model.objInd(l,2),:)*Tr(1:3,1:3);
		double* PXTr = (double*)mkl_malloc(num_elements * 3 * sizeof(double), 64);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					num_elements, 3, 3, 1, &PX[(objInd[i*2]-1)*3], 3, Tr, 4, 0, PXTr, 3);


		// PXTr = PXTr';
		// part2 = PXTr(:)'*featurePart;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					1, L, num_elements * 3, 1, PXTr, num_elements * 3, &shapeFeature[(objInd[i*2]-1)*3*shapeFeatureCols], shapeFeatureCols, 0, part2, L);
		

		// shapeDeriv = shapeDeriv + 2*(part1-part2)';
		vdSub(L, part1, part2, part1_2);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					L, 1, 1, 1, part1_2, 1, two, 1, 1, deriv, 1);

		// Free loop variables from memory
		mkl_free_buffers();
		mkl_free(MTr);
		mkl_free(PMTr_temp);
		mkl_free(PMTr);
		mkl_free(PXTr);

	}


	// Free all local variables
	mkl_free_buffers();
	mkl_free(Tr);
	mkl_free(Tr2);
	mkl_free(pose_feature_transform);
	mkl_free(pose_feature_transform2);
	mkl_free(ones);
	mkl_free(part1);
	mkl_free(part2);
	mkl_free(part1_2);

	
}