#include "Registration.h"
#include <mkl.h>
#include <ctime>



#define DIM 3

using namespace std;


//=================================================================================Functions=====//
double squarefunction(double x)
{
	return x*x;
}
double sqrtfunction(double x)
{
	return 1/sqrt(x);
}

double expfunction(double x)
{
	return exp(x);
}

void Registration::printMatrix(double* mat, int x, int y, string name)
{
	for(int i = 0; i < x; i++) {
		for(int j = 0; j < y; j++)
			cout << mat[i*y + j] << " ";
		cout << endl;

	}
	cout << name << endl;
	getchar();

}
//===============================================================================Constructor=====//
Registration::Registration(){}

Registration::Registration(MultiObjModel &model,
						   double shape_coef_length, 
						   double pose_coef_length, 
						   double gamma_shape_coef, 
						   double gamma_pose_coef)
{
	model_ = &model;
	shape_coef_ = new vnl_matrix<double>;
	pose_coef_ = new vnl_matrix<double>;
	shape_coef_->set_size(shape_coef_length,1);
	pose_coef_->set_size(pose_coef_length,1);
	shape_coef_->fill(0);
	pose_coef_->fill(0);
	gamma_shape_.set_size(shape_coef_length, shape_coef_length);
	gamma_pose_.set_size(pose_coef_length, pose_coef_length);

	gamma_shape_.fill(0);
	gamma_pose_.fill(0);
	
	shape_coef_length_ = shape_coef_length;
	pose_coef_length_ = pose_coef_length;

	gamma_shape_.set_diagonal(gamma_shape_coef * model.get_shape_sigma_()->apply(sqrtfunction).get_column(0));
	gamma_pose_.set_diagonal(gamma_pose_coef * model.get_pose_sigma_()->apply(sqrtfunction).get_column(0));

	generated_point_ = new vnl_matrix<double>; 
	generated_point_prb_ = new vnl_matrix<double>; 
	mu_generated_point_ = new vnl_matrix<double>; //test
	mu_pose_deformed_ = new vnl_matrix<double>; //test
	Px_ = new vnl_matrix<double>;
	Pt1_ = new vnl_matrix<double>;
	P1_ = new vnl_matrix<double>;
	rigid_tr_ = new vnl_matrix<double>;
}

Registration::Registration(MultiObjModelMKL &model,
						   double shape_coef_length, 
						   double pose_coef_length, 
						   double gamma_shape_coef, 
						   double gamma_pose_coef)
{
	model_MKL_ = &model;
	
	shape_coef_length_ = shape_coef_length;
	pose_coef_length_ = pose_coef_length;

	shape_coef_MKL_ = (double*)mkl_malloc(shape_coef_length * 1 * sizeof(double), 64);
	pose_coef_MKL_ = (double*)mkl_malloc(pose_coef_length * 1 * sizeof(double), 64);
	gamma_shape_MKL_ = (double*)mkl_malloc(shape_coef_length * shape_coef_length * sizeof(double), 64);
	gamma_pose_MKL_ = (double*)mkl_malloc(pose_coef_length * pose_coef_length * sizeof(double), 64);
	for(int i = 0; i < shape_coef_length; i++) shape_coef_MKL_[i] = 0.0;
	for(int i = 0; i < pose_coef_length; i++) pose_coef_MKL_[i] = 0.0;
	for(int i = 0; i < shape_coef_length * shape_coef_length; i++) gamma_shape_MKL_[i] = 0.0;
	for(int i = 0; i < pose_coef_length * pose_coef_length; i++) gamma_pose_MKL_[i] = 0.0;


	// gamma_shape_.set_diagonal(gamma_shape_coef * model.get_shape_sigma_()->apply(sqrtfunction).get_column(0));
	int row = 0;
	for(int i = 0; i < shape_coef_length * shape_coef_length; i += shape_coef_length+1) {
		gamma_shape_MKL_[i] = gamma_shape_coef * 1 / sqrt((model.get_shape_sigma_())[row]) ;
		row++;
	}

	// gamma_pose_.set_diagonal(gamma_pose_coef * model.get_pose_sigma_()->apply(sqrtfunction).get_column(0));
	row = 0;
	for(int i = 0; i < pose_coef_length * pose_coef_length; i += pose_coef_length+1) {
		gamma_pose_MKL_[i] = gamma_pose_coef * 1 / sqrt((model.get_pose_sigma_())[row]) ;
		row++;
	}

	rigid_tr_MKL_ = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);

	/*
	generated_point_ = new vnl_matrix<double>; 
	generated_point_prb_ = new vnl_matrix<double>; 
	mu_generated_point_ = new vnl_matrix<double>; //test
	mu_pose_deformed_ = new vnl_matrix<double>; //test
	Px_ = new vnl_matrix<double>;
	Pt1_ = new vnl_matrix<double>;
	P1_ = new vnl_matrix<double>;
	*/
}

//================================================================================Destructor=====//
Registration::~Registration()
{
	delete generated_point_;
	delete generated_point_prb_; //test
	delete shape_coef_;
	delete pose_coef_;
	delete mu_generated_point_; //test
	delete mu_pose_deformed_; //test
	delete Px_;
	delete Pt1_;
	delete P1_;
	delete rigid_tr_;
}

//=======================================================================Setters And Getters=====//
void Registration::set_model_(MultiObjModel model) {*model_ = model;}
void Registration::set_US_vol_(USVolume USVol) {US_vol_ = USVol;}
void Registration::set_max_iter_(int maxIter) {max_iter_ = maxIter;}
void Registration::set_sigma_(double sigma) {sigma_ = sigma;}
void Registration::set_sigma_threshold_(double sigma_threshold) {sigma_threshold_ = sigma_threshold;}
void Registration::set_gen_point_num_(int gen_point_num) {gen_point_num_ = gen_point_num;}
double Registration::get_gen_point_num_(void) { return gen_point_num_;}

void Registration::set_generated_point_(vnl_matrix<double>* generated_point) {generated_point_ = generated_point;}
void Registration::set_generated_point_prb_(vnl_matrix<double>* generated_point_prb) {generated_point_prb_ = generated_point_prb;}
void Registration::set_mu_generated_point_(vnl_matrix<double>* mu_generated_point) {mu_generated_point_ = mu_generated_point;}
void Registration::set_mu_pose_deformed_(vnl_matrix<double>* mu_pose_deformed) {mu_pose_deformed_ = mu_pose_deformed;}
void Registration::set_rigid_tr_(vnl_matrix<double>* rigid_tr) {rigid_tr_ = rigid_tr;}
void Registration::set_rigid_tr_MKL_(double* rigid_tr) {rigid_tr_MKL_ = rigid_tr; }
void Registration::set_P1_(vnl_matrix<double>* P1) {P1_ = P1;}
void Registration::set_Pt1_(vnl_matrix<double>* Pt1) {Pt1_ = Pt1;}
void Registration::set_Px_(vnl_matrix<double>* Px) {Px_ = Px;}
double Registration::get_sigma_(void) {return sigma_;}

void Registration::set_outlier_(double outlier) {outlier_ = outlier;}



//============================================================generateProbabilityMap with MKL=====//
void Registration::generateProbabilityMap4(double *x, int x_rows, int D,
										   double *xPr,
										   double *y, int y_rows,
		                                   double sigma2,
										   double outlier, 
										   double *P1,
										   double *Pt1,
										   double *Px)
{
	// Initialize N, M, and D from input
	int N = x_rows;
	int M = y_rows;
	int P1_rows = y_rows;
	int P1_cols = 1;
	double ksig, outlier_tmp, sp;

	// Lookup table for the exponential
	double* expTable = (double*)mkl_malloc(10000 * sizeof(double), 64);//new double[1000];
	for(int i=0;i < 10000;i++)
		expTable[i] = exp(-(double)i/1000);

	double* P = (double *)mkl_malloc( M*1*sizeof( double ), 64 );
	double* temp_x = (double *)mkl_malloc( D*1*sizeof( double ), 64 );
	


	// Set sizes of matrices P1,Pt and Pt1. Fill them with zeros.
	//P1 = (double *)mkl_malloc( M*1*sizeof( double ), 64 );
	for(int i = 0; i < M*1; i++) P1[i] = 0.0;
	
	//Pt1 = (double *)mkl_malloc( N*1*sizeof( double ), 64 );
	//fill_mkl_matrix(Pt1, M, 1, 0.0);
	
	//Px = (double *)mkl_malloc( M*D*sizeof( double ), 64 );
	//fill_mkl_matrix(Px, M*D, 0.0);
	for(int i = 0; i < M*D; i++) Px[i] = 0.0;

	ksig = -2.0 * sigma2;
	outlier_tmp = (outlier * M * pow(-ksig*3.14159265358979,0.5*D) )/((1-outlier)*N);   


	// Matrices used for main loop
	double* Mx1 = (double *)mkl_malloc( M*1*sizeof( double ), 64 );
	for(int i = 0; i < M; i++) Mx1[i] = 1.0;

	double* Q = (double *)mkl_malloc( M*3*sizeof( double ), 64 );
	for(int i = 0; i < M*3; i++) Q[i] = 0.0;

	double* F = (double *)mkl_malloc( M*1*sizeof( double ), 64 );
	for(int i = 0; i < M*1; i++) F[i] = 0.0;

	double* tempM = (double *)mkl_malloc( 3*1*sizeof( double ), 64 );
	for(int i = 0; i < 3; i++) tempM[i] = 1.0;

	double* x_mod;
	double* result;
	double one = 1.0;
	double negone = -1.0;
	double zero = 0.0;
	double beta = 1.0;
	double alpha = 1.0;

	double* x_nth_row;
	x_nth_row = (double *)mkl_malloc( D*sizeof( double ), 64 ); //x + n * D * sizeof(double);
	double* temp_array;
	temp_array = (double *)mkl_malloc( 1*1*sizeof( double ), 64 );

	int temp_matrix_size;

	// Main loop going over two point sets to calculate the probability map.
	for(int n = 0; n < N; n++) {

		//Q = Mx1 * x->get_n_rows(n,1)
		for(int i = 0; i<D; i++)
			x_nth_row[i] = x[n*D+i];

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    M, D, 1, one, Mx1, 1, x_nth_row, D, zero, Q, D);
	

		// Q = Q - y
		temp_matrix_size = y_rows * D;
		vdsub(&temp_matrix_size, Q, y, Q); 
		
		//Q->apply(squarefunction)
		int Q_rows = y_rows;
		int Q_cols = D;
		temp_matrix_size = Q_rows*Q_cols;
		vdmul(&temp_matrix_size, Q, Q, Q);		

		//*F = ( *Q * *tempM / ksig)
		beta = 1.0 / ksig;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    Q_rows, 1, Q_cols, beta, Q, Q_cols, tempM, 1, zero, F, 1);

		//F->apply(expfunction)
		int F_rows = Q_rows;
		int F_cols = 1;
		temp_matrix_size = F_rows * F_cols;

		// Calcuate exponential through the lookup table
		for(int i=0;i < Q_rows;i++)
		{/*
			if(F[i] < -10)
				F[i] = 0;
			else
				F[i] = expTable[-(int)floor(F[i]*1000)];*/
			F[i] = exp(F[i]);
		}
		//vdExp(temp_matrix_size, F, F);	
		
		//sp = (Mx1->transpose()* *F).get(0,0);
		
		//temp_array[0] = 0.0;
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                    1, 1, M, one, Mx1, 1, F, F_cols, zero, temp_array, 1);
		sp = temp_array[0];
		sp += outlier_tmp;

		//*P = (*F/sp) * xPrb->get(n, 0) = F*(sp/xPrb->get(n,0))
		double multiplier = 1/ sp * xPr[n];
		for(int i = 0; i < F_rows; i++) {
			P[i] = F[i] * multiplier;
		}
		
		//Pt1->put(n,0,(1 - outlier_tmp/sp) * xPrb->get(n,0))
		Pt1[n] = (1 - outlier_tmp/sp) * xPr[n];

		//*P1 = *P1 + (*P)
		temp_matrix_size = M * 1;
		vdadd(&temp_matrix_size, P1, P, P1);


		//*Px = *Px + *P*x->get_n_rows(n,1)
		alpha = 1;
		beta = 1;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    M, D, 1, one, P, 1, x_nth_row, D, one, Px, D);

	}

	mkl_free_buffers();
	//mkl_thread_free_buffers();
	mkl_free(expTable);
	mkl_free(P);
	mkl_free(temp_x);
	mkl_free(Mx1);
	mkl_free(Q);
	mkl_free(F);
	mkl_free(tempM);
	mkl_free(x_mod);
	mkl_free(result);
	mkl_free(x_nth_row);
	mkl_free(temp_array);

	return;	
}

// Helper function for MKL implementation
void Registration::fill_mkl_matrix(double* matrix, int length, double fill_constant)
{
	for(int i = 0; i < length; i++) {
		matrix[i] = fill_constant;
	}

}

//=========================================================RigidRegistration with MKL Optimization=====//
double* Registration::rigidRegistration2(double* x, int x_rows, int D,
									    double* xPr, 
										double* y, int y_rows,
										double* y_init, 
										double* sigma, 
										double outlier,
										double* Px,
										double* Pt1,
										double* P1)
{
	int N = x_rows;
	int M = y_rows;
	int temp_matrix_size;

	// Initialize rigid transformation matrices
	double* rigidTr;
	rigidTr = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);

	double* mu_x;
	mu_x = (double*)mkl_malloc(D * 1 * sizeof(double), 64);
	double* mu_y;
	mu_y = (double*)mkl_malloc(D * 1 * sizeof(double), 64 );
	double* A;
	A = (double*)mkl_malloc(D * D * sizeof(double), 64 );
	double* rotation;
	rotation = (double*)mkl_malloc(D * D * sizeof(double), 64);
	double* translation;
	translation = (double*)mkl_malloc(D * 1 * sizeof(double), 64);
	
	double Np = 0;
	
	//double* Px;
	//Px = (double*)mkl_malloc(N * D * sizeof(double), 64);
	//double* Pt1;
	//Pt1 = (double*)mkl_malloc(N * 1 * sizeof(double), 64);
	//double* P1; 
	//P1 = (double*)mkl_malloc(M * 1 * sizeof(double), 64);

	generateProbabilityMap4(x, x_rows, D, xPr, y, y_rows, *sigma * *sigma, outlier, P1, Pt1, Px);

	// For MKL functions
	double alpha = 1.0;
	double beta = 1.0;
	double zero = 0.0;
	double one = 1.0;
	double negone = -1.0;

	//Np = Pt1->get_column(0).sum();
	for(int i = 0; i < M; i++)
		Np += Pt1[i];

	//mu_x = x->transpose() * (*Pt1/Np);
	beta = 1.0 / Np;
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                D, 1, x_rows, beta, x, D, Pt1, 1, zero, mu_x, 1);

	//mu_y = y_init->transpose() * (*P1/Np);
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		        D, 1, M, beta, y_init, D, P1, 1, zero, mu_y, 1);

	//A = Px->transpose() * *y_init - Np * (mu_x * mu_y.transpose());
	double* A1;
	double* A2;
	A1 = (double*)mkl_malloc(D * D * sizeof(double), 64 );
	A2 = (double*)mkl_malloc(D * D * sizeof(double), 64 );

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		        D, D, M, one, Px, D, y_init, D, zero, A1, D);
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				D, D, 1, Np, mu_x, 1, mu_y, 1, zero, A2, D);

	temp_matrix_size = D*D;
	vdsub(&temp_matrix_size, A1, A2, A);

	// Solve for SVD of A using VNL
	vnl_matrix<double>* A_vnl = new vnl_matrix<double>(D, D);
	for(int i = 0; i < D; i++) {
		for(int j = 0; j < D; j++) {
			A_vnl->put(i, j, A[i*D + j]);
		}
	}
	vnl_svd<double> svd(*A_vnl);

	double*U = (double*)mkl_malloc(D * D * sizeof(double), 64);
	double*V = (double*)mkl_malloc(D * D * sizeof(double), 64);
	double*W = (double*)mkl_malloc(D * D * sizeof(double), 64);

	for(int i = 0; i < D; i++) {
		for(int j = 0; j < D; j++) {
			U[i*D + j] = svd.U().get(i, j);
		}
	}

	for(int i = 0; i < D; i++) {
		for(int j = 0; j < D; j++) {
			V[i*D + j] = svd.V().get(i, j);
		}
	}

	for(int i = 0; i < D; i++) {
		for(int j = 0; j < D; j++) {
			if (i == j)
				W[i*D + j] = svd.W().get(i, j);
			else
				W[i*D + j] = 0.0;
		}
	}
	
	// temp_matrix = svd.U() * svd.V().transpose()
	double* temp_matrix;
	temp_matrix = (double*)mkl_malloc(D * D * sizeof(double), 64);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                    D, D, D, one, U, D, V, D, zero, temp_matrix, D);	

	//C.set_size(x->cols(), x->cols());
	//C.set_identity();
	//C.put(C.rows()-1, C.cols()-1, vnl_determinant( temp_matrix ));
	double* C;
	C = (double*)mkl_malloc(D * D * sizeof(double), 64);
	
	//Set identity matrix
	for(int i = 0; i < 9; i++) C[i] = 0.0;
	C[0] = 1.0; C[4] = 1.0;
	
	//Find determinant
	C[8] = temp_matrix[0]*temp_matrix[4]*temp_matrix[8] + temp_matrix[1]*temp_matrix[5]*temp_matrix[6]
		 + temp_matrix[2]*temp_matrix[3]*temp_matrix[7] - temp_matrix[2]*temp_matrix[4]*temp_matrix[6]
		 - temp_matrix[1]*temp_matrix[3]*temp_matrix[8] - temp_matrix[0]*temp_matrix[5]*temp_matrix[7];

	// rotation = svd.U() * C * svd.V().transpose(); 
	double* rotation_temp;
	rotation_temp = (double*)mkl_malloc(D * D * sizeof(double), 64);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		        D, D, D, one, U, D, C, D, zero, rotation_temp, D);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		        D, D, D, one, rotation_temp, D, V, D, zero, rotation, D);

	//translation = mu_x - rotation * mu_y;
	double* translation_temp;
	translation_temp = (double*)mkl_malloc(D * 1 * sizeof(double), 64);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				D, 1, D, one, rotation, D, mu_y, 1, zero, translation_temp, 1);
	temp_matrix_size = D*1;
	vdsub(&temp_matrix_size, mu_x, translation_temp, translation);


	// rigidTr.fill(0);
	for(int i = 0; i < 4*4; i++) rigidTr[i] = 0.0;

	
	// Update the rigid transformation matrix with rotation and translation matrices
	// rigidTr.update(rotation, 0, 0); 
	for(int i = 0; i < D; i++) {
		rigidTr[i] = rotation[i];
		rigidTr[i+4] = rotation[i+3];
		rigidTr[i+8] = rotation[i+6];
	}

	// rigidTr.update(translation, 0, 3);
	rigidTr[3] = translation[0];
	rigidTr[7] = translation[1];
	rigidTr[11] = translation[2];


	// rigidTr.put(3, 3, 1);
	rigidTr[15] = 1.0;


	// Calculate sigma
	double* sigma_temp1;
	double* sigma_temp3;
	double* sigma_temp4;
	double* sigma_temp5;
	sigma_temp1 = (double*)mkl_malloc(D * 1 * sizeof(double), 64);
	sigma_temp3 = (double*)mkl_malloc(1 * D * sizeof(double), 64);
	sigma_temp5 = (double*)mkl_malloc(D * D * sizeof(double), 64);
	double sigma_temp1_final = 0.0;
	double sigma_temp2_final = 0.0;
	double sigma_temp3_final = 0.0;
	double sigma_temp4_final = 0.0;
	double sigma_temp5_final = 0.0;

	// (Pt1->transpose() * x->apply(squarefunction)).get_row(0).sum()
	double* x_squared;
	x_squared = (double*)mkl_malloc(N * D * sizeof(double), 64);
	for(int i = 0; i < N * D; i++)
		x_squared[i] = x[i] * x[i];

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				1, D, N, one, Pt1, 1, x_squared, D, zero, sigma_temp1, D);
	
	for(int i = 0; i < D; i++)
		sigma_temp1_final += sigma_temp1[i];

	// Np*(mu_x.transpose() * mu_x).get(0,0)
	for(int i = 0; i < D; i++)
		sigma_temp2_final += mu_x[i] * mu_x[i];
	sigma_temp2_final *= Np;

	//(P1->transpose() * y_init->apply(squarefunction)).get_row(0).sum()
	double* y_init_squared;
	y_init_squared = (double*)mkl_malloc(M * D * sizeof(double), 64);
	for(int i = 0; i < M * D; i++)
		y_init_squared[i] = y_init[i] * y_init[i];

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				1, D, M, one, P1, 1, y_init_squared, D, zero, sigma_temp3, D);

	for(int i=0; i< D; i++)
		sigma_temp3_final += sigma_temp3[i];

	// Np*(mu_y.transpose() * mu_y).get(0,0)
	for(int i = 0; i < D; i++)
		sigma_temp4_final += mu_y[i] * mu_y[i];
	sigma_temp4_final *= Np;

	// 2.0*vnl_trace(svd.W()*C)
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				D, D, D, one, W, D, C, D, zero, sigma_temp5, D);
	sigma_temp5_final = 2 * (sigma_temp5[0] + sigma_temp5[4] + sigma_temp5[8]);

	// Stitching sigma together
	*sigma = sqrt( abs( sigma_temp1_final
						- sigma_temp2_final
						+ sigma_temp3_final
						- sigma_temp4_final
						- sigma_temp5_final )
				  / (Np * 3) );

	//cout << sigma_temp1_final << " " << sigma_temp2_final << " " << sigma_temp3_final << " " << sigma_temp4_final << " " << sigma_temp5_final << endl;

	/*
	*sigma = sqrt(
					abs((Pt1->transpose() * x->apply(squarefunction)).get_row(0).sum()
						 - Np*(mu_x.transpose() * mu_x).get(0,0)
				         + (P1->transpose() * y_init->apply(squarefunction)).get_row(0).sum()
				         - Np*(mu_y.transpose() * mu_y).get(0,0)
				         - 2.0*vnl_trace(svd.W()*C)
					   )/(Np*3)
				 );
	*/

	// Deallocate local variables
	mkl_free_buffers();
	mkl_free(mu_x);
	mkl_free(mu_y);
	mkl_free(A);
	mkl_free(rotation);
	mkl_free(translation);
	mkl_free(A1);
	mkl_free(A2);
	mkl_free(U);
	mkl_free(V);
	mkl_free(W);
	mkl_free(temp_matrix);
	mkl_free(C);
	mkl_free(rotation_temp);
	mkl_free(translation_temp);
	mkl_free(sigma_temp1);
	//mkl_free(sigma_temp2);
	mkl_free(sigma_temp3);
	//mkl_free(sigma_temp4);
	mkl_free(sigma_temp5);
	mkl_free(x_squared);
	mkl_free(y_init_squared);


	return rigidTr;
}


void Registration::optimizeShapeAndPoseCoefMKL(double* x_min, 
											   int shapeCoefLength,
											   int poseCoefLength,
											   MultiObjModelMKL *model, 
											   double* Px, 
											   double* P1, 
											   int M,
											   double* gammaPose, 
											   double* gammaShape)
{
	struct functor : public vnl_cost_function
	{
		double *x_min;
		double *P1;
		double *Px;
		
		int shapeCoefLength;
		int poseCoefLength;
		int M;
		
		MultiObjModelMKL *model;
		double *GammaShape;
		double *GammaPose;

		double f(vnl_vector<double> const& params)
		{
			//cout << "params: " << params[0] << endl;
			//getchar();


			double value;


			double one = 1.0;
			double three = 3.0;
			double zero = 0.0;
			
			double* poseCoef = (double*)mkl_malloc((params.size() - shapeCoefLength) * 1 * sizeof(double), 64);
			double* shapeCoef = (double*)mkl_malloc(shapeCoefLength * 1 * sizeof(double), 64);
			
			double* identity = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
			for(int i = 0; i < 16; i++) identity[i] = 0.0;
			for(int i = 0; i < 16; i+=5) identity[i] = 1.0;
			
			// shapeCoef->set_column(0,params.extract(shapeCoefLength, 0));
			for(int i = 0; i < shapeCoefLength; i++)
				shapeCoef[i] = params(i);

			// poseCoef->set_column(0, params.extract(params.size() - shapeCoefLength, shapeCoefLength));
			int len = params.size() - shapeCoefLength;
			for(int i = 0; i < len; i++)
				poseCoef[i] = params(shapeCoefLength + i);
//cout << "opt1" << endl;
			// model->transferModel(shapeCoef, poseCoef, identity);
			model->transferModel(shapeCoef, shapeCoefLength, poseCoef, poseCoefLength, identity);
			model->extractVisiblePoints();
//cout << "opt2" << endl;
			// tmpSq.set_size(model->moved_shape_visible_->rows(), model->moved_shape_visible_->cols());
			double* tmpSq = (double*)mkl_malloc(model->get_visible_point_nr_() * 3 * sizeof(double), 64);

			
			// tmpSq = model->moved_shape_visible_->apply(squarefunction);
			vdMul(model->get_visible_point_nr_() * 3, model->moved_shape_visible_, model->moved_shape_visible_, tmpSq);
			
			// vnl_matrix<double> sumMask(3, 1, 1);
			double* sumMask = (double*)mkl_malloc(3 * 1 * sizeof(double), 64);
			for (int i = 0; i < 3; i++)
				sumMask[i] = 1.0;

			// movedShapeSum = tmpSq * sumMask;
			double* movedShapeSum = (double*)mkl_malloc(model->get_visible_point_nr_() * 1 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						model->get_visible_point_nr_(), 1, 3, one, tmpSq, 3, sumMask, 1, zero, movedShapeSum, 1);
			//cout << "opt3" << endl;
			// tmpMul.set_size(1, 1);
			// tmpMul = vnl_trace(Px->transpose() * *model->moved_shape_visible_); // Px = 5x3, model.movedShape = 7500x3 // (3x5) * (7500x3) = error
			double* tmpMul = (double*)mkl_malloc(1 * 1 * sizeof(double), 64);
			double* tmpMul1 = (double*)mkl_malloc(3 * 3 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,			
						3, 3, M, one, Px, 3, model->moved_shape_visible_, 3, zero, tmpMul1, 3);
			tmpMul[0] = 0.0;
			for(int i = 0; i < 9; i+=4)
				tmpMul[0] += tmpMul1[i];


			// value = (P1->transpose() * movedShapeSum).get(0,0) - 2*tmpMul.get(0,0) + (shapeCoef->transpose()**GammaShape**shapeCoef).get(0,0) + (poseCoef->transpose()**GammaPose**poseCoef).get(0,0); 
			//		1. (P1->transpose() * movedShapeSum).get(0,0)
			double* value_tmp1 = (double*)mkl_malloc(1 * 1 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
						1, 1, M, one, P1, 1, movedShapeSum, 1, zero, value_tmp1, 1);
			value = value_tmp1[0];
//cout << "opt4" << endl;
			//		2. 2*tmpMul.get(0,0)
			value -= 2 * tmpMul[0];

			//		3. (shapeCoef->transpose()**GammaShape**shapeCoef).get(0,0)
			double* value_tmp2 = (double*)mkl_malloc(1 * (params.size() - shapeCoefLength) * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
						1, (params.size() - shapeCoefLength), shapeCoefLength, one, shapeCoef, 1, GammaShape, (params.size() - shapeCoefLength), zero, value_tmp2, (params.size() - shapeCoefLength));
		//cout << "opt5" << endl;	
			double* value_tmp2a = (double*)mkl_malloc(1 * 1 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						1, 1, shapeCoefLength, one, value_tmp2, (params.size() - shapeCoefLength), shapeCoef, 1, zero, value_tmp2a, 1);
			value += value_tmp2a[0];
//cout << "opt6" << endl;
			//		4. (poseCoef->transpose()**GammaPose**poseCoef).get(0,0)
			double* value_tmp3 = (double*)mkl_malloc(1 * poseCoefLength * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
						1, poseCoefLength, (params.size() - shapeCoefLength), one, poseCoef, (params.size() - shapeCoefLength), GammaPose, poseCoefLength, zero, value_tmp3, poseCoefLength);
			//cout << "opt6a" << endl;
			double* value_tmp3a = (double*)mkl_malloc(1 * 1 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						1, 1, (params.size() - shapeCoefLength), one, value_tmp3, poseCoefLength, poseCoef, 1, zero, value_tmp3a, 1);

	
			value += value_tmp3a[0];
			//cout << "opt7" << endl;
			mkl_free_buffers();
			mkl_free(poseCoef);
			mkl_free(shapeCoef);
			mkl_free(identity);
			mkl_free(tmpSq);
			mkl_free(tmpMul);
			mkl_free(tmpMul1);
			mkl_free(sumMask);
			mkl_free(movedShapeSum);
			mkl_free(value_tmp1);
			mkl_free(value_tmp2);
			mkl_free(value_tmp2a);
			mkl_free(value_tmp3);
			mkl_free(value_tmp3a);
	
			return value;
		}

		functor() : vnl_cost_function(1) {}
	};

	functor f;	
	f.Px = Px;
	f.P1 = P1;
	f.x_min = x_min;		
	
	f.GammaPose = gammaPose;
	f.GammaShape = gammaShape;

	f.shapeCoefLength = shapeCoefLength;
	f.poseCoefLength = poseCoefLength;
	f.M = M;
	f.model = model;

	// Minimization steps - will use soemthing else
	vnl_amoeba minimizer(f);	

	minimizer.set_max_iterations(1000);					
	minimizer.set_x_tolerance(0.01);
	minimizer.set_f_tolerance(0.001);

	vnl_vector<double>* x_min_local = new vnl_vector<double>(shapeCoefLength + poseCoefLength);
	for(int i = 0; i < shapeCoefLength + poseCoefLength; i++) {
		x_min_local->put(i, x_min[i]);
	}
	minimizer.minimize(*x_min_local);

	for(int i = 0; i < shapeCoefLength + poseCoefLength; i++) {
		x_min[i] = x_min_local->get(i);
	}

	free ((void*) x_min_local);
	
}
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

		

		temp_array_size = 16;
		cblas_dscal(temp_array_size, theta[i], transform_tmp, onei);
		

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
		cblas_dcopy(temp_array_size, transform_tmp2, 1, &Tr[i*16], 1);
		
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
		
		/*
		cout << "value2" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << value2[p*4 + o] << " ";
		cout << endl;
		}
		getchar();
		*/
		
		temp_array_size = 16;
		cblas_dcopy(temp_array_size, value2, 1, value, 1);
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

	// value = value * poseMean;	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				4, 4, 4, one, value, 4, poseMean, 4, zero, value2, 4);
	cblas_dcopy(16, value2, 1, value, 1);

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
	for(int i = 0; i < 4 * 4 * L; i++) deriv_intermd[i] = 0.0;

	
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

	/*
	cout << "deriv" << endl;
	for(int i = 0; i < 4 * 2; i++) {
		for(int j = 0; j < 4; j++)
			cout << deriv[i*4 + j] << " ";
		cout << endl;

	}
	getchar();
	*/
	
	mkl_free_buffers();
	//mkl_thread_free_buffers();
	mkl_free(Tr);
	mkl_free(transform_tmp);
	mkl_free(transform_tmp2);
	mkl_free(value2);
	mkl_free(deriv_intermd);
	
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
	

	// Need to add in the calculation in order to return TrY's values
	
	
	
	// TrY = bsxfun(@plus, y*Tr(1:3,1:3)', Tr(1:3,4)');
	double* TrY = (double*)mkl_malloc(N * 3 * sizeof(double), 64);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, 3, 3, one, y, 3, Tr, 4, zero, TrY, 3);
	
	double* Nx1 = (double*)mkl_malloc(N * 1 * sizeof(double), 64);
	for(int i = 0; i < N * 1; i++) Nx1[i] = 1.0;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, 3, 1, one, Nx1, 1, &Tr[3], 4, one, TrY, 3);
	
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
	
	
	for(int p = 0; p < 2; p++) {
		cout << deriv[p] << " ";
		
	}
	cout << endl;
	cout << "deriv " << endl;
	getchar();
	
	mkl_free_buffers();
	//mkl_thread_free_buffers();
	mkl_free(TrDeriv);
	mkl_free(Tr);
	mkl_free(basis_trim);
	mkl_free(TrY);
	mkl_free(Nx1);
	mkl_free(TrDerivY);
	mkl_free(tmp1);
	mkl_free(tmp2);
	mkl_free(sum1);
	mkl_free(sum2);
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



vnl_matrix<double>* Registration::get_registered_model(void) 
{
	return model_->moved_shape_;
}

double* Registration::get_registered_model_MKL(void)
{
	return model_MKL_->moved_shape_;
}

double Registration:: get_shape_coef_length_(void)
{
	return shape_coef_length_;
}

double Registration::get_pose_coef_length_(void)
{
	return pose_coef_length_;
}
