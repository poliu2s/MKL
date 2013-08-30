//Author: Changhoon Baek(45322096)
//Edited by: Po Liu (13623079)

#include "Registration.h"
#include <mkl.h>
#include "stdafx.h"

#ifdef TRACE
// VNL exp
#include <vnl/vnl_matrix_exp.h>
#include <vnl/vnl_matrix.h>
#endif


#define DIM 3

using namespace std;


//=================================================================================Functions=====//
double Registration::squarefunction(double x)
{
	return x*x;
}
double Registration::sqrtfunction(double x)
{
	return 1/sqrt(x);
}

double Registration::expfunction(double x)
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
	cout << name.c_str() << endl;
	getchar();

}
//===============================================================================Constructor=====//
Registration::Registration(){}


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

	#ifdef TRACE
	/*
	generated_point_ = new vnl_matrix<double>; 
	generated_point_prb_ = new vnl_matrix<double>; 
	mu_generated_point_ = new vnl_matrix<double>; //test
	mu_pose_deformed_ = new vnl_matrix<double>; //test
	Px_ = new vnl_matrix<double>;
	Pt1_ = new vnl_matrix<double>;
	P1_ = new vnl_matrix<double>;
	*/
	#endif
}

//================================================================================Destructor=====//
Registration::~Registration()
{
	
}

//=======================================================================Setters And Getters=====//
void Registration::set_max_iter_(int maxIter) {max_iter_ = maxIter;}
void Registration::set_sigma_(double sigma) {sigma_ = sigma;}
void Registration::set_sigma_threshold_(double sigma_threshold) {sigma_threshold_ = sigma_threshold;}
void Registration::set_gen_point_num_(int gen_point_num) {gen_point_num_ = gen_point_num;}
double Registration::get_gen_point_num_(void) { return gen_point_num_;}

double Registration::get_sigma_(void) {return sigma_;}

void Registration::set_outlier_(double outlier) {outlier_ = outlier;}



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
