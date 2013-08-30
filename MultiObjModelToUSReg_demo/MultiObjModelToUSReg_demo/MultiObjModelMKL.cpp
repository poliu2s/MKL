// MKL implementation of MultiObjModel

#define DIM 3

//#include <vcl_fstream.h>
//#include <vcl_iostream.h>
//#include <vcl_sstream.h>
//#include <vnl/vnl_matrix_exp.h>
#include "MultiObjModelMKL.h"
#include <Windows.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <mkl.h>
#include "stdafx.h"
#include <string>

using namespace std;


void MultiObjModelMKL::matrix_exponential(double* matrix, double* result)
{
	
	#ifdef TRACE
	//mkl_disable_fast_mm();
	//mkl_free_buffers();
	//mkl_thread_free_buffers();

	MKL_INT64 AllocatedBytes;
	int N_AllocatedBuffers;
	AllocatedBytes = mkl_mem_stat(&N_AllocatedBuffers);
	cout << "allocated: " << AllocatedBytes << endl;
	#endif
	
	int accuracy = 10;

	// Scaling
	int N = 4;

	//cout << "me1" << endl;
	//M_small = M/(2^N);
	double* M_small = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* M_power = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	//double* M_power1 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	cblas_dcopy(16, matrix, 1, M_small, 1);
	cblas_dscal(16, 1/ pow(2.0, (double)N), M_small, 1);
	/*
	for(int i = 0; i < 16; i++) {
		M_small[i] = matrix[i] / pow(2.0, (double)N);
	}
	*/
	cblas_dcopy(16, M_small, 1, M_power, 1);
	

	// Exp part
	double* m_exp1 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);

	for(int i = 0; i < 16; i++) result[i] = 0.0;
	for(int i = 0; i < 16; i+=5) result[i] = 1.0;

	double* result_tmp = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* tmpM1 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);

	double factorial_i = 1.0;
	for(int i = 1; i < accuracy; i++) {
		factorial_i = factorial_i * i;

		#ifdef TRACE
		//m_exp = m_exp + M_power/factorial(i);
		//for(int x = 0; x < 16; x++) tmpM1[x] = M_power[x] / factorial_i;
		#endif
		cblas_dcopy(16, M_power, 1, tmpM1, 1);
		cblas_dscal(16, 1/factorial_i, tmpM1, 1); 
		#ifdef TRACE
		/*
		cblas_dcopy(16, result, 1, result_tmp, 1);
		vdAdd(16, result_tmp, tmpM1, result);
		*/
		#endif
		vdAdd(16, result, tmpM1, result);

		

		//M_power = M_power * M_small;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, 1.0, M_power, 4, M_small, 4, 0.0, M_power, 4);
		#ifdef TRACE
		//cblas_dcopy(16, M_power1, 1, M_power, 1);
		#endif

	}

	//cout << "me2" << endl;
	
	// Squaring step
	for(int i = 0; i < N; i++) {
		// m_exp = m_exp*m_exp;
		cblas_dcopy(16, result, 1, m_exp1, 1);

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, 1.0, m_exp1, 4, m_exp1, 4, 0.0, result, 4);
	}

	#ifdef TRACE
	//AllocatedBytes = mkl_mem_stat(&N_AllocatedBuffers);
	//cout << "allocated: " << AllocatedBytes << endl;
	#endif

	//cout << "me3" << endl;
	mkl_free_buffers();
	//mkl_thread_free_buffers();

	//AllocatedBytes = mkl_mem_stat(&N_AllocatedBuffers);
	//cout << "allocated: " << AllocatedBytes << endl;

	mkl_free(M_small);
	//cout << "me3a" << endl;
	mkl_free(m_exp1);
	//cout << "me3b" << endl;
	mkl_free(M_power);
	//cout << "me3c" << endl;
	//mkl_free(M_power1);
	//cout << "me3d" << endl;
	mkl_free(tmpM1);
	//cout << "me3e" << endl;

	//cout << "me3f" << endl;
	mkl_free(result_tmp);
	//cout << "me4" << endl;

	#ifdef TRACE
	//AllocatedBytes = mkl_mem_stat(&N_AllocatedBuffers);
	//cout << "allocated: " << AllocatedBytes << endl;
	#endif

	return;
}

#ifdef TRACE
double* MultiObjModelMKL::matrix_exponential_old(double* matrix)
{
	//mkl_disable_fast_mm();

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
	cblas_dcopy(16, M_small, 1, M_power, 1);

	double* tmpM1 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);

	double factorial_i = 1.0;
	for(int i = 1; i < accuracy; i++) {
		factorial_i = factorial_i * i;

		//m_exp = m_exp + M_power/factorial(i);
		for(int x = 0; x < 16; x++) tmpM1[x] = M_power[x] / factorial_i;

		vdAdd(sxtn, m_exp, tmpM1, m_exp);

		//M_power = M_power * M_small;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, 1.0, M_power, 4, M_small, 4, 0.0, M_power1, 4);
		cblas_dcopy(16, M_power1, 1, M_power, 1);

	}

	// Squaring step
	for(int i = 0; i < N; i++) {
		// m_exp = m_exp*m_exp;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, 1.0, m_exp, 4, m_exp, 4, 0.0, m_exp1, 4);
		cblas_dcopy(16, m_exp1, 1, m_exp, 1);
	}

	//mkl_free_buffers();
	//mkl_thread_free_buffers();

	mkl_free(M_small);
	mkl_free(m_exp1);
	mkl_free(M_power);
	mkl_free(M_power1);
	mkl_free(tmpM1);

	return m_exp;
}
#endif


// Constructor
MultiObjModelMKL::MultiObjModelMKL()	
{
	obj_ind_ = 0;
	shape_mean_ = 0;
	shape_sigma_ = 0;
	pose_mean_ = 0;
}

// Destructor
MultiObjModelMKL::~MultiObjModelMKL()
{
	mkl_free(obj_ind_);
	mkl_free(shape_mean_);
	mkl_free(shape_sigma_);
	mkl_free(pose_mean_);
}

void MultiObjModelMKL::readModel()
{
	//vcl_ifstream inFile(filename.c_str());

	std:ifstream inFile("C:/MultiObjModelToUSReg/data/EuclideanModel.txt");

	
	
	
	if(!inFile.is_open()) {
		cout << "Unable to open file.";
		return;
	}
	
	//get obj_nr_
	readIntFromFile(inFile, &obj_nr_);
	cout << obj_nr_ <<endl;
	
	//get obj_ind_ array. should be a objNrx2 matrix
	obj_ind_ = (int*)mkl_malloc(obj_nr_ * 2 * sizeof(int), 64 );
	if(!readMatrixFromFile(inFile, obj_ind_, obj_nr_, 2)) {
		cout << "Unable to open file.";
		return;
	}


	//read the shape_mean_ values. shapeMean is a objInd(end, 1)x3 matrix.
	int shapeMeanRows = obj_ind_[obj_nr_*2 - 1];
	shape_mean_ = (double*)mkl_malloc(shapeMeanRows * 3 * sizeof(double), 64 );
	if(!readMatrixFromFile(inFile, shape_mean_, shapeMeanRows, 3)) {
		cout << "Unable to read Shape Mean.";
		return;
	}
	
	
	//get shape_modes_nr_ from the file
	readIntFromFile(inFile, &shape_modes_nr_);

	
	//get shape_sigma_ from file. Is a shapeModesNrx1 matrix.
	shape_sigma_ = (double*)mkl_malloc(shape_modes_nr_ * 1 * sizeof(double), 64);
	if(!readMatrixFromFile(inFile, shape_sigma_, shape_modes_nr_, 1)) {
		cout << "Unable to read shape sigma.";
		return;
	}

	
	//get shape_features_ from file. Is a 3*shapeMeanRows x shapeModesNr matrix.
	shape_feature_rows_ = 3 * shapeMeanRows;
	shape_feature_ = (double*)mkl_malloc(shape_feature_rows_ * shape_modes_nr_ * sizeof(double), 64);
	if(!readMatrixFromFile(inFile, shape_feature_, shape_feature_rows_, shape_modes_nr_)) {
		cout << "Unable to read shape feature.";
		return;
	}

	
	//get pose_mean_ from file. Is a 3D matrix: obj_nr_x4x4 matrix.
	//This 3D matrix will be stored as a vector of 4x4 matrices.
	int pose_mean_nr_ = 4 * 4 * obj_nr_;
	pose_mean_ = (double*)mkl_malloc(pose_mean_nr_ * sizeof(double), 64);
	if(!readMatrixFromFile(inFile, pose_mean_, obj_nr_ * 4, 4)) {
		cout << "Unable to read pose mean.";
		return;
	}

	readIntFromFile(inFile, &pose_modes_nr_);

	
	//get pose_sigma_ from file. Is a shapeModesNr x 1 matrix.
	pose_sigma_ = (double*)mkl_malloc(pose_modes_nr_ * 1 * sizeof(double), 64);
	if(!readMatrixFromFile(inFile, pose_sigma_, pose_modes_nr_, 1)) {
		cout << "Unable to read pose sigma.";
		return;
	}

	
	//get pose_feature_ from file. Is a 7*objNr x shapeModesNr.
	pose_feature_rows_ = 7*obj_nr_;
	pose_feature_ = (double*)mkl_malloc(pose_feature_rows_ * pose_modes_nr_ * sizeof(double), 64);
	if(!readMatrixFromFile(inFile, pose_feature_, pose_feature_rows_, pose_modes_nr_)) {
		cout << "Unable to read pose feature.";
		return;
	}
	
	//get faces_
	readIntFromFile(inFile, &faces_nr_);
	int facesRows = faces_nr_;
	faces_ = (double*)mkl_malloc(facesRows * 3 * sizeof(double), 64);
	if(!readMatrixFromFile(inFile, faces_, facesRows, 3)) {
		cout << "Unable to read faces.";
		return;
	}


	//get visible_point_nr_
	readIntFromFile(inFile, &visible_point_nr_);
	int visiblePointRows = visible_point_nr_;
	visible_point_ = (double*)mkl_malloc(visiblePointRows * 1 * sizeof(double), 64);
	if(!readMatrixFromFile(inFile, visible_point_, visiblePointRows, 1)) {
		cout << "Unable to visible points.";
		return;
	}

	
	// Allocate memory for the bigger class vars of size n
	int n = obj_ind_[obj_nr_*2 - 1];
	shape_deformed_ = (double*)mkl_malloc(n * 3 * sizeof(double), 64);
	pose_deformed_ = (double*)mkl_malloc(n * 3 * sizeof(double), 64);
	moved_shape_ = (double*)mkl_malloc(n * 3 * sizeof(double), 64);

	moved_shape_visible_ = (double*)mkl_malloc(visible_point_nr_ * 3 * sizeof(double), 64);
	pose_deformed_visible_ = (double*)mkl_malloc(visible_point_nr_ * 3 * sizeof(double), 64);

	inFile.close();
	return;
	
}

bool MultiObjModelMKL::readIntFromFile(ifstream &inFile ,int *i)
{
	string dummy;
	int intVal;

	do
	{
		inFile >> intVal;
		if(inFile.good())
		{
			*i = intVal;
			return true;
		}
		//what we read was not a valid integer. Clear stream of errors and keep looking for an int.
		inFile.clear();
		inFile >> dummy;	//throw away the data if it is not the value we are looking for

	} while(!inFile.eof());

	//if we reach here, then we have reached the end of the file without finding anything
	return false;

}

bool MultiObjModelMKL::readMatrixFromFile(ifstream &inFile, int* m, int m_rows, int m_cols)
{
	string dummy;
	if(!m)
		return false;

	//advance the stream to the start of the data by reading the first value
	//readIntFromFile(inFile, &(*m)(0,0));

	for(int i = 0; i < m_rows; i++)
		for(int j = 0; j < m_cols; j++)
		{
			//this will keep reading and throwing away data until we have successfully read an int
			do
			{
				inFile >> m[i*m_cols + j];
				if(inFile.good())
					break;
				inFile.clear();
				inFile >> dummy;
			} while (!inFile.eof());	
		}

	//at this point, m is filled with the values from the file
	return true;
}

bool MultiObjModelMKL::readMatrixFromFile(ifstream &inFile, double* m, int m_rows, int m_cols)
{
	string dummy;
	if(!m)
		return false;

	for(int i = 0; i < m_rows; i++)
		for(int j = 0; j < m_cols; j++){

			//this will keep reading and throwing away data until we have successfully read a double
			do
			{
				inFile >> m[i*m_cols + j];
				if(inFile.good())
					break;
				inFile.clear();
				inFile >> dummy;
			} while (!inFile.eof());	
		}

	//at this point, m is filled with the values from the file
	return true;
}


/*
bool MultiObjModelMKL::readIntFromFile(vcl_ifstream &inFile ,int *i)
{
	vcl_string dummy;
	int intVal;

	do
	{
		inFile >> intVal;
		if(inFile.good())
		{
			*i = intVal;
			return true;
		}
		//what we read was not a valid integer. Clear stream of errors and keep looking for an int.
		inFile.clear();
		inFile >> dummy;	//throw away the data if it is not the value we are looking for

	} while(!inFile.eof());

	//if we reach here, then we have reached the end of the file without finding anything
	return false;
}


bool MultiObjModelMKL::readMatrixFromFile(vcl_ifstream &inFile, int* m, int m_rows, int m_cols)
{
	vcl_string dummy;
	if(!m)
		return false;

	//advance the stream to the start of the data by reading the first value
	//readIntFromFile(inFile, &(*m)(0,0));

	for(int i = 0; i < m_rows; i++)
		for(int j = 0; j < m_cols; j++)
		{
			//this will keep reading and throwing away data until we have successfully read an int
			do
			{
				inFile >> m[i*m_cols + j];
				if(inFile.good())
					break;
				inFile.clear();
				inFile >> dummy;
			} while (!inFile.eof());	
		}

	//at this point, m is filled with the values from the file
	return true;
}

bool MultiObjModelMKL::readMatrixFromFile(vcl_ifstream &inFile, double* m, int m_rows, int m_cols)
{
	vcl_string dummy;
	if(!m)
		return false;

	for(int i = 0; i < m_rows; i++)
		for(int j = 0; j < m_cols; j++){

			//this will keep reading and throwing away data until we have successfully read a double
			do
			{
				inFile >> m[i*m_cols + j];
				if(inFile.good())
					break;
				inFile.clear();
				inFile >> dummy;
			} while (!inFile.eof());	
		}

	//at this point, m is filled with the values from the file
	return true;
}
*/

void MultiObjModelMKL::writeModel(MultiObjModelMKL *model) {
	
	ofstream outFile("movedShape.txt");
	
	if (outFile.is_open()){
		outFile << "# multi object shape+pose model \nobjNr " << model->obj_nr_ << "\nobject index";
		
		for (int i = 0; i < model->obj_nr_; i++)
			outFile << model->obj_ind_[i*2] << " " << model->obj_ind_[i*2 + 1] << endl;
		
		outFile << "mean shape\n" ;

		for (int i = 0; i < model->obj_ind_[obj_nr_*2 - 1]; i++)
			outFile << model->shape_mean_[i*3] << " " << model->shape_mean_[i*3 + 1] << " " << model->shape_mean_[i*3 + 2] << endl;
		
		outFile << "modesNr " << model->shape_modes_nr_ << " \nshape sigma" << endl;
		for(int i = 0; i < shape_modes_nr_; i++)
			outFile << model->shape_sigma_[i] << endl;
		
		outFile << "shape features" << endl;
		for (int i = 0; i < model->shape_feature_rows_; i++) {
			for (int j = 0; j < model->shape_modes_nr_; j++) {
				outFile << model->shape_feature_[i*model->shape_modes_nr_ + j] << " ";

			}
			outFile << endl;
		}		
		
		outFile << "pose mean" << endl;
		for (int i = 0; i < model->obj_nr_*4; i++) {
			for(int j = 0; j < 4; j++)
				outFile << model->pose_mean_[i*4+j] << " ";
			outFile << endl;
		}

		outFile << "modesNr " << model->pose_modes_nr_
			    << "\npose sigma" << endl;
		
		for(int i = 0; i < pose_modes_nr_; i++)
			outFile << model->pose_sigma_[i] << endl;


		outFile << "pose features " ;		
		for(int i = 0; i < pose_feature_rows_; i++) {
			outFile << endl;
			for (int j = 0; j < pose_modes_nr_; j++) 
				outFile << model->pose_feature_[i*pose_modes_nr_ + j] << " ";
			
		}
		outFile << endl;

		outFile << "Faces: " << faces_nr_ << endl;
		for(int i = 0; i < faces_nr_; i++) {
			for(int j = 0; j < 3; j++) 
				outFile << faces_[i*3 + j] << " ";
			outFile << endl;
		}
		
		outFile << "Visible Point Nr " << visible_point_nr_ << endl;
		for(int i = 0; i < visible_point_nr_; i++)
			outFile << visible_point_[i] << endl;
	}


	
	return;
}



// Transfers the multi-object pose+shape model
// Inputs:
//	 shapeCoeff - an Nx1 matrix
//   poseCoeff - Mx1 matrix
//   rigidTransform - 4x4 matrix
void MultiObjModelMKL::transferModel(double* shapeCoeff, int shapeCoeff_rows, double* poseCoeff, int poseCoeff_rows, double* rigidTransform) {
	
	int temp_array_size;
	double zero = 0.0;
	double one = 1.0;
	int onei = 1;
	int three = 3;
	
	if (shapeCoeff_rows == 0) {
		cblas_dcopy(obj_ind_[obj_nr_*2 - 1] * 3, shape_mean_, 1, shape_deformed_, 1);

	} else {
		double* A = (double*)mkl_malloc(obj_ind_[obj_nr_*2 - 1] * 3 * sizeof(double), 64);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
			  obj_ind_[obj_nr_*2 - 1]*3, 1, shapeCoeff_rows, one, shape_feature_, shape_modes_nr_, shapeCoeff, 1, zero, A, 1);  

		temp_array_size = obj_ind_[obj_nr_*2 - 1] * 3;
		vdadd(&temp_array_size, shape_mean_, A, shape_deformed_);
		mkl_free(A);

	}

	// --------------------------------------------------------------

	double* Tr = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* Tr1 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* TransMat = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* vec = (double*)mkl_malloc(7 * 1 * sizeof(double), 64);
	double* temp = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* temp2 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);

	for(int l = 0; l < obj_nr_; l++)
	{
		//transfer with pose coefficients, result is poseDeformed
		
		//Tr.set_identity();
		for(int i = 0; i < 16; i++) Tr[i] = 0.0;
		for(int i = 0; i < 16; i+=5) Tr[i] = 1.0;

		
		if(poseCoeff != 0)
		{
			for(int i = 0; i < poseCoeff_rows; i++)
			{
				for(int x = 0; x < 7; x++)
					vec[x] = pose_feature_[7*l*pose_modes_nr_ + pose_modes_nr_*x + i];
				getTransformationMatrixFromVector(vec, 1, TransMat);
				
				//vnl_matrix_exp((*poseCoeff)(i,0)*TransMat, tmp, 0.0000001);
				for(int x = 0; x < 16; x++)
					temp[x] = TransMat[x] * poseCoeff[i];
				matrix_exponential(temp, temp2);
				cblas_dcopy(16, temp2, 1, temp, 1);
		

				// Tr = Tr * tmp;
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
							4, 4, 4, one, Tr, 4, temp, 4, zero, Tr1, 4);
				cblas_dcopy(16, Tr1, 1, Tr, 1);
			}
		}

		// Tr = Tr * (*pose_mean_[l]);
		temp_array_size = 16;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, one, Tr, 4, &(pose_mean_[l*16]), 4, zero, temp, 4);
		cblas_dcopy(16, temp, 1, Tr, 1);

		//extract the rotation and translation components from Tr
		// int objStartIndex = obj_ind_->get(l, 0);
		int objStartIndex = obj_ind_[l*2];

		// int objEndIndex = obj_ind_->get(l, 1);
		int objEndIndex = obj_ind_[l*2 + 1];

		// int sz = objEndIndex - objStartIndex + 1;
		int sz = objEndIndex - objStartIndex + 1;

		// pose_deformed = shape_deformed_->extract(sz, 3, objStartIndex-1, 0) * rotation.transpose();
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
					sz, 3, 3, one, &shape_deformed_[3 * (objStartIndex-1)], 3, Tr, 4, zero, &pose_deformed_[3 * (objStartIndex-1)], 3);
	
		// translation
		double* arrayone = (double*)mkl_malloc(sz * 1 * sizeof(double), 64);
		for(int i = 0; i < sz; i++) arrayone[i] = 1.0;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
					sz, 3, 1, one, arrayone, 1, &Tr[3], 4, one, &pose_deformed_[3 * (objStartIndex-1)], 3);
		mkl_free(arrayone);
	}
	
	//moved_shape_->update(*pose_deformed_); 
	temp_array_size = obj_ind_[obj_nr_*2 - 1] * 3; // obj_ind_[obj_nr_*2 - 1]
	cblas_dcopy(temp_array_size, pose_deformed_, 1, moved_shape_, 1);
	int entireArray = obj_ind_[obj_nr_*2 - 1];
	if(rigidTransform != 0)
	{
		/*
		ALGORITHM:
		rot = rigidTrans(0:2,0:2) - 3x3 matrix
		trans = rigidTrans(0:2,3) - 3x1 matrix
		movedShape = (rot * shapeDeformed.transpose()) + trans
		*/
		
		// rotation
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
					entireArray, 3, 3, one, &pose_deformed_[0], 3, rigidTransform, 4, zero, &moved_shape_[0], 3);
	
		// translation
		double* arrayone = (double*)mkl_malloc(entireArray * 1 * sizeof(double), 64);
		for(int i = 0; i < entireArray; i++) arrayone[i] = 1.0;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
					entireArray, DIM, 1, one, arrayone, 1, &rigidTransform[3], 4, one, &moved_shape_[0], DIM);
		mkl_free(arrayone);
	}

	mkl_free_buffers();
	mkl_thread_free_buffers();
	
	mkl_free(Tr);
	mkl_free(Tr1);
	mkl_free(TransMat);
	mkl_free(vec);
	mkl_free(temp);
	mkl_free(temp2);

	mkl_free_buffers();
	mkl_thread_free_buffers();

	/*
	cout << "Moved Shape in Transfer: " << endl;
	for (int j = 0; j < entireArray; j++) {
		for(int i = 0; i < 3; i++)
			cout << moved_shape_[entireArray*3-3 +i] << " ";
		cout << endl;
	}
	*/
	return;
}

//converts a 7xN vector vec into a 4x4 transformation matrix
void MultiObjModelMKL::getTransformationMatrixFromVector(double* vec, int N, double* out) {
	out[0*4 + 0] = vec[6*N]; out[0*4 + 1] = -vec[5*N]; out[0*4 + 2] = vec[4*N]; out[0*4 + 3] = vec[0*N];
	out[1*4 + 0] = vec[5*N]; out[1*4 + 1] = vec[6*N]; out[1*4 + 2] = -vec[3*N]; out[1*4 + 3] = vec[1*N];
	out[2*4 + 0] = -vec[4*N]; out[2*4 + 1] = vec[3*N]; out[2*4 + 2] = vec[6*N]; out[2*4 + 3] = vec[2*N];
	out[3*4 + 0] = 0; out[3*4 + 1] = 0; out[3*4 + 2] = 0; out[3*4 + 3] = 0;
	
	return;
}

double* MultiObjModelMKL::extractVisiblePoints(void)
{
	int visibleElement;
	int one = 1;

	for (int i = 0; i < visible_point_nr_; i++) {
		visibleElement = visible_point_[i] - 1;		// -1 is to compensate for difference between Matlab and C++
		
		// moved_shape_visible_->set_row(i, moved_shape_->get_row(visibleElement)); 
		cblas_dcopy(3, &moved_shape_[visibleElement*3], 1, &moved_shape_visible_[i*3], 1);
		
		// Extra copy operation?
		// pose_deformed_visible_->set_row(i, pose_deformed_->get_row(visibleElement)); 
		cblas_dcopy(3, &pose_deformed_[visibleElement*3], 1, &pose_deformed_visible_[i*3], 1);
	}

	return moved_shape_visible_;
}


// Accessor methods for shape_sigma_, pose_sigma_, and faces_
double* MultiObjModelMKL::get_shape_sigma_(void)
{
	return shape_sigma_;
}

int MultiObjModelMKL::get_shape_sigma_rows(void)
{
	return shape_modes_nr_;
}

double* MultiObjModelMKL::get_pose_sigma_(void)
{
	return pose_sigma_;
}

int MultiObjModelMKL::get_pose_sigma_rows(void)
{
	return pose_modes_nr_;
}

double* MultiObjModelMKL::get_faces_(void)
{
	return faces_;
}

int MultiObjModelMKL::get_faces_rows_(void)
{
	return faces_nr_;
}

double* MultiObjModelMKL::get_shape_mean_(void)
{
	return shape_mean_;
}

int MultiObjModelMKL::get_moved_shape_rows_(void)
{
	return obj_ind_[obj_nr_*2 - 1];
}

int MultiObjModelMKL::get_visible_point_nr_(void)
{
	return visible_point_nr_;
}

int* MultiObjModelMKL::get_obj_ind_(void) 
{
	return obj_ind_;
}

int MultiObjModelMKL::get_obj_nr_(void)
{
	return obj_nr_;
}

double* MultiObjModelMKL::get_pose_mean_(void)
{
	return pose_mean_;
}

double* MultiObjModelMKL::get_pose_feature_(void)
{
	return pose_feature_;
}

double* MultiObjModelMKL::get_shape_feature_(void)
{
	return shape_feature_;
}

int MultiObjModelMKL::get_pose_modes_nr_(void)
{
	return pose_modes_nr_;
}

int MultiObjModelMKL::get_shape_modes_nr_(void)
{
	return shape_modes_nr_;
}

double* MultiObjModelMKL::get_visible_point_(void)
{
	return visible_point_;
}