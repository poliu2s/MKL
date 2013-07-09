// MKL implementation of MultiObjModel

#include <vcl_string.h>
#include <vnl/vnl_matrix.h>
#include <vcl_vector.h>

using namespace std;

class MultiObjModelMKL
{
public:
	MultiObjModelMKL();
	~MultiObjModelMKL();

	void readModel(vcl_string filename);
	void writeModel(MultiObjModelMKL* model);
	void transferModel(double* shapeCoeff, int shapeCoeff_rows, double* poseCoeff, int poseCoeff_rows, double* rigidTransform);
	static void writePolyDataInVTK(double* vertices, int vertices_rows, int D, double* faces, int faces_rows, string fileName);
	double* extractVisiblePoints(void);
	
	double *shape_deformed_, // shape_deformed: obj_ind_[obj_nr_*2 - 1] x 3 array
		   *pose_deformed_, // pose_deformed: obj_ind_[obj_nr_*2 - 1] x 3 array
		   *moved_shape_,
		   *moved_shape_visible_,
		   *pose_deformed_visible_; // moved_shape: obj_ind_[obj_nr_*2 - 1] x 3 array

	double* get_shape_sigma_(void);
	int get_shape_sigma_rows(void);
	double* get_pose_sigma_(void);
	int get_pose_sigma_rows(void);
	double* get_faces_(void);
	double* get_shape_mean_(void);
	int get_moved_shape_rows_(void);
	int get_faces_rows_(void);
	int get_visible_point_nr_(void);
	int* get_obj_ind_(void); 
	int get_obj_nr_(void);
	double* get_pose_feature_(void);
	double* get_pose_mean_(void);
	int get_pose_modes_nr_(void);

	static void getTransformationMatrixFromVector(double* vec, int N, double* out);
	static void matrix_exponential(double* matrix, double* result);
	static double* matrix_exponential_old(double* matrix);

private:

	bool readIntFromFile(vcl_ifstream &inFile ,int* i);
	bool readMatrixFromFile(vcl_ifstream &inFile ,int* m, int m_rows, int m_cols);
	bool readMatrixFromFile(vcl_ifstream &inFile ,double* m, int m_rows, int m_cols);
	
	
	int obj_nr_,
		shape_modes_nr_,
		pose_modes_nr_,
		faces_nr_,
		visible_point_nr_;

	int shape_feature_rows_, pose_feature_rows_;

	double* shape_mean_;
	double* shape_feature_;
	double* shape_sigma_;
	double* pose_sigma_;
	double* pose_feature_;
	double* faces_;
	double* visible_point_;

	int* obj_ind_;

	// pose_mean_ is a 1D matrix holding an array of 4x4 matrices
	// there are obj_nr_ 4x4 matrix stored sequentially
	double* pose_mean_;
};

