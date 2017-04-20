#ifndef DT_FEATURE_H
#define DT_FEATURE_H

/*
 * Handles a dense trajectory feature point
 * Check http://lear.inrialpes.fr/people/wang/dense_trajectories for feature format
 */

#include <iostream>
#include <sstream>
#include <string>

using namespace std;

#define TRAJ_DIM 30
#define HOG_DIM 96
#define HOF_DIM 108
#define MBHX_DIM 96
#define MBHY_DIM 96

class DTFeature_f	{
	public:
		DTFeature_f();
		DTFeature_f(string featureLine);
		DTFeature_f(const DTFeature_f &f);
		~DTFeature_f();
        DTFeature_f &operator=(const DTFeature_f &f);   // overload copy operator
	public:
		int frameNum;
		float mean_x;
		float mean_y;
		float var_x;
		float var_y;
		float length;
		float scale;
		float x_pos;
		float y_pos;
		float t_pos;
		float *traj;
		float *hog;
		float *hof;
		float *mbhx;
		float *mbhy;
};

#endif
