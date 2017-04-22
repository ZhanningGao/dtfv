/*
 * getIDTfeature.h
 *
 *  Created on: Apr 20, 2017
 *      Author: zp
Usage.cpp:
#include "getIDTfeature.h"
int getIdtFeature(char*,float[]);
int main() {
	float featuremass[MAXFEATURE];
	char* videoname = "./test_sequences/person01_boxing_d1_uncomp.avi";
	int n = getIdtFeature(videoname,featuremass);
}
 */
#ifndef GETIDTFEATURE_H_
#define GETIDTFEATURE_H_
#define TRAJ_DIM 30
#define HOG_DIM 96
#define HOF_DIM 108
#define MBHX_DIM 96
#define MBHY_DIM 96
#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"

long getIdtFeature(char *video, long MAXFEATURE, float * Traj,float *Hog, float *Hof, float *Mbhx, float *Mbhy) {
	printf("computing...\n");
	clock_t start,end;
	start = clock(); //CLOCK
	VideoCapture capture;
	capture.open(video);
	if (!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}
	long frame_num = 0;
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;

	//initialize track info
	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);
	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video);

	//initialize surf dector
	SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf(true, true);

	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;

	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;
	Mat flow, human_mask;

	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0);
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points

	int featureindex=0; //index of output track features
	float *tmpHog = new float[HOG_DIM];
	float *tmpHof = new float[HOF_DIM];
	float *tmpMbhx = new float[MBHX_DIM];
	float *tmpMbhy = new float[MBHY_DIM];

	float *pTraj, *pHog, *pHof, *pMbhx, *pMbhy;
	pTraj = Traj;
    pHog  = Hog;
    pHof  = Hof;
    pMbhx = Mbhx;
    pMbhy = Mbhy;

	while (true) {

		Mat frame;
		// get a new frame
		capture >> frame;
		if (frame.empty())
			break;

		//Process for the first frame(extract features only)
		if (frame_num == start_frame) {
			//Initialize sliding temporary image
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

			InitPry(frame, fscales, sizes);

			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, grey_pyr);
			BuildPry(sizes, CV_32FC2, flow_pyr);
			BuildPry(sizes, CV_32FC2, flow_warp_pyr);

			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_warp_pyr);

			xyScaleTracks.resize(scale_num);
			frame.copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);

			//dense sampling for all scale
			for (int iScale = 0; iScale < scale_num; iScale++) {
				if (iScale == 0)
					prev_grey.copyTo(prev_grey_pyr[0]);
				else
					resize(prev_grey_pyr[iScale - 1], prev_grey_pyr[iScale],
							prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				// dense sampling feature points
				std::vector<Point2f> points(0);
				DenseSample(prev_grey_pyr[iScale], points, quality,
						min_distance);

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for (int i = 0; i < points.size(); i++)
					tracks.push_back(
							Track(points[i], trackInfo, hogInfo, hofInfo,
									mbhInfo));
			}
			// compute polynomial expansion
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

			//mask filter
			human_mask = Mat::ones(frame.size(), CV_8UC1);
			//compute surf
			detector_surf.detect(prev_grey, prev_kpts_surf, human_mask);
			extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);

			frame_num++;
			continue;
		}
		init_counter++;
		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);
		// match surf features
		detector_surf.detect(grey, kpts_surf, human_mask);
		extractor_surf.compute(grey, kpts_surf, desc_surf);
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf,
				prev_pts_surf, pts_surf);
		extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);
		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

		MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow,
				human_mask);
		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf,
				prev_pts_all, pts_all);

		//RANCAC for Homography
		Mat H = Mat::eye(3, 3, CV_64FC1);
		if (pts_all.size() > 50) {
			std::vector<unsigned char> match_mask;
			Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1,
					match_mask);
			if (countNonZero(Mat(match_mask)) > 25)
				H = temp;
		}
		// compute optical flow after warp
		Mat H_inv = H.inv();
		Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
		MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // warp the second frame

		my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_warp_pyr,
				flow_warp_pyr, 10, 2);

		//calculate features in a certain scale
		for (int iScale = 0; iScale < scale_num; iScale++) {

			if (iScale == 0) //not to scale the image
				grey.copyTo(grey_pyr[0]);
			else
				resize(grey_pyr[iScale - 1], grey_pyr[iScale],
						grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

			int width = grey_pyr[iScale].cols;
			int height = grey_pyr[iScale].rows;
			//compute the integral histograms
			//hog
			DescMat* hogMat = InitDescMat(height + 1, width + 1, hogInfo.nBins);
			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);

			//hof
			DescMat* hofMat = InitDescMat(height + 1, width + 1, hofInfo.nBins);
			HofComp(flow_warp_pyr[iScale], hofMat->desc, hofInfo);
			//mbh

			DescMat* mbhMatX = InitDescMat(height + 1, width + 1,
					mbhInfo.nBins);
			DescMat* mbhMatY = InitDescMat(height + 1, width + 1,
					mbhInfo.nBins);
			MbhComp(flow_warp_pyr[iScale], mbhMatX->desc, mbhMatY->desc,
					mbhInfo);

			// track feature points in each scale separately

			std::list<Track>& tracks = xyScaleTracks[iScale];
			for (std::list<Track>::iterator iTrack = tracks.begin();
					iTrack != tracks.end();) {
				//track features by tracking flow matches
				int index = iTrack->index;
				Point2f prev_point = iTrack->point[index];
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0),
						width - 1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0),
						height - 1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2 * x];
				point.y = prev_point.y
						+ flow_pyr[iScale].ptr<float>(y)[2 * x + 1];

				if (point.x <= 0 || point.x >= width || point.y <= 0
						|| point.y >= height) {			//outside the sight
					iTrack = tracks.erase(iTrack);
					continue;
				}

				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2
						* x];
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2
						* x + 1];

				// get the descriptors for the feature point
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);
				GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				iTrack->addPoint(point);

				// if the trajectory achieves the maximal length(15) then output the features
				if (iTrack->index >= trackInfo.length) {
					std::vector<Point2f> trajectory(trackInfo.length + 1);
					for (int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i] * fscales[iScale];

					std::vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i] * fscales[iScale];

					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
					if (IsValid(trajectory, mean_x, mean_y, var_x, var_y,
							length) && IsCameraMotion(displacement)) {
						//featuremss.back()
						int tmpdim=0;

						if((featureindex+1) > MAXFEATURE)
						{
							printf("memory error.\n");
							delete [] tmpHog;
							delete [] tmpHof;
							delete [] tmpMbhx;
							delete [] tmpMbhy;
							return featureindex;
						}

							// output the trajectory
						for (int i = 0; i < trackInfo.length; ++i)
						{
							Traj[long(featureindex*TRAJ_DIM+i*2)]=displacement[i].x;
							Traj[long(featureindex*TRAJ_DIM+i*2+1)]=displacement[i].y;
						}

						OutputDesc(iTrack->hog, hogInfo, trackInfo,tmpHog);
						OutputDesc(iTrack->hof, hofInfo, trackInfo,tmpHof);
						OutputDesc(iTrack->mbhX, mbhInfo, trackInfo,tmpMbhx);
						OutputDesc(iTrack->mbhY, mbhInfo, trackInfo,tmpMbhy);

        				memcpy(pHog,  tmpHog, sizeof(float)*HOG_DIM); pHog += HOG_DIM;
        				memcpy(pHof,  tmpHof, sizeof(float)*HOF_DIM); pHof += HOF_DIM;
        				memcpy(pMbhx, tmpMbhx, sizeof(float)*MBHX_DIM); pMbhx += MBHX_DIM;
        				memcpy(pMbhy, tmpMbhy, sizeof(float)*MBHY_DIM); pMbhy += MBHY_DIM;


						featureindex++;
						//printf("%d\n",featureindex);
						// output the trajectory
						/*printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t", frame_num,
								mean_x, mean_y, var_x, var_y, length,
								fscales[iScale]);

						// for spatio-temporal pyramid
						printf("%f\t",
								std::min<float>(
										std::max<float>(
												mean_x / float(seqInfo.width),
												0), 0.999));
						printf("%f\t",
								std::min<float>(
										std::max<float>(
												mean_y / float(seqInfo.height),
												0), 0.999));
						printf("%f\t",
								std::min<float>(
										std::max<float>(
												(frame_num
														- trackInfo.length / 2.0
														- start_frame)
														/ float(seqInfo.length),
												0), 0.999));

						// output the trajectory
						for (int i = 0; i < trackInfo.length; ++i)
							printf("%f\t%f\t", displacement[i].x,
									displacement[i].y);

						PrintDesc(iTrack->hog, hogInfo, trackInfo);
						PrintDesc(iTrack->hof, hofInfo, trackInfo);
						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo);
						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo);
						printf("\n");*/
					}

					iTrack = tracks.erase(iTrack);
					continue;
				}
				++iTrack;
			}

			ReleDescMat(hogMat);
			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);

			if (init_counter != trackInfo.gap)//trackinfo.gap use to downsample the feature to track
				continue;

			// detect new feature points every gap frames
			std::vector<Point2f> points(0);
			for (std::list<Track>::iterator iTrack = tracks.begin();
					iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);

			DenseSample(grey_pyr[iScale], points, quality, min_distance);
			// save the new feature points
			for (int i = 0; i < points.size(); i++)
				tracks.push_back(
						Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		init_counter = 0;
		grey.copyTo(prev_grey);
		for (int i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}

		prev_kpts_surf = kpts_surf;
		desc_surf.copyTo(prev_desc_surf);

		frame_num++;
	}
	end = clock(); //CLOCK
	printf("complete in %4f seconds. %d frame and %d feature in total", (double)(end-start)/CLOCKS_PER_SEC, frame_num,featureindex);

	delete [] tmpHog;
	delete [] tmpHof;
	delete [] tmpMbhx;
	delete [] tmpMbhy;
	return featureindex;
}



#endif /* GETIDTFEATURE_H_ */
