#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <sstream>
#include <unistd.h>
#include <ctime>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#define MIN_NUM_FEAT 2000

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

struct p3dant
{
    Point3d punto;
    int curr_index;
    int prev_index;
    int visitado=1;
};
void feature_detection_and_description(Mat& img,vector<KeyPoint>& features,Mat& descriptors)
{
    Ptr<SURF> pt=SURF::create(300);
    pt->detect(img,features);
    pt->compute(img,features,descriptors);
    
}

void feature_matching(Mat& descript1,Mat& descript2,vector<KeyPoint> features1,vector<KeyPoint> features2,vector<vector<DMatch>>& matches,vector<Point2f>& source,vector<Point2f>& destination,vector<int>& prev_index,vector<int>& curr_index)
{
    Ptr<DescriptorMatcher> pt=DescriptorMatcher::create("BruteForce");
    pt->knnMatch(descript1,descript2,matches,2);
    for (int i=0;i<matches.size();i++)
    {
                // TODO: Was 0.7. Too low?
                if(matches[i][0].distance < 0.8*matches[i][1].distance)
                {
                    source.push_back(features1[matches[i][0].queryIdx].pt);
                    destination.push_back(features2[matches[i][0].queryIdx].pt);

                    prev_index.push_back(matches[i][0].queryIdx);
                    curr_index.push_back(matches[i][0].trainIdx);
                }
    }
}
void estimate_rotation_translation(Mat& E,Mat& R,Mat& t,Mat& mask,vector<Point2f>puntos1,vector<Point2f>puntos2,Mat intrinsecad)
{
  double focal=intrinsecad.at<double>(0,0);
	Point2d pp=Point2d(intrinsecad.at<double>(0,2),intrinsecad.at<double>(1,2));
  E=findEssentialMat(puntos2,
					puntos1,
					focal,
					pp,
					8,
					0.999,
					1,
					mask);
	recoverPose(E,
				puntos2,
				puntos1,
				R,
				t,
			    focal,
				pp,
			    mask);
}
void delete_outliers(vector<Point2f> outliers_1,vector<Point2f> outliers_2,Mat mask,vector<int> prev_index,vector<int> curr_index,vector<Point2d>& triang1,vector<Point2d>& triang2,vector<int>& prev_used_features,vector<int>& curr_used_features)
{
  for(int i = 0; i < mask.rows; i++) 
	{
    if(mask.at<unsigned char>(i))
		{
			triang1.push_back(Point2d((double)outliers_1[i].x,(double)outliers_1[i].y));
			triang2.push_back(Point2d((double)outliers_2[i].x,(double)outliers_2[i].y));
            prev_used_features.push_back(prev_index[i]);
            curr_used_features.push_back(curr_index[i]);
	  }
  }
}
void update_pose_and_calculate_3d_points(Mat R,Mat& t,Mat intrinsecad,double scale,vector<Point2d> triang1,vector<Point2d> triang2,Mat prev_proyec,Mat& curr_proyec,Mat prev_pose,Mat& current_pose,Eigen::Affine3f& cam_pos,Mat& point3d_homo)
{
    //first we calculate cam_pose
    Mat auxiliar=Mat::eye(4,4,CV_64F);
    t*=scale;
    Eigen::Matrix4f eig_mat;
    R.copyTo(auxiliar(Range(0, 3), Range(0, 3)));
    t.copyTo(auxiliar(Range(0, 3), Range(3, 4)));
    current_pose=prev_pose*auxiliar;
    current_pose.convertTo(current_pose, CV_32F);
    eig_mat(0,0) = current_pose.at<float>(0,0);eig_mat(0,1) = current_pose.at<float>(0,1);eig_mat(0,2) = current_pose.at<float>(0,2);eig_mat(0,3) = current_pose.at<float>(0,3);
	eig_mat(1,0) = current_pose.at<float>(1,0);eig_mat(1,1) = current_pose.at<float>(1,1);eig_mat(1,2) = current_pose.at<float>(1,2);eig_mat(1,3) = current_pose.at<float>(1,3);
	eig_mat(2,0) = current_pose.at<float>(2,0);eig_mat(2,1) = current_pose.at<float>(2,1);eig_mat(2,2) = current_pose.at<float>(2,2);eig_mat(2,3) = current_pose.at<float>(2,3);
	eig_mat(3,0) = current_pose.at<float>(3,0);eig_mat(3,1) = current_pose.at<float>(3,1);eig_mat(3,2) = current_pose.at<float>(3,2);eig_mat(3,3) = current_pose.at<float>(3,3);
	current_pose.convertTo(current_pose, CV_64F);
	cam_pos = eig_mat;
    //now we calculate 3d points by triangulation
    Mat P(3,4,CV_64F);
    Mat Rot=current_pose(Range(0, 3), Range(0, 3));
    Mat tras=current_pose(Range(0, 3), Range(3, 4));
	P(Range(0, 3), Range(0, 3)) = Rot.t();
    P(Range(0, 3), Range(3, 4)) = -Rot.t()*tras;
    P = intrinsecad*P;
    curr_proyec=P;
    triangulatePoints(prev_proyec,P,triang1,triang2,point3d_homo);
}
void update_3d_points(Mat point3d_homo,vector<p3dant>& nube,vector<int> curr_used_features,vector<int> prev_used_features,int inic,vector<Point3d>& p3d_prev,vector<Point3d>& p3d_curr,int& visitado)
{
    
    if(inic)//when we have only to add points
    {
        for(int i = 0; i < point3d_homo.cols; i++) 
        {
            Mat p3d;
            Mat _p3h = point3d_homo.col(i);
            convertPointsFromHomogeneous(_p3h.t(), p3d);
            struct p3dant mipt;
            mipt.punto.x = p3d.at<float>(0);
            mipt.punto.y = p3d.at<float>(1);
            mipt.punto.z = p3d.at<float>(2);
            mipt.curr_index=curr_used_features[i];
            mipt.prev_index=prev_used_features[i];
            nube.push_back(mipt);
        }
    }
    else //when we have to check if we have the same 3d points
    {
        vector<int> betos;
        visitado++;
       for(int i = 0; i < point3d_homo.cols; i++) 
        {   
            Mat p3d;
            int flag=0;
            Mat _p3h = point3d_homo.col(i);
            convertPointsFromHomogeneous(_p3h.t(), p3d);
            struct p3dant mipt;
            mipt.punto.x = p3d.at<float>(0);
            mipt.punto.y = p3d.at<float>(1);
            mipt.punto.z = p3d.at<float>(2);
            mipt.curr_index=curr_used_features[i];
            mipt.prev_index=prev_used_features[i];
            int only1=0;
            for(int j = nube.size() - 1; j >= 0; --j)
            {
                if(nube[j].curr_index==mipt.prev_index && only1==0)
                {
                    //we update existent point and we take their previous values to compute scale
                    p3d_prev.push_back(nube[j].punto);
                    p3d_curr.push_back(mipt.punto);
                    nube[j].punto+=mipt.punto;
                    nube[j].punto/=2;
                    betos.push_back(1);
                    nube[j].visitado=visitado;
                    //we update de index for next iteration
                    nube[j].curr_index=mipt.curr_index;
                    nube[j].prev_index=mipt.prev_index;
                    flag=1;
                    only1=1;
                }
            }
            if(!flag) betos.push_back(0);
        }
        for(int l=0;l<betos.size();l++)
        {
            if(!betos[l])
            {
                Mat p3d;
                Mat _p3h = point3d_homo.col(l);
                convertPointsFromHomogeneous(_p3h.t(), p3d);
                struct p3dant mipt;
                mipt.punto.x = p3d.at<float>(0);
                mipt.punto.y = p3d.at<float>(1);
                mipt.punto.z = p3d.at<float>(2);
                mipt.curr_index=curr_used_features[l];
                mipt.prev_index=prev_used_features[l];
                nube.push_back(mipt);
            }
        }
    }
}
void calculate_scale(double& escala,vector<Point3d>& p3d_prev,vector<Point3d>& p3d_curr)
{
    vector<double> escalas;
    //now we calculate scale 
    for(int k=1;k<p3d_curr.size();k++)
    {
        escalas.push_back(norm(p3d_prev[k-1] - p3d_prev[k]) / norm(p3d_curr[k-1] - p3d_curr[k]));
    }
    sort(escalas.begin(),escalas.end());
    size_t talla=escalas.size();
    if(talla==0)
    {
        //bad odometry
        escala=(double)1.0;
    }
    else
    {
        if(talla==1)
        {
            escala=escalas[talla-1];
        }
        else
        {
            if((talla%2) == 0)
            {
                escala=(escalas[talla / 2 - 1] + escalas[talla / 2]) / 2;
            }
            else
            {
                escala=escalas[talla/2];
            }    
        }   
    }
    p3d_prev.clear();
    p3d_curr.clear();
}
void good_3d_point(Mat point3d_homo,vector<int> curr_used_features,vector<int> prev_used_features,vector<p3dant>& nube,int visitado)
{
    for(int i = 0; i < point3d_homo.cols; i++) 
    {
        Mat p3d;
        Mat _p3h = point3d_homo.col(i);
        convertPointsFromHomogeneous(_p3h.t(), p3d);
        struct p3dant mipt;
        mipt.punto.x = p3d.at<float>(0);
        mipt.punto.y = p3d.at<float>(1);
        mipt.punto.z = p3d.at<float>(2);
        mipt.curr_index=curr_used_features[i];
        mipt.prev_index=prev_used_features[i];
        for (int j = nube.size() - 1; j >= 0; --j)
        {
            if(nube[j].visitado==visitado)
            {
                if(nube[j].curr_index==mipt.curr_index && nube[j].prev_index==mipt.prev_index)
                {
                    nube[j].punto=mipt.punto;
                }
            }
        }
            
    }  
}
void update_cloud(vector<p3dant>& puntos3d,pcl::PointCloud<pcl::PointXYZRGB>::Ptr &nube,pcl::visualization::PCLVisualizer &viewer,int &id)
{
    vector<Point3d> pts3d;
    for(int i=0;i<puntos3d.size();i++)
    {
       Point3d mipunto;
       mipunto.x=puntos3d[i].punto.x;
       mipunto.y=puntos3d[i].punto.y;
       mipunto.z=puntos3d[i].punto.z;
       pts3d.push_back(mipunto);

    }
    nube->points.resize (pts3d.size());

    for(int i = 0; i < pts3d.size(); i++) 
	{
   	pcl::PointXYZRGB &point = nube->points[i];
    point.x = pts3d[i].x;
    point.y = pts3d[i].y;
    point.z = pts3d[i].z;
    point.r = 0;
    point.g = 0;
    point.b = 255;
  }
  stringstream ss;
	ss<<id;
	string nombre=ss.str();
    viewer.addPointCloud(nube,nombre);
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,nombre);
    id+=1;
    //viewer.spin();
}
