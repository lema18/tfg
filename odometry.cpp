#include "odometry.h"
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

int main(int argc,char **argv)
{
    int paso_foto;
	int id=0;
    int visitado=1;
    double scale=1;
	sscanf(argv[1],"%d",&paso_foto);
	Mat distcoef =(Mat_<float>(1,5)<<0.2624 ,-0.9531, -0.0054 ,0.0026, 1.1633);
	/*parámetros intrínsecos de la cámara*/
	Mat distcoefd;
	distcoef.convertTo(distcoefd,CV_64F);
	Mat intrinseca=(Mat_<float>(3,3)<<517.3, 0., 318.6, 0., 516.5, 255.3, 0., 0., 1.);
	Mat intrinsecad;
	intrinseca.convertTo(intrinsecad,CV_64F);
	/*inicialización de vectores,matrices y variables a utilizar durante el código*/
	Mat prev_pose=Mat::eye(4,4,CV_64F);/*transformación afín correspondiente a la matriz extrínseca de la captura i+1*/
	Mat current_pose(4,4,CV_64F);
    Mat prev_proyec=intrinsecad*Mat::eye(3,4,CV_64F);
    Mat curr_proyec(3,4,CV_64F);
    Mat point3d_homo;
    vector<p3dant> nube;
	Eigen::Affine3f cam_pos;
	Eigen::Affine3f cam_posinic;
	Eigen::Matrix4f eig_matinic;
	//vector<DMatch> matches;/*vector para almacenar el matching realizado entre las features de dos imágenes consecutivas*/
	vector<Point2f> puntos1,puntos2;/*para filtros intermedios*/
	vector<Point2d> triangulation_points1, triangulation_points2;
	vector<Point3d> prev_points3d,curr_points3d;				
	// preparamos un viewer
	pcl::visualization::PCLVisualizer viewer("Viewer");
  	viewer.setBackgroundColor (255, 255, 255);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	//añadimos un sistema de referencia(la posición inicial de la cámara)
	prev_pose.convertTo(prev_pose, CV_32F);
	eig_matinic(0,0) = prev_pose.at<float>(0,0);eig_matinic(0,1) = prev_pose.at<float>(0,1);eig_matinic(0,2) = prev_pose.at<float>(0,2);eig_matinic(0,3) = prev_pose.at<float>(0,3);
	eig_matinic(1,0) = prev_pose.at<float>(1,0);eig_matinic(1,1) = prev_pose.at<float>(1,1);eig_matinic(1,2) = prev_pose.at<float>(1,2);eig_matinic(1,3) = prev_pose.at<float>(1,3);
	eig_matinic(2,0) = prev_pose.at<float>(2,0);eig_matinic(2,1) = prev_pose.at<float>(2,1);eig_matinic(2,2) = prev_pose.at<float>(2,2);eig_matinic(2,3) = prev_pose.at<float>(2,3);
	eig_matinic(3,0) = prev_pose.at<float>(3,0);eig_matinic(3,1) = prev_pose.at<float>(3,1);eig_matinic(3,2) = prev_pose.at<float>(3,2);eig_matinic(3,3) = prev_pose.at<float>(3,3);
	prev_pose.convertTo(prev_pose,CV_64F);
	cam_posinic = eig_matinic;
	viewer.addCoordinateSystem(1.0, cam_posinic,"pos_inic");
    //viewer.spin();
	int l=0;/*para el nombre de cada imagen*/
	/*lectura de la primera imagen*/	
	stringstream ss; /*convertimos el entero l a string para poder establecerlo como nombre de la captura*/
	ss<<l;
	string nombre="left_";
	nombre+=ss.str();
	nombre+=".png";
	Mat foto1_c=imread(nombre,CV_LOAD_IMAGE_COLOR);/*leemos la imagen de memoria en escala de grises*/
	if(!foto1_c.data)
	{
		cout<<"error en la lectura de imágenes"<<endl;
		return -1;
	}
	l+=paso_foto;
	stringstream ss2;
	ss2<<l;
	string nombre2="left_";
	nombre2+=ss2.str();
	nombre2+=".png";
	Mat foto2_c=imread(nombre2,CV_LOAD_IMAGE_COLOR);/*leemos la imagen de memoria en escala de grises*/
	if(!foto2_c.data)
	{
		cout<<"error en la lectura de imágenes"<<endl;
		return -1;
	}
	Mat foto1;
	Mat foto2;
	cvtColor(foto1_c,foto1,COLOR_BGR2GRAY);
	cvtColor(foto2_c,foto2,COLOR_BGR2GRAY);
    Mat aux2;/*para almacenar la segunda imagen sin distorsión de lente*/
	Mat aux1;/*para almacenar la primera imagen sin distorsión de lente*/
	undistort(foto1,aux1,intrinsecad,distcoefd);
	undistort(foto2,aux2,intrinsecad,distcoefd);
    Mat descript1,descript2;
    vector<KeyPoint> features1,features2;
    feature_detection_and_description(aux1,features1,descript1);
    feature_detection_and_description(aux2,features2,descript2);
    vector<int> prev_index,curr_index;
    vector<vector<DMatch>> matches;
    feature_matching(descript1,descript2,features1,features2,matches,puntos1,puntos2,prev_index,curr_index);
    Mat E;/*para almacenar la matriz esencial*/
	Mat R;/*para almacenar la rotación relativa entre dos capturas*/
	Mat t;/*para almacenar la traslación relativa entre dos capturas*/
	Mat mask;/*para inliers(1) y outliers(0)*/
    estimate_rotation_translation(E,R,t,mask,puntos1,puntos2,intrinsecad);
    vector<int> prev_used_features,curr_used_features;
    delete_outliers(puntos1,puntos2,mask,prev_index,curr_index,triangulation_points1,triangulation_points2,prev_used_features,curr_used_features);
    update_pose_and_calculate_3d_points(R,t,intrinsecad,scale,triangulation_points1,triangulation_points2,prev_proyec,curr_proyec,prev_pose,current_pose,cam_pos,point3d_homo);
    prev_pose=current_pose;
    prev_proyec=curr_proyec;
    update_3d_points(point3d_homo,nube,curr_used_features,prev_used_features,1,prev_points3d,curr_points3d,visitado);
    viewer.addCoordinateSystem(1.0, cam_pos,nombre2);
  	viewer.initCameraParameters ();
    //update_cloud(nube,cloud,viewer,id);
	triangulation_points1.clear();
	triangulation_points2.clear();
	puntos1.clear();
	puntos2.clear();
	descript1.release();
	features1.clear();
	Mat prevImage=aux2;
	Mat currImage;
	vector<KeyPoint> prev_features;
	prev_features=features2;
	features2.clear();
	Mat prev_descript;
	prev_descript=descript2;
	descript2.release();
	prev_index.clear();
	curr_index.clear();
	prev_used_features.clear();
	curr_used_features.clear();
	point3d_homo.release();
	int fin=0;
	while(!fin)
	{	
		Mat point3d_homo;
		scale=1;
		l+=paso_foto;
		stringstream ss3;
		ss3<<l;
		string nombre3="left_";
		nombre3+=ss3.str();
		nombre3+=".png";
		Mat foto_color=imread(nombre3,CV_LOAD_IMAGE_COLOR);/*leemos la imagen de memoria en escala de grises*/
		Mat foto_bn;
		if(!foto_color.data)
		{
			cout<<"no hay más imágenes \n"<<endl;
			fin=1;
			continue;
		}
		cvtColor(foto_color,foto_bn,COLOR_BGR2GRAY);
		undistort(foto_bn,currImage,intrinsecad,distcoefd);
		Mat curr_descript;
		vector<KeyPoint> curr_features;
		vector<vector<DMatch>> new_matches;
		feature_detection_and_description(currImage,curr_features,curr_descript);
		feature_matching(prev_descript,curr_descript,prev_features,curr_features,new_matches,puntos1,puntos2,prev_index,curr_index);
		estimate_rotation_translation(E,R,t,mask,puntos1,puntos2,intrinsecad);
		delete_outliers(puntos1,puntos2,mask,prev_index,curr_index,triangulation_points1,triangulation_points2,prev_used_features,curr_used_features);
		update_pose_and_calculate_3d_points(R,t,intrinsecad,scale,triangulation_points1,triangulation_points2,prev_proyec,curr_proyec,prev_pose,current_pose,cam_pos,point3d_homo);
		update_3d_points(point3d_homo,nube,curr_used_features,prev_used_features,0,prev_points3d,curr_points3d,visitado);
		calculate_scale(scale,prev_points3d,curr_points3d);
		update_pose_and_calculate_3d_points(R,t,intrinsecad,scale,triangulation_points1,triangulation_points2,prev_proyec,curr_proyec,prev_pose,current_pose,cam_pos,point3d_homo);
		good_3d_point(point3d_homo,curr_used_features,prev_used_features,nube,visitado);
		viewer.addCoordinateSystem(1.0, cam_pos,nombre3);
  		viewer.initCameraParameters ();
		prev_features=curr_features;
		prev_descript=curr_descript;
		triangulation_points1.clear();
		triangulation_points2.clear();
		puntos1.clear();
		puntos2.clear();
		prev_index.clear();
		curr_index.clear();
		prev_used_features.clear();
		curr_used_features.clear();
		prev_proyec=curr_proyec;
		prev_pose=current_pose;
		//prev_points3d.clear();
		//curr_points3d.clear();
		point3d_homo.release();
	}
	//update_cloud(nube,cloud,viewer,id);
	while (!viewer.wasStopped ())
	{
		viewer.spin();
	}
    return 0;
}