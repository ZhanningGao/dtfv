#ifndef UTILS_H
#define UTILS_H

#define TYPE float

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <ctime>
#include <set>
#include <iterator>
#include <cassert>
#include <dirent.h>
#include <random>

#define IDX2C(r,c, ROW_NUM) (c*ROW_NUM + r) //column frist
#define IDX2R(r,c, COL_NUM) (r*COL_NUM + c) //column frist

void printMat(const TYPE *data, int rows, int cols){
	for (int c = 0; c < cols; c++){
		for (int r = 0; r < rows; r++){
			printf("%0.4f\t",data[IDX2R(r,c, cols)]);
		}
		printf("\n");
	}
	printf("\n");
}

void saveMatUtils(float* arr, int rows, int column, string filename){
  FILE *fp;
  //WARNING: Use 'a' for file writing
  assert( (fp=fopen(filename.c_str(), "w")) && "File open eror!!" ); 
  for(int r = 0; r < rows; r++){
    for (int c = 0; c < column; c++){
      fprintf(fp, "%0.4f\t", arr[IDX2C(r,c,rows)]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void saveMatUtilsRow(float* arr, int rows, int column, string filename){
  FILE *fp;
  //WARNING: Use 'a' for file writing
  assert( (fp=fopen(filename.c_str(), "w")) && "File open eror!!" ); 
  for(int r = 0; r < rows; r++){
    for (int c= 0; c < column; c++){
  
      fprintf(fp, "%0.7f\t", arr[IDX2R(r,c,column)]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

#endif