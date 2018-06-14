#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include<string>
#include<iostream>
#include<cstdlib>

#include "armadillo"
#include "dynamic_het_types.h"

using namespace std;
using namespace arma;

template<typename T>
Mat<T> csv_to_mat(const char *file_name, const char &sep = ','){
  ifstream file_reader;
  file_reader.open(file_name);

  string file_line_buf;

  getline(file_reader,file_line_buf);

  // count how many elements in this line..
  n_n n_elem = 1;
  int comma_occur = 0;
  while((string::npos) != (comma_occur = file_line_buf.find(",",comma_occur+1))) n_elem++;

  n_n n_lines = 1;
  while(getline(file_reader,file_line_buf)) n_lines++;
  file_reader.close();

  file_reader.open(file_name);
  getline(file_reader,file_line_buf);
  Mat<T> file_in_mat(n_lines,n_elem);


  // i don't know how but i made it work..
  n_n n_start;
  int n_end;
  n_n n_cnt;
  n_n line_cnt = 0;
  do{
    // main loop..
    n_start = 0;
    n_cnt = 0;
    n_end = 0;
    while(n_end != file_line_buf.size()){
      n_end = file_line_buf.find(sep, n_start+1);
      if(n_end == string::npos) n_end = file_line_buf.size();
      file_in_mat(line_cnt,n_cnt) = stoi(file_line_buf.substr(n_start, n_end - n_start));
      n_cnt++;
      n_start = n_end + 1;
    }
    line_cnt++;
    getline(file_reader,file_line_buf);
  }while(line_cnt < n_lines);

  file_reader.close();

  return file_in_mat;
}

template<typename T>
void mat_to_csv(const char *file_name, Mat<T> file_in_mat){
  ofstream file_writer;
  // warning: will overwrite data
  file_writer.open(file_name, std::ofstream::out | std::ofstream::trunc);

  n_n mat_nrow = file_in_mat.n_rows;
  n_n mat_ncol = file_in_mat.n_cols;

  for(n_n n_row = 0; n_row < mat_nrow; n_row++){
    for(n_n n_col = 0; n_col < mat_ncol; n_col++){
      file_writer << double(file_in_mat(n_row, n_col));
      if(n_col != mat_ncol - 1) file_writer << ',';
    }
    if(n_row != mat_nrow - 1) file_writer << '\n';
  }
  file_writer.close();
}

#endif
