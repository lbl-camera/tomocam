#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <vector_types.h>

#include "pyGnufft.h"
#include <numpy/arrayobject.h>


void generate_points(float2 *point_pos, float *point_value, int npoints,
                     uint2 grid_size) {
  for (int i = 0; i < npoints; i++) {
    point_pos[i].x = rand() / (RAND_MAX + 1.0f) * grid_size.x;
    point_pos[i].y = rand() / (RAND_MAX + 1.0f) * grid_size.y;
    point_value[i] = rand() / (RAND_MAX + 1.0f);
  }
}

void generate_polar_points(float2 *point_pos, float *point_value,
                           int points_per_line, int nlines, uint2 grid_size) {
  int idx = 0;
  for (int j = 0; j < nlines; j++) {
    float alpha = ((float)j * M_PI) / nlines;
    for (int i = 0; i < points_per_line; i++) {
      point_pos[idx].x = points_per_line / 2 +
                         (points_per_line / 2) * cosf(alpha) *
                             (i - points_per_line / 2) / (points_per_line / 2);
      point_pos[idx].y = points_per_line / 2 +
                         (points_per_line / 2) * sinf(alpha) *
                             (i - points_per_line / 2) / (points_per_line / 2);
      point_value[idx] = rand() / (RAND_MAX + 1.0f);
      idx++;
    }
  }
}

void generate_polar_points(float2 *point_pos, float2 *point_value,
                           int points_per_line, int nlines, uint2 grid_size) {
  int idx = 0;
  for (int j = 0; j < nlines; j++) {
    float alpha = ((float)j * M_PI) / nlines;
    for (int i = 0; i < points_per_line; i++) {
      point_pos[idx].x = points_per_line / 2 +
                         (points_per_line / 2) * cosf(alpha) *
                             (i - points_per_line / 2) / (points_per_line / 2);
      point_pos[idx].y = points_per_line / 2 +
                         (points_per_line / 2) * sinf(alpha) *
                             (i - points_per_line / 2) / (points_per_line / 2);
      point_value[idx].x = rand() / (RAND_MAX + 1.0f);
      point_value[idx].y = rand() / (RAND_MAX + 1.0f);
      idx++;
    }
  }
}

void divide_bin(float2 *points, int npoints, uint2 grid_size,
                std::vector<int> &points_per_bin,
                std::vector<std::vector<int> > &binned_points,
                std::vector<int> &bin_location,
                std::vector<uint2> &bin_dimensions, int kernel_radius,
                int bin_to_divide) {
  /* take the given bin and divide it into 4 bins */

  uint2 new_bin_dimensions[4];
  int new_bin_location[4];
  /* one of the bins might be slightly larger than the others if the division is
   * not even */
  int idx = 0;
  for (int y = 0; y < 2; y++) {
    for (int x = 0; x < 2; x++) {
      new_bin_dimensions[idx].x =
          (bin_dimensions[bin_to_divide].x + (1 - x)) / 2;
      new_bin_dimensions[idx].y =
          (bin_dimensions[bin_to_divide].y + (1 - y)) / 2;
      new_bin_location[idx] = bin_location[bin_to_divide] +
                              x * new_bin_dimensions[0].x +
                              y * new_bin_dimensions[0].y * grid_size.x;
      idx++;
    }
  }

  for (int i = 0; i < 4; i++) {
    int2 corner;
    std::vector<int> new_bin_binned_points;
    corner.x = new_bin_location[i] % grid_size.x;
    corner.y = new_bin_location[i] / grid_size.x;
    int new_bin_points_per_bin = 0;
    for (int j = 0; j < points_per_bin[bin_to_divide]; j++) {
      /* check if point j belongs to bin i*/
      if ((points[binned_points[bin_to_divide][j]].x - corner.x) >
              -kernel_radius &&
          (points[binned_points[bin_to_divide][j]].x -
           (corner.x + new_bin_dimensions[i].x)) <= kernel_radius &&
          (points[binned_points[bin_to_divide][j]].y - corner.y) >
              -kernel_radius &&
          (points[binned_points[bin_to_divide][j]].y -
           (corner.y + new_bin_dimensions[i].y)) <= kernel_radius) {
        /* the point is part of the bin */
        new_bin_binned_points.push_back(binned_points[bin_to_divide][j]);
        new_bin_points_per_bin++;
      }
    }
    binned_points.push_back(new_bin_binned_points);
    bin_location.push_back(new_bin_location[i]);
    points_per_bin.push_back(new_bin_points_per_bin);
    bin_dimensions.push_back(new_bin_dimensions[i]);
  }
  /* remove divided bin */
  binned_points.erase(binned_points.begin() + bin_to_divide);
  bin_location.erase(bin_location.begin() + bin_to_divide);
  points_per_bin.erase(points_per_bin.begin() + bin_to_divide);
  bin_dimensions.erase(bin_dimensions.begin() + bin_to_divide);
}

int find_most_populated_bin(std::vector<int> &points_per_bin) {
  int max = 0;
  int max_idx = 0;
  for (int i = 0; i < points_per_bin.size(); i++) {
    if (points_per_bin[i] > max) {
      max = points_per_bin[i];
      max_idx = i;
    }
  }
  return max_idx;
}

int find_most_computationaly_intensive_bin(std::vector<int> &points_per_bin,
                                           std::vector<uint2> &bin_dimensions) {
  int max = 0;
  int max_idx = 0;
  for (int i = 0; i < points_per_bin.size(); i++) {
    if ((points_per_bin[i] + 100) * bin_dimensions[i].x * bin_dimensions[i].y >
        max) {
      max =
          (points_per_bin[i] + 100) * bin_dimensions[i].x * bin_dimensions[i].y;
      max_idx = i;
    }
  }
  return max_idx;
}

class Bin {
public:
  Bin(std::vector<int> points, uint2 dimensions, int location) {
    m_dimensions = dimensions;
    m_location = location;
    m_points = points;
  }
  uint2 m_dimensions;
  std::vector<int> m_points;
  int m_location;
  bool operator<(const Bin &b) const { return this->m_location < b.m_location; }
};

std::vector<Bin> adaptative_bin_points(float2 *points, int npoints,
                                       uint2 grid_size, int total_bins,
                                       int kernel_radius) {

  std::vector<Bin> bins;
  std::vector<int> points_per_bin;
  points_per_bin.push_back(npoints);
  std::vector<uint2> bin_dimensions;
  bin_dimensions.push_back(grid_size);
  std::vector<std::vector<int> > binned_points;
  std::vector<int> point_list;
  for (int i = 0; i < npoints; i++) {
    point_list.push_back(i);
  }
  binned_points.push_back(point_list);
  std::vector<int> bin_location;
  bin_location.push_back(0);

  while (points_per_bin.size() < total_bins) {
    int bin_to_divide =
        find_most_computationaly_intensive_bin(points_per_bin, bin_dimensions);
    if (bin_dimensions[bin_to_divide].x > 1 ||
        bin_dimensions[bin_to_divide].y > 1) {
      divide_bin(points, npoints, grid_size, points_per_bin, binned_points,
                 bin_location, bin_dimensions, kernel_radius, bin_to_divide);
    } else {
      /* cannot divide the most populated tile anymore */
      break;
    }
    if (points_per_bin.size() >= total_bins) {
      /* remove bins with 0 points or 0 area */
      for (int i = 0; i < points_per_bin.size(); i++) {
        if (points_per_bin[i] == 0 || bin_dimensions[i].x == 0 ||
            bin_dimensions[i].y == 0) {
          binned_points.erase(binned_points.begin() + i);
          bin_location.erase(bin_location.begin() + i);
          points_per_bin.erase(points_per_bin.begin() + i);
          bin_dimensions.erase(bin_dimensions.begin() + i);
          i--;
        }
      }
    }
  }
  int nbins = points_per_bin.size();
  int smallest_area = bin_dimensions[0].x * bin_dimensions[0].y;
  int largest_area = bin_dimensions[0].x * bin_dimensions[0].y;
  int most_intense = 0;
  int most_intense_idx = 0;
  int bin_points_size = 0;
  for (int i = 0; i < nbins; i++) {
    //    printf("points = %d area = %d\n",points_per_bin[i],
    // bin_dimensions[i].x*bin_dimensions[i].y);
    if (bin_dimensions[i].x * bin_dimensions[i].y < smallest_area &&
        bin_dimensions[i].x * bin_dimensions[i].y > 0) {
      smallest_area = bin_dimensions[i].x * bin_dimensions[i].y;
    }
    if (bin_dimensions[i].x * bin_dimensions[i].y > largest_area) {
      largest_area = bin_dimensions[i].x * bin_dimensions[i].y;
    }
    if (bin_dimensions[i].x * bin_dimensions[i].y * points_per_bin[i] >
        most_intense) {
      most_intense =
          bin_dimensions[i].x * bin_dimensions[i].y * points_per_bin[i];
      most_intense_idx = i;
    }
    bin_points_size += 256 * (points_per_bin[i] + 255) / 256;
  }
  printf("max points per bin %d\n",
         points_per_bin[find_most_populated_bin(points_per_bin)]);
  printf("Largest area = %d\n", largest_area);
  printf("Smallest area = %d\n", smallest_area);
  printf("Most intense area = %d\n", (bin_dimensions[most_intense_idx].x *
                                      bin_dimensions[most_intense_idx].y));
  printf("Most intense points = %d\n", points_per_bin[most_intense_idx]);
  printf("Most intense score = %d\n", most_intense);
  printf("Divided in %d bins\n", nbins);

  /* Sort the bins */
  for (int i = 0; i < nbins; i++) {
    bins.push_back(Bin(binned_points[i], bin_dimensions[i], bin_location[i]));
  }
  /* Sorting doesn't help in the tesla c1060 */
  //  sort(bins.begin(),bins.end());
  for (int i = 0; i < nbins; i++) {
    binned_points[i] = bins[i].m_points;
    bin_dimensions[i] = bins[i].m_dimensions;
    bin_location[i] = bins[i].m_location;
    points_per_bin[i] = bins[i].m_points.size();
  }

  return bins;
}

PyObject * cPolarBin(PyObject *self, PyObject *args){
    PyObject *in0, *in1, *in2;
    int approx_n_bins;
    float kernel_radius;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "OOOII", &in0, &in1, &in2, &approx_n_bins, &kernel_radius))
        return NULL;

    if ((in0 == NULL) || (in1 == NULL) || (in2 == NULL))
        return NULL;

    //import_array();
    PyObject * pySx   = PyArray_FROM_OT(in0, NPY_FLOAT);
    PyObject * pySy   = PyArray_FROM_OT(in1, NPY_FLOAT);
    PyObject * pyDims = PyArray_FROM_OT(in2, NPY_INT);

    float * samples_x = (float *) PyArray_DATA(pySx);
    float * samples_y = (float *) PyArray_DATA(pySy);
    int   * grid_dim  = (int   *) PyArray_DATA(pyDims);
    size_t npoints = (size_t) PyArray_SIZE(pySx);

    float2 * point_pos = new float2[npoints];
    for (int i = 0; i < npoints; i++){
        point_pos[i].x = samples_x[i];
       point_pos[i].y = samples_y[i];
    } 

    uint2 grid_size = { grid_dim[0], grid_dim[1] };
    std::vector<Bin> bins = adaptative_bin_points(point_pos, npoints, grid_size,
            approx_n_bins, kernel_radius);

    /* align to 32 bits */
    int bin_points_size = 0;
    for (int i = 0; i < bins.size(); i++) {
        bin_points_size += 32 * (bins[i].m_points.size() + 31) / 32;
    }
    
    /* construct output containers */
    npy_intp dims[2];
    dims[0] = (npy_intp) bins.size(); dims[0] = 0;
    PyObject * out0 = PyArray_SimpleNew(1, dims, NPY_INT);
    PyObject * out1 = PyArray_SimpleNew(1, dims, NPY_INT);
    PyObject * out2 = PyArray_SimpleNew(1, dims, NPY_INT);
    PyObject * out4 = PyArray_SimpleNew(1, dims, NPY_INT);
    PyObject * out5 = PyArray_SimpleNew(1, dims, NPY_INT);
    dims[0] = bin_points_size;
    PyObject * out3 = PyArray_SimpleNew(1, dims, NPY_INT);
    PyObject * out6 = PyArray_SimpleNew(1, dims, NPY_FLOAT);
    PyObject * out7 = PyArray_SimpleNew(1, dims, NPY_FLOAT);

   
    /* map data to containers */
    int * points_per_bin = (int *) PyArray_DATA(out0);
    int * bin_dimensions_x = (int *) PyArray_DATA(out1);
    int * bin_dimensions_y = (int *) PyArray_DATA(out2);
    int * binned_points = (int *) PyArray_DATA(out3);
    int * bin_offset = (int *) PyArray_DATA(out4);
    int * bin_location = (int *) PyArray_DATA(out5);
    float * binned_points_x = (float *) PyArray_DATA(out6);
    float * binned_points_y = (float *) PyArray_DATA(out7);

    int idx = 0;
    for (int i = 0; i < bins.size(); i++){
        points_per_bin[i] = bins[i].m_points.size();
        bin_dimensions_x[i] = bins[i].m_dimensions.x;
        bin_dimensions_y[i] = bins[i].m_dimensions.y;
        bin_offset[i] = idx;
        bin_location[i] = bins[i].m_location;
        for (int j = 0; j < points_per_bin[i]; j++) {
            binned_points[idx + j] = bins[i].m_points[j];
            binned_points_x[idx + j] = samples_x[bins[i].m_points[j]];
            binned_points_y[idx + j] = samples_y[bins[i].m_points[j]];
        }
        idx += points_per_bin[i];
    }

    /* pack arrays into a python tuple */
    PyObject * lhs = PyTuple_New(8);
    PyTuple_SetItem(lhs, 0, out0);
    PyTuple_SetItem(lhs, 1, out1);
    PyTuple_SetItem(lhs, 2, out2);
    PyTuple_SetItem(lhs, 3, out3);
    PyTuple_SetItem(lhs, 4, out4);
    PyTuple_SetItem(lhs, 5, out5);
    PyTuple_SetItem(lhs, 6, out6);
    PyTuple_SetItem(lhs, 7, out7);

    /* clean up */
    delete [] point_pos;
    Py_XDECREF(pySx);
    Py_XDECREF(pySy);
    Py_XDECREF(pyDims);

    return lhs;
}

