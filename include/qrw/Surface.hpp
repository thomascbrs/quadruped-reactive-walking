///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Surface class
///
/// \details Surface data structure 
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef SURFACE_H_INCLUDED
#define SURFACE_H_INCLUDED


#include "qrw/Types.h"
#include <Eigen/Core>
#include <Eigen/Dense>



class Surface
{
public:
    // Constructor
    Surface() ; 
    Surface(const  Eigen::MatrixXd &A_in , const Eigen::VectorXd &b_in, const Eigen::MatrixXd &vertices_in);

    bool operator==(const Surface& other) const {return false;}
    bool operator!=(const Surface& other) const {return true;}

    // Destructor
    ~Surface() {} 

    // Usefull for python binding
    Eigen::MatrixXd get_A() ;
    void set_A(const  Eigen::MatrixXd &A_i) ;

    Eigen::VectorXd get_b() ;
    void set_b(const  Eigen::VectorXd &b_i) ;

    Eigen::MatrixXd get_vertices() ;
    void set_vertices(const  Eigen::MatrixXd &vertices_i) ;

    double getHeight(Vector3 point);


    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic > A;
    Eigen::Matrix<double, Eigen::Dynamic, 1 > b;
    Eigen::Matrix<double, Eigen::Dynamic, 3 > vertices;
};

#endif  // SURFACE_H_INCLUDED
