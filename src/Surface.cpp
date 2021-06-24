#include "qrw/Surface.hpp"

Surface::Surface(){}

Surface::Surface(const  Eigen::MatrixXd &A_in , const Eigen::VectorXd &b_in, const Eigen::MatrixXd &vertices_in){
    
    A = A_in ;
    b = b_in ;
    vertices = vertices_in ;

}

void Surface::set_A(const  Eigen::MatrixXd &A_i){
    A = A_i ;
}

Eigen::MatrixXd Surface::get_A() {
    return A; 
}

Eigen::VectorXd Surface::get_b() {
    return b; 
}

void Surface::set_b(const  Eigen::VectorXd &b_i){
    b = b_i ;
}

Eigen::MatrixXd Surface::get_vertices() {
    return vertices; 
}

void Surface::set_vertices(const  Eigen::MatrixXd &vertices_i){
    vertices = vertices_i ;
}

// For a given X,Y point that belongs to the surface, return the height
//     d/c -a/c*x -b/c*y
//     Args : 
//     - point (array x2), works with arrayx3
double Surface::getHeight(Vector3 point){
    int id = A.rows() -1 ;
    return abs( b(id) - point(0)*A(id,0) / A(id,2)  - point(1)*A(id,1) / A(id,2)  ) ;
}







