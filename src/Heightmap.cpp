#include "qrw/Heightmap.hpp"

Heightmap::Heightmap()
    : result_(VectorN::Zero(3))
    , fitSize_(0.6)
    , nFit_(10)
    , A_(MatrixN::Ones(nFit_ * nFit_, 3))
    , b_(VectorN::Zero(nFit_ * nFit_))
{
    // empty
}

void Heightmap::initialize(const std::string& file_name)
{
    // Open the binary file
    std::ifstream iF(file_name, std::ios::in | std::ios::out | std::ios::binary);
    if (!iF)
    {
        throw std::runtime_error("Error while opening heighmap binary file");
    }

    // Extract header from binary file
    iF.read(reinterpret_cast<char*>(&header_), sizeof header_);

    // Resize matrix and vector according to header
    z_ = MatrixN::Zero(header_.size_x, header_.size_y);
    x_ = VectorN::LinSpaced(header_.size_x, header_.x_init, header_.x_end);
    y_ = VectorN::LinSpaced(header_.size_y, header_.y_init, header_.y_end);

    dx_ = std::abs((header_.x_init - header_.x_end) / (header_.size_x - 1));
    dy_ = std::abs((header_.y_init - header_.y_end) / (header_.size_y - 1));

    int i = 0;
    int j = 0;
    double read;
    // Read the file and extract heightmap matrix
    while (i < header_.size_x && !iF.eof())
    {
        j = 0;
        while (j < header_.size_y && !iF.eof())
        {
            iF.read(reinterpret_cast<char*>(&read), sizeof read);
            z_(i, j) = read;
            j++;
        }
        i++;
    }
}

int Heightmap::xIndex(double x)
{
    if (x < header_.x_init || x > header_.x_end)
    {
        return -10;
    }
    else
    {
        return (int)std::round((x - header_.x_init) / dx_);
    }
}

int Heightmap::yIndex(double y)
{
    if (y < header_.y_init || y > header_.y_end)
    {
        return -10;
    }
    else
    {
        return (int)std::round((y - header_.y_init) / dy_);
    }
}

double Heightmap::getHeight(double x, double y)
{
    int index_x = xIndex(x);
    int index_y = yIndex(y);
    if (index_x == -10 || index_y == -10)
    {
        return 0.0;
    }
    else
    {
        return z_(index_x, index_y);
    }
}

Vector3 Heightmap::computeMeanSurface(double x, double y)
{
    VectorN x_surface = VectorN::LinSpaced(nFit_, x - fitSize_ / 2, x + fitSize_ / 2);
    VectorN y_surface = VectorN::LinSpaced(nFit_, y - fitSize_ / 2, y + fitSize_ / 2);

    int i_pb = 0;
    for (int i = 0; i < nFit_; i++)
    {
        for (int j = 0; j < nFit_; j++)
        {
            A_.block(i_pb, 0, 1, 2) << x_surface[i], y_surface[j];
            b_.block(i_pb, 0, 1, 1) << getHeight(x_surface[i], y_surface[j]);
            i_pb += 1;
        }
    }

    qp.reset(3, 0, 0);
    P_ = A_.transpose() * A_;
    q_ = -A_.transpose() * b_;
    status = qp.solve_quadprog(P_, q_, C_, d_, G_, h_, result_);

    return result_;
}
