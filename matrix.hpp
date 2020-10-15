#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>


namespace sp
{
    template<typename T>
    class Matrix2D
    {
    public:
        uint32_t _cols;
        uint32_t _rows;
        std::vector<T> _vals;


    public:
        Matrix2D(uint32_t cols, uint32_t rows)
            : _cols(cols),
            _rows(rows),
            _vals({})
        {
            _vals.resize(rows * cols, T());
        }

        Matrix2D()
             : _cols(0),
            _rows(0),
            _vals({})
        {
        }

        T& at(uint32_t col, uint32_t row)
        {
            return _vals[row * _cols + col];
        }

        bool isSquare()
        {
            return _rows == _cols;
        }


        Matrix2D negetive()
        {
            Matrix2D output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = -at(x, y);
                }
            return output;
        }


        Matrix2D multiply(Matrix2D& target)
        {
            assert(_cols == target._rows);
            Matrix2D output(target._cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    T result = T();
                    for (uint32_t k = 0; k < _cols; k++)
                        result += at(k, y) * target.at(x, k);
                    output.at(x, y) = result;
                }
            return output;
        }

        Matrix2D multiplyElements(Matrix2D& target)
        {
            assert(_rows == target._rows && _cols == target._cols);
            Matrix2D output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) * target.at(x, y);
                }
            return output;
        }


        Matrix2D add(Matrix2D& target)
        {
            assert(_rows == target._rows && _cols == target._cols);
            Matrix2D output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) + target.at(x, y);
                }
            return output;
        }
        Matrix2D applyFunction(std::function<T(const T&)> func)
        {
            Matrix2D output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = func(at(x, y));
                }
            return output;
        }

        Matrix2D multiplyScaler(float s)
        {
            Matrix2D output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) * s;
                }
            return output;

        }

        Matrix2D addScaler(float s)
        {
            Matrix2D output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) + s;
                }
            return output;

        }
        Matrix2D transpose()
        {
            Matrix2D output(_rows, _cols);
            for (uint32_t y = 0; y < _rows; y++)
                for (uint32_t x = 0; x < _cols; x++)
                {
                    output.at(y, x) = at(x, y);
                }
            return output;
        }

        Matrix2D cofactor(uint32_t col, uint32_t row)
        {
            Matrix2D output(_cols - 1, _rows - 1);
            uint32_t i = 0;
            for (uint32_t y = 0; y < _rows; y++)
                for (uint32_t x = 0; x < _cols; x++)
                {
                    if (x == col || y == row) continue;
                    output._vals[i++] = at(x, y);
                }

            return output;
        }

        T determinant()
        {
            assert(_rows == _cols);
            T output = T();
            if (_rows == 1)
            {
                return _vals[0];
            }
            else
            {
                int32_t sign = 1;
                for (uint32_t x = 0; x < _cols; x++)
                {
                    output += sign * at(x, 0) * cofactor(x, 0).determinant();
                    sign *= -1;
                }
            }

            return output;
        }

        Matrix2D adjoint()
        {
            assert(_rows == _cols);
            Matrix2D output(_cols, _rows);
            int32_t sign = 1;
            for (uint32_t y = 0; y < _rows; y++)
                for (uint32_t x = 0; x < _cols; x++)
                {
                    output.at(x, y) = sign * cofactor(x, y).determinant();
                    sign *= -1;
                }
            output = output.transpose();

            return output;
        }

        Matrix2D inverse()
        {
            Matrix2D adj = adjoint();
            T factor = determinant();
            for (uint32_t y = 0; y < adj._cols; y++)
                for (uint32_t x = 0; x < adj._rows; x++)
                {
                    adj.at(x, y) = adj.at(x, y) / factor;
                }
            return adj;
        }



    }; // class Matrix2D

    template<typename T>
    void LogMatrix2D(Matrix2D<T>& mat)
    {
        for (uint32_t y = 0; y < mat._rows; y++)
        {
            for (uint32_t x = 0; x < mat._cols; x++)
                std::cout << std::setw(10) << mat.at(x, y) << " ";
            std::cout << std::endl;
        }
    }

}
