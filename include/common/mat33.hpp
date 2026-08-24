#ifndef MAT33_HPP
#define MAT33_HPP

#include <cmath>
#include <vector>

// 三维向量
struct Vec3 {
        double x = 0.0, y = 0.0, z = 0.0;

        Vec3() = default;
        Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

        Vec3 operator+(const Vec3& o) const     { return {x + o.x, y + o.y, z + o.z}; }
        Vec3 operator-(const Vec3& o) const     { return {x - o.x, y - o.y, z - o.z}; }
        Vec3 operator*(double k) const          { return {x * k, y * k, z * k}; }
        double norm() const                     { return std::sqrt(x * x + y * y + z * z); }
};

// 3x3 旋转矩阵（行主序存储），替代 Eigen 保持轻量
// 默认构造为单位阵（旋转矩阵的合理默认值）
struct Mat33 {
        double m[9] = { 1, 0, 0,
                        0, 1, 0,
                        0, 0, 1};

        static Mat33 identity();
        static Mat33 rotZ(double a);                    // 绕 z 轴旋转 a 弧度
        static Mat33 rotY(double a);                    // 绕 y 轴旋转 a 弧度
        static Mat33 fromRows(const std::vector<double>& v);  // 行主序 9 个数

        Mat33 transposed() const;
        Mat33 operator*(const Mat33& o) const;
        Vec3 operator*(const Vec3& v) const;
};

inline Mat33 Mat33::identity() {
        Mat33 r;
        r.m[0] = r.m[4] = r.m[8] = 1.0;
        return r;
}

inline Mat33 Mat33::rotZ(double a) {
        Mat33 r = identity();
        double c = std::cos(a), s = std::sin(a);
        r.m[0] = c; r.m[1] = -s;
        r.m[3] = s; r.m[4] =  c;
        return r;
}

inline Mat33 Mat33::rotY(double a) {
        Mat33 r = identity();
        double c = std::cos(a), s = std::sin(a);
        r.m[0] = c; r.m[2] = s;
        r.m[6] = -s; r.m[8] = c;
        return r;
}

inline Mat33 Mat33::fromRows(const std::vector<double>& v) {
        Mat33 r;
        for (int i = 0; i < 9; ++i) r.m[i] = v[i];
        return r;
}

inline Mat33 Mat33::transposed() const {
        Mat33 r;
        for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                        r.m[i * 3 + j] = m[j * 3 + i];
        return r;
}

inline Mat33 Mat33::operator*(const Mat33& o) const {
        Mat33 r;
        for (int i = 0; i < 3; ++i)
                for (int k = 0; k < 3; ++k)
                        for (int j = 0; j < 3; ++j)
                                r.m[i * 3 + j] += m[i * 3 + k] * o.m[k * 3 + j];
        return r;
}

inline Vec3 Mat33::operator*(const Vec3& v) const {
        return {m[0] * v.x + m[1] * v.y + m[2] * v.z, 
                m[3] * v.x + m[4] * v.y + m[5] * v.z,
                m[6] * v.x + m[7] * v.y + m[8] * v.z};
}

#endif
