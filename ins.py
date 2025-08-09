import numpy as np
import matplotlib.pyplot as plt

class SigmaFitter:
    def __init__(self, centers, sigmas, degree=2):
        self.centers = np.array(centers)
        self.sigmas = np.array(sigmas)
        self.degree = degree
        self.coeffs = np.polyfit(self.centers, self.sigmas, degree)
        self.poly = np.poly1d(self.coeffs)

    def predict(self, center):
        return self.poly(center)

    def get_polynomial(self):
        return self.poly

    def print_polynomial(self):
        print(f"Fitted polynomial (degree {self.degree}):")
        terms = [f"{coef:.6g}*x^{i}" if i > 0 else f"{coef:.6g}" 
                 for i, coef in enumerate(self.coeffs[::-1])]
        poly_str = " + ".join(terms[::-1])
        print("sigma(x) =", poly_str)

    def plot_fit(self, num_points=200):
        plt.figure(figsize=(8,5))
        # 原始数据点
        plt.scatter(self.centers, self.sigmas, color='blue', label='Data Points')
        
        # 拟合曲线
        x_min, x_max = self.centers.min(), self.centers.max()
        x_vals = np.linspace(x_min, x_max, num_points)
        y_vals = self.poly(x_vals)
        plt.plot(x_vals, y_vals, 'r-', label=f'Poly fit (deg {self.degree})')
        
        plt.xlabel('Center (2θ degrees)')
        plt.ylabel('Sigma')
        plt.title('Sigma vs Center Polynomial Fit')
        plt.legend()
        plt.grid(True)
        plt.show()


# # 你的拟合数据
# centers = [28.402, 47.268, 56.088, 69.099, 76.347, 88.000, 94.920]
# sigmas =  [0.023,  0.022,  0.020,  0.022,  0.021,  0.021,  0.026]

# # 实例化拟合器，选择多项式阶数
# sigma_fitter = SigmaFitter(centers, sigmas, degree=2)

# # 打印拟合公式
# sigma_fitter.print_polynomial()

# # 预测一个新的 sigma 值
# two_diffraction = 60.0
# predicted_sigma = sigma_fitter.predict(two_diffraction)
# print(f"\nPredicted sigma at center={two_diffraction:.3f}°: {predicted_sigma:.6f}")

# # 绘制拟合曲线和数据点
# sigma_fitter.plot_fit()
