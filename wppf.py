import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev
from scipy.special import voigt_profile
from scipy.optimize import curve_fit

class XRDAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.x, self.y = self._import_file()
        self.background = None
        self.y_corrected = None
        self.mask = np.ones_like(self.x, dtype=bool)
        self.fitted_peaks = []

    def _import_file(self):
        if self.filename.split('.')[-1] == 'txt':
            df = pd.read_csv(self.filename, sep=r'\s+', header=None)
            x, y = np.array(df).T
            return x, y
        else:
            raise ValueError("Only .txt files supported")

    def add_mask_region(self, lower, upper):
        self.mask &= ~((self.x > lower) & (self.x < upper))

    def fit_background(self, method='chebyshev', degree=4):
        x_masked = self.x[self.mask]
        y_masked = self.y[self.mask]

        if method == 'chebyshev':
            cheb_fit = Chebyshev.fit(x_masked, y_masked, degree)
            self.background = cheb_fit(self.x)
        else:
            raise ValueError("Only 'chebyshev' method is implemented")

        self.y_corrected = self.y - self.background

    def voigt_area_model(self, x, *params):
        """多个Voigt峰的叠加模型，每四个参数表示一个峰：area, center, sigma, gamma"""
        y = np.zeros_like(x)
        for i in range(0, len(params), 4):
            area, center, sigma, gamma = params[i:i+4]
            y += area * voigt_profile(x - center, sigma, gamma)
        return y

    def fit_multiple_peaks(self, peak_guesses, window=1.5):
        """
        联合拟合多个衍射峰（仅主峰）
        - peak_guesses: 初始Kα1主峰位置列表
        """
        if self.y_corrected is None:
            raise ValueError("Background must be fitted before peak fitting.")
        self.fitted_peaks.clear()

        x_fit_region = []
        y_fit_region = []
        p0 = []
        bounds_lower = []
        bounds_upper = []

        for center in peak_guesses:
            mask = (self.x > center - window) & (self.x < center + window)
            x_fit_region.append(self.x[mask])
            y_fit_region.append(self.y_corrected[mask])
            
            area_guess = np.trapezoid(self.y_corrected[mask], self.x[mask])

            p0 += [area_guess, center, 0.2, 0.2]
            bounds_lower += [0, center - 0.5, 0.001, 0.001]
            bounds_upper += [np.inf, center + 0.5, 5.0, 5.0]

        x_all = np.concatenate(x_fit_region)
        y_all = np.concatenate(y_fit_region)

        try:
            popt, _ = curve_fit(lambda x, *params: self.voigt_area_model(x, *params),
                                x_all, y_all, p0=p0, bounds=(bounds_lower, bounds_upper))

            for i in range(0, len(popt), 4):
                area, center, sigma, gamma = popt[i:i+4]
                fit_curve = self.voigt_area_model(self.x, *popt[i:i+4])
                self.fitted_peaks.append((center, fit_curve, (area, center, sigma, gamma)))

        except RuntimeError:
            print("⚠️ 联合拟合失败，请检查初始参数或窗口设置。")

    def print_peak_parameters(self):
        print("\nFitted Peaks (area-based):")
        for i, (center, _, params) in enumerate(self.fitted_peaks, start=1):
            if params is not None:
                area, center, sigma, gamma = params
                print(f"{i}: center={center:.3f}°, area={area:.3f}, sigma={sigma:.3f}, gamma={gamma:.3f}")
            else:
                print(f"{i}: center={center:.3f}° — fit failed")

    def plot_results(self):
        plt.figure(figsize=(14, 8))
        
        # 原始数据
        plt.plot(self.x, self.y, label='Raw Data', alpha=0.6)

        # 计算总Voigt曲线
        total_voigt = np.sum([fit for _, fit, _ in self.fitted_peaks], axis=0) if self.fitted_peaks else 0

        # 总拟合 = Voigt 总和 + 背景
        if self.background is not None:
            total_fit = total_voigt + self.background
            plt.plot(self.x, total_fit, label='Total Fit', color='red', linewidth=2)

        plt.xlabel('2θ (degrees)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)
        plt.title("XRD Peak Fitting")
        plt.show()
    
    def plot_all_voigt(self):
        plt.figure(figsize=(14, 8))

        # 原始数据
        plt.plot(self.x, self.y, label='Raw Data', color='black', alpha=0.6)

        if self.background is not None:
            background = self.background
        else:
            background = np.zeros_like(self.x)

        # 每个 Voigt 拟合曲线 + 背景
        for i, (center, fit_curve, _) in enumerate(self.fitted_peaks, start=1):
            voigt_with_bg = fit_curve + background
            plt.plot(self.x, voigt_with_bg, label=f'Peak {i} @ {center:.2f}°', linewidth=1)

        # 总拟合曲线
        if self.fitted_peaks:
            total_voigt = np.sum([fit for _, fit, _ in self.fitted_peaks], axis=0)
            total_fit = total_voigt + background
            plt.plot(self.x, total_fit, label='Total Fit (Voigt + Background)', color='red', linewidth=2)

        plt.xlabel('2θ (degrees)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)
        plt.title("XRD Fit with Individual Voigt Peaks + Background")
        plt.show()


