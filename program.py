import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
import time
import random
import numpy as np
from collections import Counter, defaultdict
import threading


# ==================== 随机数生成器 ====================
def c_style_rand(N, size):
    return [random.randint(0, N - 1) for _ in range(size)]


def uniform_rand(N, size):
    return np.random.randint(0, N, size=size)


def normal_rand(N, size):
    mean = (N - 1) / 2
    std = (N - 1) / 6
    values = np.random.normal(mean, std, size=size)
    values = np.clip(np.round(values), 0, N - 1).astype(int)
    return values


# ==================== 测评工具实现 ====================
class RandomnessTester:
    def __init__(self, rand_func, N=100, sample_size=10000):
        self.rand_func = rand_func
        self.N = N
        self.sample_size = sample_size
        self.samples = rand_func(N, sample_size)

    def distribution_test(self, ax1, ax2):
        freq = Counter(self.samples)
        freq = {k: freq.get(k, 0) for k in range(self.N)}
        ax1.clear()
        ax1.bar(freq.keys(), freq.values())
        ax1.set_title('Frequency Distribution')
        ax1.set_xlabel('Number')
        ax1.set_ylabel('Count')


        positions = defaultdict(list)
        for idx, val in enumerate(self.samples):
            positions[val].append(idx)
        intervals = np.concatenate([np.diff(np.array(pos)) for pos in positions.values()])  
        ax2.clear()
        ax2.hist(intervals, bins=30, density=True)
        ax2.set_title('Interval Distribution')
        ax2.set_xlabel('Interval Between Repetitions')
        ax2.set_ylabel('Frequency')

    def correctness_test(self, ax1, ax2):
        mean = np.mean(self.samples)
        var = np.var(self.samples)
        skew = stats.skew(self.samples)
        kurtosis = stats.kurtosis(self.samples)
        result = f"统计特征：均值={mean:.2f}, 方差={var:.2f}, 偏度={skew:.2f}, 峰度={kurtosis:.2f}\n"

        ax1.clear()
        stats.probplot(self.samples, dist="uniform", sparams=(0, self.N - 1), plot=ax1)
        ax1.set_title('Q-Q Plot (Uniform)')

        ax2.clear()
        stats.probplot(self.samples, dist="norm", sparams=(mean, np.std(self.samples)), plot=ax2)
        ax2.set_title('Q-Q Plot (Normal)')
        return result

    def distribution_check(self):
        observed = np.array([Counter(self.samples).get(i, 0) for i in range(self.N)])
        expected = np.full(self.N, len(self.samples) / self.N)
        chi2, p = stats.chisquare(observed, expected)
        result = f"卡方检验：χ²={chi2:.2f}, p值={p:.4f}\n"

        uniform_samples = np.random.uniform(0, self.N - 1, len(self.samples))
        ks_stat, p_value = stats.ks_2samp(self.samples, uniform_samples)
        result += f"KS检验：D={ks_stat:.4f}, p值={p_value:.4f}\n"
        return result

    def randomness_test(self, ax):

        lag = 1
        acf = np.corrcoef(self.samples[:-lag], self.samples[lag:])[0, 1]
        result = f"滞后1阶自相关系数：{acf:.4f}\n"


        samples_array = np.array(self.samples)


        delta = np.diff(samples_array)
        trend_changes = np.diff(np.sign(delta)) != 0


        runs = np.sum(trend_changes) + 1

        result += f"游程数量：{runs}\n"


        ax.clear()
        ax.acorr(self.samples, maxlags=10)
        ax.set_title('Autocorrelation')
        return result

    def performance_test(self):
        start_time = time.time()
        test_samples = self.rand_func(self.N, self.sample_size) 
        duration = time.time() - start_time


        if isinstance(test_samples, np.ndarray):
            mem_usage = test_samples.nbytes / 1024
        else:
            mem_usage = sum(x.__sizeof__() for x in test_samples[:self.sample_size]) / 1024

        result = f"生成速度：{int(self.sample_size / duration)} samples/sec\n"
        result += f"内存占用：约{mem_usage:.2f} KB for {self.sample_size} samples\n"
        return result

    def entropy_test(self):
        freq = Counter(self.samples)
        probabilities = np.array([freq.get(i, 0) / len(self.samples) for i in range(self.N)])
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return f"信息熵：{entropy:.4f}（理论最大熵：{np.log2(self.N):.4f}）\n"


# ==================== GUI 应用 ====================
class RandomnessApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("随机数分析系统")
        self.geometry("1200x800")
        self.configure(bg="#f0f0f0")
        self.create_widgets()
        self.tester = None

    def create_widgets(self):
        style = ttk.Style()
        style.configure("Header.TLabel", font=("微软雅黑", 14, "bold"))
        style.configure("Result.TLabel", padding=5)

        title_frame = ttk.Frame(self)
        title_frame.pack(pady=10)
        ttk.Label(title_frame, text="🎲 随机数生成器分析工具", style="Header.TLabel").pack()

        input_frame = ttk.LabelFrame(self, text="参数设置", padding=10)
        input_frame.pack(padx=10, pady=5, fill="x")

        self.rand_choice = tk.IntVar(value=1)
        ttk.Radiobutton(input_frame, text="C风格rand()%N", variable=self.rand_choice, value=1).grid(row=0, column=0,
                                                                                                    sticky="w", padx=5,
                                                                                                    pady=2)
        ttk.Radiobutton(input_frame, text="均匀分布", variable=self.rand_choice, value=2).grid(row=1, column=0,
                                                                                               sticky="w", padx=5,
                                                                                               pady=2)
        ttk.Radiobutton(input_frame, text="正态分布", variable=self.rand_choice, value=3).grid(row=2, column=0,
                                                                                               sticky="w", padx=5,
                                                                                               pady=2)

        ttk.Label(input_frame, text="N值:").grid(row=0, column=1, sticky="e", padx=5)
        self.n_entry = ttk.Entry(input_frame, width=15)
        self.n_entry.insert(0, "100")
        self.n_entry.grid(row=0, column=2, padx=5)

        ttk.Label(input_frame, text="样本数量:").grid(row=1, column=1, sticky="e", padx=5)
        self.sample_entry = ttk.Entry(input_frame, width=15)
        self.sample_entry.insert(0, "10000")
        self.sample_entry.grid(row=1, column=2, padx=5)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(padx=10, pady=5, fill="x")

        tests = [
            (1, "分布测试"),
            (2, "正确性验证"),
            (3, "统计检验"),
            (4, "随机性检测"),
            (5, "性能测试"),
            (6, "熵值计算"),
            (7, "退出")
        ]
        self.buttons = {}
        for i, (key, text) in enumerate(tests):
            btn = ttk.Button(btn_frame, text=text,
                             command=lambda k=key: self.run_test(k),
                             width=15)
            btn.grid(row=0, column=i, padx=2)
            self.buttons[key] = btn

        result_frame = ttk.Notebook(self)
        result_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.text_output = tk.Text(result_frame, wrap="word", height=20, bg="#ffffff")
        result_frame.add(self.text_output, text="文本输出")

        plot_container = ttk.Frame(result_frame)
        result_frame.add(plot_container, text="图表显示")
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_test(self, test_num):
        try:
            N = int(self.n_entry.get())
            sample_size = int(self.sample_entry.get())
            if N <= 0 or sample_size <= 0:
                raise ValueError("数值必须大于0")
        except ValueError as e:
            messagebox.showerror("输入错误", f"请确保输入有效的数字：{str(e)}")
            return

        rand_func_map = {
            1: c_style_rand,
            2: uniform_rand,
            3: normal_rand
        }
        rand_func = rand_func_map[self.rand_choice.get()]
        self.tester = RandomnessTester(rand_func, N, sample_size)
        threading.Thread(target=self.execute_test, args=(test_num,), daemon=True).start()

    def execute_test(self, test_num):
        self.buttons[test_num].config(state="disabled")
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, "正在运行测试...\n")
        try:
            self.fig.clear()
            if test_num == 1:
                self.run_distribution_test()
            elif test_num == 2:
                self.run_correctness_test()
            elif test_num == 3:
                self.run_distribution_check()
            elif test_num == 4:
                self.run_randomness_test()
            elif test_num == 5:
                self.run_performance_test()
            elif test_num == 6:
                self.run_entropy_test()
            elif test_num == 7:
                self.destroy()
        finally:
            self.buttons[test_num].config(state="normal")

    def run_distribution_test(self):
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        self.tester.distribution_test(ax1, ax2)
        self.fig.tight_layout()
        self.canvas.draw()
        self.text_output.insert(tk.END, "分布测试完成\n")

    def run_correctness_test(self):
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        result = self.tester.correctness_test(ax1, ax2)
        self.fig.tight_layout()
        self.canvas.draw()
        self.text_output.insert(tk.END, result + "正确性验证完成\n")

    def run_distribution_check(self):
        result = self.tester.distribution_check()
        self.text_output.insert(tk.END, result + "统计分布检验完成\n")

    def run_randomness_test(self):
        ax = self.fig.add_subplot(111)
        result = self.tester.randomness_test(ax)
        self.fig.tight_layout()
        self.canvas.draw()
        self.text_output.insert(tk.END, result + "随机性检测完成\n")

    def run_performance_test(self):
        result = self.tester.performance_test()
        self.text_output.insert(tk.END, result + "性能测试完成\n")

    def run_entropy_test(self):
        result = self.tester.entropy_test()
        self.text_output.insert(tk.END, result + "熵值计算完毕\n")


if __name__ == "__main__":
    app = RandomnessApp()
    app.mainloop()
