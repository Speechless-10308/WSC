import re
import matplotlib.pyplot as plt
import matplotlib as mpl

def parse_log_file(filepath):
    """
    解析指定的日志文件，提取 epoch 和 validation accuracy。
    """
    epochs = []
    accuracies = []
    
    # 正则表达式匹配包含 "Validation" 和 "Acc" 的行
    # \d+ 匹配一个或多个数字, [\d.]+ 匹配一个或多个数字或点（用于浮点数）
    val_pattern = re.compile(r"Validation - .* Acc: ([\d.]+)")
    
    with open(filepath, 'r') as f:
        # 假设每个 Validation 行对应一个 Epoch，从 1 开始计数
        current_epoch = 1
        for line in f:
            match = val_pattern.search(line)
            if match:
                # 提取 Acc 后面的数值，并转换为浮点数
                acc = float(match.group(1)) * 100
                accuracies.append(acc)
                epochs.append(current_epoch)
                current_epoch += 1
                
    return epochs, accuracies

def setup_publication_style():
    """
    设置 Matplotlib 以生成符合学术出版物标准的图表。
    """
    # 检查并设置字体为 Times New Roman
    try:
        mpl.font_manager.findfont("Times New Roman", fallback_to_default=False)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    except:
        print("Times New Roman not found, using default serif font.")
        plt.rcParams['font.family'] = 'serif'

    # 设置字体大小
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams['figure.titlesize'] = 24
    
    # 设置坐标轴线宽和刻度线样式
    plt.rcParams['axes.linewidth'] = 1.5     # 坐标轴线框的宽度
    plt.rcParams['xtick.direction'] = 'in'   # X轴刻度线朝内
    plt.rcParams['ytick.direction'] = 'in'   # Y轴刻度线朝内
    
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    # 确保数学公式也使用 Times New Roman 风格字体
    plt.rcParams['mathtext.fontset'] = 'stix'


# --- Part 3 & 4: 绘制并美化图表 ---
def plot_accuracy_curves(log_data, title, output_filename="accuracy_plot.pdf"):
    """
    绘制多条 accuracy 曲线并保存为高质量文件。
    
    Args:
        log_data (dict): 一个字典，键是曲线的标签(如 'Experiment 1'), 
                         值是 (epochs, accuracies) 元组。
        output_filename (str): 保存图像的文件名。
    """
    # 创建一个图像和一个子图
    fig, ax = plt.subplots(figsize=(4, 3))

    # 定义一些颜色和线条样式
    colors = ["#4c9fd5", "#f18b95", '#d62728', '#9467bd']
    linestyles = ['-', '-', '-.', ':']
    
    # 遍历所有实验数据并绘制
    for i, (label, data) in enumerate(log_data.items()):
        epochs, accuracies = data
        ax.plot(epochs, accuracies,
                label=label,
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=1.5)
        
    ax.set_title(title, fontsize=19, fontweight='regular')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    

    ax.set_ylim(bottom=0, top=75) # 紧凑的y轴可以让变化更明显
    ax.set_yticks([10, 30, 60])

    ax.legend(
        loc='upper right',
        fancybox=True,
        framealpha=0.8,
        edgecolor='gray'
    )
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    setup_publication_style()
    
    # 要处理的日志文件和它们在图例中的标签
    log_files = {
        "clean": "/home/wangjunjie/WSC/test/results/linear_prob/cifar100_0.0_elr+/0826_1407/training.log",
        "noise": "/home/wangjunjie/WSC/test/results/linear_prob/cifar100_0.9_elr+/0826_1400/training.log"
    }
    
    # 3. 解析所有日志文件
    all_data = {}
    for label, filepath in log_files.items():
        try:
            epochs, accuracies = parse_log_file(filepath)
            all_data[label] = (epochs, accuracies)
        except FileNotFoundError:
            print(f"Error: Log file not found at '{filepath}'")
    
    # 4. 如果成功解析了数据，则绘制图表
    if all_data:
        # 保存为 PDF (矢量图, 无限放大不失真，推荐用于 LaTeX)
        plot_accuracy_curves(all_data, "ELR+/CIFAR100", output_filename="elr_linear_prob_c100.pdf")
        
        # 也可以保存为 PNG (位图, 适用于 PPT 或 Word)
        # plot_accuracy_curves(all_data, output_filename="accuracy_plot.png")