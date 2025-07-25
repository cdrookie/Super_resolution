import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QLineEdit
)
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer

def get_base_path():
    """获取程序运行时的基础路径（支持打包后的exe）"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

class ImageLabel(QLabel):
    """自定义 QLabel 类，用于绘制九宫格线和显示鼠标坐标"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)  # 启用鼠标跟踪
        self.mouse_pos_label = None  # 用于显示坐标的 QLabel
        self.image_size = None  # 存储图片的原始大小

    def setMousePosLabel(self, label):
        """设置用于显示鼠标坐标的 QLabel"""
        self.mouse_pos_label = label

    def setPixmap(self, pixmap):
        """重写 setPixmap，确保图片保持比例并完整显示"""
        self.image_size = pixmap.size()
        # 按比例缩放图片以适应固定区域 (900x500)，保持宽高比
        scaled_pixmap = pixmap.scaled(900, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(scaled_pixmap)
        # 调整 QLabel 大小以匹配缩放后的图片大小
        self.setFixedSize(scaled_pixmap.size())

    def paintEvent(self, event):
        """重写 paintEvent，在图片上绘制九宫格线"""
        super().paintEvent(event)
        if not self.pixmap():
            return

        painter = QPainter(self)
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)

        # 获取当前 QLabel 的宽度和高度
        width = self.width()
        height = self.height()

        # 绘制横线（1/3 和 2/3 位置）
        painter.drawLine(0, height // 3, width, height // 3)
        painter.drawLine(0, 2 * height // 3, width, 2 * height // 3)

        # 绘制竖线（1/3 和 2/3 位置）
        painter.drawLine(width // 3, 0, width // 3, height)
        painter.drawLine(2 * width // 3, 0, 2 * width // 3, height)

        painter.end()

    def mouseMoveEvent(self, event):
        """处理鼠标移动事件，显示归一化坐标"""
        if self.pixmap() and self.image_size and self.mouse_pos_label:
            # 获取鼠标在 QLabel 上的位置
            pos = event.pos()
            # 计算相对于图片的归一化坐标
            x_ratio = pos.x() / self.width()
            y_ratio = pos.y() / self.height()
            # 显示坐标，保留两位小数
            self.mouse_pos_label.setText(f"({x_ratio:.2f}W, {y_ratio:.2f}H)")
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        """鼠标离开时清空坐标显示"""
        if self.mouse_pos_label:
            self.mouse_pos_label.setText("")
        super().leaveEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("构图标注工具")
        self.setGeometry(100, 100, 1200, 600)

        # 选择数据集目录
        directory = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if not directory:
            QMessageBox.critical(self, "错误", "未选择数据集目录，程序将退出")
            sys.exit()

        # 加载图片路径（包括所有子目录）
        self.images = []
        for root, _, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.images.append(os.path.join(root, f))
        self.images.sort()

        if not self.images:
            QMessageBox.critical(self, "错误", "所选目录中未找到图片，程序将退出")
            sys.exit()

        # 初始化标签字典
        self.labels = {}
        for root, _, files in os.walk(directory):
            for f in files:
                if f.lower().endswith('.txt'):
                    txt_path = os.path.join(root, f)
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as file:
                            line = file.readline().strip()
                            parts = line.split()
                            if len(parts) == 23:  # image_name + 22 labels
                                image_name = parts[0]
                                image_path = os.path.join(root, image_name)
                                labels = [int(x) for x in parts[1:23]]
                                if len(labels) == 22 and all(0 <= x <= 1 for x in labels):
                                    self.labels[image_path] = labels
                                else:
                                    print(f"Invalid labels in {txt_path}: Expected 22 binary values (0-1)")
                            else:
                                print(f"Invalid format in {txt_path}: Expected 'image_name label1 ... label22'")
                    except Exception as e:
                        print(f"Failed to read {txt_path}: {e}")

        # 当前图片索引
        self.current_index = 0

        # 创建界面元素
        self.image_label = ImageLabel()  # 使用自定义 ImageLabel
        self.image_label.setAlignment(Qt.AlignCenter)

        # 用于显示鼠标坐标的标签
        self.mouse_pos_label = QLabel()
        self.mouse_pos_label.setFixedHeight(20)
        self.mouse_pos_label.setStyleSheet("font-size: 18px; color: blue;")
        self.image_label.setMousePosLabel(self.mouse_pos_label)

        # 标签按钮
        self.class_buttons = []
        self.class_names = [
            "Center_P\n中心-点", "Center_A\n中心-面", "RoT_P\n三分-点", "RoT_L\n三分-线", "RoT_A\n三分-面", 
            "Horizontal_L\n水平", "HorizonArranged_P/A\n水平排列", "Vertical_L\n垂直", "VerticalArranged_P/A\n垂直排列",
            "Diagonal_P\n对角-点", "Diagonal_L\n对角-线", "Diagonal_A\n对角-面", "Curved_P\n曲线-点", "Curved_L\n曲线-线",
            "Pattern\n重复", "FillFrame_A\n填充", "VanishedPoint_L\n消失点", "Symmetric\n对称",
            "Triangle_P\n三角-点", "Triangle_LA\n三角-线面", "Framework\n框架", "Diffuse_L\n扩散"
        ]
        # 大类分组（用于颜色分配）
        self.class_groups = [
            ["Center_P\n中心-点", "Center_A\n中心-面"],
            ["RoT_P\n三分-点", "RoT_L\n三分-线", "RoT_A\n三分-面"],
            ["Horizontal_L\n水平", "HorizonArranged_P/A\n水平排列"],
            ["Vertical_L\n垂直", "VerticalArranged_P/A\n垂直排列"],
            ["Diagonal_P\n对角-点", "Diagonal_L\n对角-线", "Diagonal_A\n对角-面"],
            ["Curved_P\n曲线-点", "Curved_L\n曲线-线"],
            ["Pattern\n重复"],
            ["FillFrame_A\n填充"],
            ["VanishedPoint_L\n消失点"],
            ["Symmetric\n对称"],
            ["Triangle_P\n三角-点", "Triangle_LA\n三角-线面"],
            ["Framework\n框架"],
            ["Diffuse_L\n扩散"]
        ]
        for i, name in enumerate(self.class_names):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setFixedHeight(60)  # 固定按钮高度以适应两行
            group_idx = next(idx for idx, group in enumerate(self.class_groups) if name in group)
            bg_color = "#ecf2ff" if group_idx % 2 == 0 else "#ffecf3"
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg_color};
                    border: 1px solid gray;
                    padding: 5px;
                    color: black;
                    min-width: 140px;
                    font-size: 15px;
                    height: 60px;
                    text-align: center;
                }}
                QPushButton:checked {{
                    background-color: #90EE90;
                    border: 1px solid darkgreen;
                    color: red;
                }}
            """)
            btn.toggled.connect(lambda checked, idx=i: self.handle_label_toggle(idx, checked))
            self.class_buttons.append(btn)

        self.prev_button = QPushButton("上一张")
        self.prev_button.clicked.connect(self.prev_image)
        self.prev_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9933;
                border: 1px solid gray;
                padding: 10px;
                font-size: 18px;
            }
        """)
        self.prev_button.setFixedHeight(80)

        self.next_button = QPushButton("下一张\n(或者按s保存并切图)")
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9933;
                border: 1px solid gray;
                padding: 10px;
                font-size: 18px;
            }
        """)
        self.next_button.setFixedHeight(80)

        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.save_labels)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9933;
                border: 1px solid gray;
                padding: 10px;
                font-size: 18px;
            }
        """)
        self.save_button.setFixedHeight(80)

        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("输入图片ID (如 123)")
        self.jump_input.setFixedWidth(100)
        self.jump_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                font-size: 14px;
            }
        """)
        self.jump_input.setFixedHeight(80)

        self.jump_button = QPushButton("跳转")
        self.jump_button.clicked.connect(self.jump_to_image)
        self.jump_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9933;
                border: 1px solid gray;
                padding: 10px;
                font-size: 18px;
            }
        """)
        self.jump_button.setFixedHeight(80)

        self.status_label = QLabel()
        self.message_label = QLabel()

        # 设置布局
        main_layout = QVBoxLayout()
        
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label)
        
        main_layout.addLayout(image_layout)
        main_layout.addWidget(self.mouse_pos_label)  # 添加鼠标坐标显示标签

        # 类别按钮布局，限制为4行
        button_layout = QVBoxLayout()
        row_layouts = [QHBoxLayout() for _ in range(4)]
        button_assignments = [
            [0, 1, 2, 3, 4, 5],  # Center_P, Center_A, RoT_P, RoT_L, RoT_A, Horizontal_L
            [6, 7, 8, 9, 10, 11],  # HorizonArranged_P/A, Vertical_L, VerticalArranged_P/A, Diagonal_P, Diagonal_L, Diagonal_A
            [12, 13, 14, 15, 16, 17],  # Curved_P, Curved_L, Pattern, FillFrame_A, VanishedPoint_L, Symmetric
            [18, 19, 20, 21]  # Triangle_P, Triangle_LA, Framework, Diffuse_L
        ]
        for row_idx, indices in enumerate(button_assignments):
            for idx in indices:
                row_layouts[row_idx].addWidget(self.class_buttons[idx])
            button_layout.addLayout(row_layouts[row_idx])
        main_layout.addLayout(button_layout)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.status_label)
        
        next_button_layout = QVBoxLayout()
        next_button_layout.addWidget(self.next_button)
        
        nav_layout.addLayout(next_button_layout)
        nav_layout.addWidget(self.jump_input)
        nav_layout.addWidget(self.jump_button)
        nav_layout.addWidget(self.save_button)
        nav_layout.addWidget(self.message_label)
        main_layout.addLayout(nav_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 初始显示更新
        self.update_display()

    def update_display(self):
        """更新图片显示、按钮状态和图片名称"""
        if not self.images:
            return
        image_path = self.images[self.current_index]
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)  # 使用自定义 setPixmap

        if image_path not in self.labels:
            self.labels[image_path] = [0] * 22

        for i, btn in enumerate(self.class_buttons):
            btn.blockSignals(True)
            selected = self.labels[image_path][i] == 1
            btn.setChecked(selected)
            btn.setText(self.class_names[i])
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.blockSignals(False)

        image_name = os.path.basename(image_path)
        self.status_label.setText(f"图片 {self.current_index + 1} / {len(self.images)}: {image_name}")

    def handle_label_toggle(self, idx, checked):
        """处理标签按钮切换"""
        image_path = self.images[self.current_index]
        if image_path not in self.labels:
            self.labels[image_path] = [0] * 22

        self.labels[image_path][idx] = 1 if checked else 0
        self.class_buttons[idx].blockSignals(True)
        self.class_buttons[idx].setText(self.class_names[idx])
        self.class_buttons[idx].setChecked(checked)
        self.class_buttons[idx].style().unpolish(self.class_buttons[idx])
        self.class_buttons[idx].style().polish(self.class_buttons[idx])
        self.class_buttons[idx].blockSignals(False)
        # self.show_message(f"标签 {self.class_names[idx].split('\n')[0]} {'已选择' if checked else '已取消'}")

    def save_current_label(self):
        """保存当前图片的标签"""
        if not self.images:
            return
        image_path = self.images[self.current_index]
        if image_path in self.labels:
            txt_path = os.path.splitext(image_path)[0] + '.txt'
            image_name = os.path.basename(image_path)
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"{image_name} {' '.join(map(str, self.labels[image_path]))}\n")
                self.show_message("保存成功")
            except Exception as e:
                self.show_message(f"保存失败: {e}")

    def save_labels(self):
        """保存所有图片的标签"""
        try:
            for image_path, labels in self.labels.items():
                txt_path = os.path.splitext(image_path)[0] + '.txt'
                image_name = os.path.basename(image_path)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"{image_name} {' '.join(map(str, labels))}\n")
            self.show_message("所有标签已成功保存")
        except Exception as e:
            self.show_message(f"保存失败: {e}")

    def show_message(self, message):
        """显示消息并在2秒后清空"""
        self.message_label.setText(message)
        QTimer.singleShot(2000, lambda: self.message_label.setText(""))

    def prev_image(self):
        """切换到上一张图片并保存当前标签"""
        if self.current_index > 0:
            self.save_current_label()
            self.current_index -= 1
            self.update_display()

    def next_image(self):
        """切换到下一张图片并保存当前标签"""
        if self.current_index < len(self.images) - 1:
            self.save_current_label()
            self.current_index += 1
            self.update_display()

    def jump_to_image(self):
        """跳转到指定ID的图片"""
        input_id = self.jump_input.text().strip()
        if not input_id.isdigit():
            self.show_message("请输入有效的数字ID")
            return

        formatted_id = f"{int(input_id):06d}"
        target_names = [f"img{formatted_id}.jpg", f"img{formatted_id}.jpeg"]

        for i, image_path in enumerate(self.images):
            image_name = os.path.basename(image_path)
            if image_name in target_names:
                self.save_current_label()
                self.current_index = i
                self.update_display()
                self.show_message(f"保存成功，已跳转到图片 {image_name}")
                self.show_message(f"已跳转到 {image_name}")
                return

        self.show_message(f"未找到ID为 {input_id} 的图片")

    def keyPressEvent(self, event):
        """处理键盘按键事件"""
        if event.key() == Qt.Key_S:
            self.save_current_label()
            if self.current_index < len(self.images) - len(self.images) - 1:
                self.current_index += 1
                self.update_display()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())