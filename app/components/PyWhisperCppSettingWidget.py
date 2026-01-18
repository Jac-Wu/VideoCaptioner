import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    ComboBoxSettingCard,
    ExpandLayout,
    FluentIcon,
    HyperlinkButton,
    HyperlinkCard,
    InfoBar,
    InfoBarPosition,
    MessageBoxBase,
    PrimaryPushSettingCard,
    ProgressBar,
    RangeSettingCard,
    SettingCardGroup,
    SingleDirectionScrollArea,
    SubtitleLabel,
    SwitchSettingCard,
    TableItemDelegate,
    TableWidget,
)
from qfluentwidgets import FluentIcon as FIF

from app.common.config import cfg
from app.config import MODEL_PATH
from app.core.entities import (
    TranscribeLanguageEnum,
    PyWhisperModelEnum,
    VadMethodEnum,
)
from app.core.utils.logger import setup_logger
from app.core.utils.platform_utils import open_folder
from app.thread.file_download_thread import FileDownloadThread

logger = setup_logger("pywhisper_download")

# 使用与 WhisperCpp 相同的模型配置
PYWHISPER_CPP_MODELS = [
    {
        "label": "Tiny",
        "value": "ggml-tiny.bin",
        "size": "77.7 MB",
        "downloadLink": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        "mirrorLink": "https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-tiny.bin",
        "sha": "bd577a113a864445d4c299885e0cb97d4ba92b5f",
    },
    {
        "label": "Base",
        "value": "ggml-base.bin",
        "size": "148 MB",
        "downloadLink": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "mirrorLink": "https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-base.bin",
        "sha": "465707469ff3a37a2b9b8d8f89f2f99de7299dac",
    },
    {
        "label": "Small",
        "value": "ggml-small.bin",
        "size": "488 MB",
        "downloadLink": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        "mirrorLink": "https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-small.bin",
        "sha": "55356645c2b361a969dfd0ef2c5a50d530afd8d5",
    },
    {
        "label": "Medium",
        "value": "ggml-medium.bin",
        "size": "1.53 GB",
        "downloadLink": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        "mirrorLink": "https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-medium.bin",
        "sha": "fd9727b6e1217c2f614f9b698455c4ffd82463b4",
    },
    {
        "label": "large-v1",
        "value": "ggml-large-v1.bin",
        "size": "3.09 GB",
        "downloadLink": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin",
        "mirrorLink": "https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-large-v1.bin",
        "sha": "b1caaf735c4cc1429223d5a74f0f4d0b9b59a299",
    },
    {
        "label": "large-v2",
        "value": "ggml-large-v2.bin",
        "size": "3.09 GB",
        "downloadLink": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin",
        "mirrorLink": "https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-large-v2.bin",
        "sha": "0f4c8e34f21cf1a914c59d8b3ce882345ad349d6",
    },
    {
        "label": "large-v3",
        "value": "ggml-large-v3.bin",
        "size": "3.09 GB",
        "downloadLink": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
        "mirrorLink": "https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-large-v3.bin",
        "sha": "ad82bf6a9043ceed055076d0fd39f5f186ff8062",
    },
    {
        "label": "large-v3-turbo",
        "value": "ggml-large-v3-turbo.bin",
        "size": "1.62 GB",
        "downloadLink": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        "mirrorLink": "https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-large-v3-turbo.bin",
        "sha": "4af2b29d7ec73d781377bfd1758ca957a807e941",
    },
]


class PyWhisperCppDownloadDialog(MessageBoxBase):
    """PyWhisperCpp 下载对话框"""

    # 添加类变量跟踪下载状态
    is_downloading = False

    def __init__(self, parent=None, setting_widget=None):
        super().__init__(parent)
        self.widget.setMinimumWidth(600)
        self.model_download_thread = None
        self._setup_ui()
        self.setting_widget = setting_widget

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout()
        self._setup_info_section(layout)
        layout.addSpacing(20)
        self._setup_model_section(layout)
        self._setup_progress_section(layout)

        self.viewLayout.addLayout(layout)
        self.cancelButton.setText(self.tr("关闭"))
        self.yesButton.hide()

    def _setup_info_section(self, layout):
        """设置信息部分UI"""
        # 标题
        title = SubtitleLabel(self.tr("PyWhisperCpp (CoreML)"), self)
        layout.addWidget(title)
        layout.addSpacing(8)

        # 说明文字
        info_text = BodyLabel(
            self.tr("使用 pywhispercpp 库进行本地转录，支持 CoreML 加速（仅 macOS）"),
            self,
        )
        layout.addWidget(info_text)

    def _setup_model_section(self, layout):
        """设置模型下载部分UI"""
        # 标题和按钮的水平布局
        title_layout = QHBoxLayout()

        # 标题
        model_title = SubtitleLabel(self.tr("模型下载"), self)
        title_layout.addWidget(model_title)

        # 添加打开文件夹按钮
        open_folder_btn = HyperlinkButton("", self.tr("打开模型文件夹"), parent=self)
        open_folder_btn.setIcon(FIF.FOLDER)
        open_folder_btn.clicked.connect(self._open_model_folder)
        title_layout.addStretch()
        title_layout.addWidget(open_folder_btn)

        layout.addLayout(title_layout)
        layout.addSpacing(8)

        # 模型表格
        self.model_table = self._create_model_table()
        self._populate_model_table()
        layout.addWidget(self.model_table)

    def _create_model_table(self):
        """创建模型表格"""
        table = TableWidget(self)
        table.setEditTriggers(TableWidget.NoEditTriggers)
        table.setSelectionMode(TableWidget.NoSelection)
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(
            [self.tr("模型名称"), self.tr("大小"), self.tr("状态"), self.tr("操作")]
        )

        # 设置表格样式
        table.setBorderVisible(True)
        table.setBorderRadius(8)
        table.setItemDelegate(TableItemDelegate(table))

        # 设置列宽
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)

        table.setColumnWidth(1, 100)
        table.setColumnWidth(2, 80)
        table.setColumnWidth(3, 150)

        # 设置行高
        row_height = 45
        table.verticalHeader().setDefaultSectionSize(row_height)

        # 设置表格高度
        header_height = 20
        max_visible_rows = 7
        table_height = row_height * max_visible_rows + header_height + 15
        table.setFixedHeight(table_height)

        return table

    def _setup_progress_section(self, layout):
        """设置进度显示部分UI"""
        self.progress_bar = ProgressBar(self)
        self.progress_label = BodyLabel("", self)
        self.progress_bar.hide()
        self.progress_label.hide()

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)

    def _populate_model_table(self):
        """填充模型表格数据"""
        self.model_table.setRowCount(len(PYWHISPER_CPP_MODELS))
        for i, model in enumerate(PYWHISPER_CPP_MODELS):
            self._add_model_row(i, model)

    def _add_model_row(self, row, model):
        """添加模型表格行"""
        # 模型名称
        name_item = QTableWidgetItem(model["label"])
        name_item.setTextAlignment(Qt.AlignCenter)  # type: ignore
        self.model_table.setItem(row, 0, name_item)

        # 大小
        size_item = QTableWidgetItem(f"{model['size']}")
        size_item.setTextAlignment(Qt.AlignCenter)  # type: ignore
        self.model_table.setItem(row, 1, size_item)

        # 状态
        model_bin_path = os.path.join(MODEL_PATH, model["value"])
        status_item = QTableWidgetItem(
            self.tr("已下载") if os.path.exists(model_bin_path) else self.tr("未下载")
        )
        if os.path.exists(model_bin_path):
            status_item.setForeground(Qt.green)  # type: ignore
        status_item.setTextAlignment(Qt.AlignCenter)  # type: ignore
        self.model_table.setItem(row, 2, status_item)

        # 下载按钮
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(4, 4, 4, 4)

        download_btn = HyperlinkButton(
            "",
            self.tr("重新下载") if os.path.exists(model_bin_path) else self.tr("下载"),
            parent=self,
        )
        download_btn.setIcon(FIF.DOWNLOAD)
        download_btn.clicked.connect(lambda checked, r=row: self._download_model(r))

        button_layout.addStretch()
        button_layout.addWidget(download_btn)
        button_layout.addStretch()
        self.model_table.setCellWidget(row, 3, button_container)

    def _download_model(self, row):
        """下载选中的模型"""
        if PyWhisperCppDownloadDialog.is_downloading:
            InfoBar.warning(
                self.tr("下载进行中"),
                self.tr("请等待当前下载任务完成"),
                duration=3000,
                parent=self,
            )
            return

        PyWhisperCppDownloadDialog.is_downloading = True
        self._set_all_download_buttons_enabled(False)

        model = PYWHISPER_CPP_MODELS[row]
        self.progress_bar.show()
        self.progress_label.show()
        self.progress_label.setText(self.tr(f"正在下载 {model['label']} 模型..."))

        # 禁用当前行的下载按钮
        button_container = self.model_table.cellWidget(row, 3)
        download_btn = button_container.findChild(HyperlinkButton)
        if download_btn:
            download_btn.setEnabled(False)

        def _on_model_download_progress(value, msg):
            self.progress_bar.setValue(int(value))
            self.progress_label.setText(msg)

        def _on_model_download_finished():
            PyWhisperCppDownloadDialog.is_downloading = False
            self._set_all_download_buttons_enabled(True)
            # 更新状态
            status_item = QTableWidgetItem(self.tr("已下载"))
            status_item.setForeground(Qt.green)  # type: ignore
            status_item.setTextAlignment(Qt.AlignCenter)  # type: ignore
            self.model_table.setItem(row, 2, status_item)

            # 更新下载按钮文本
            if download_btn:
                download_btn.setText(self.tr("重新下载"))
                download_btn.setEnabled(True)

            # 更新主设置对话框的模型选择
            if self.setting_widget:
                try:
                    # 保存当前值并清空
                    current_value = cfg.pywhisper_model.value
                    combo = self.setting_widget.model_card.comboBox
                    combo.clear()

                    # 找出已下载的模型
                    available = []
                    model_map = {
                        m["label"].lower(): m["value"] for m in PYWHISPER_CPP_MODELS
                    }
                    for enum_val in PyWhisperModelEnum:
                        if enum_val.value in model_map:
                            if (MODEL_PATH / model_map[enum_val.value]).exists():
                                available.append(enum_val)

                    # 重建下拉框
                    self.setting_widget.model_card.optionToText = {
                        e: e.value for e in available
                    }
                    for enum_val in available:
                        combo.addItem(enum_val.value, userData=enum_val)

                    # 恢复选择
                    if current_value in available:
                        combo.setCurrentText(current_value.value)
                    elif combo.count() > 0:
                        combo.setCurrentIndex(0)
                except Exception as e:
                    logger.error(f"更新模型选择失败: {e}")

            InfoBar.success(
                self.tr("下载成功"),
                self.tr(f"{model['label']} 模型已下载完成"),
                duration=3000,
                parent=self,
            )
            self.progress_bar.hide()
            self.progress_label.hide()

        def _on_model_download_error(error):
            PyWhisperCppDownloadDialog.is_downloading = False
            self._set_all_download_buttons_enabled(True)
            if download_btn:
                download_btn.setEnabled(True)

            InfoBar.error(self.tr("下载失败"), str(error), duration=3000, parent=self)
            self.progress_bar.hide()
            self.progress_label.hide()

        self.model_download_thread = FileDownloadThread(
            model["mirrorLink"], os.path.join(MODEL_PATH, model["value"])
        )
        self.model_download_thread.progress.connect(_on_model_download_progress)
        self.model_download_thread.finished.connect(_on_model_download_finished)
        self.model_download_thread.error.connect(_on_model_download_error)
        self.model_download_thread.start()

    def _set_all_download_buttons_enabled(self, enabled: bool):
        """设置所有下载按钮的启用状态"""
        # 设置所有模型下载按钮
        for row in range(self.model_table.rowCount()):
            button_container = self.model_table.cellWidget(row, 3)
            if button_container:
                download_btn = button_container.findChild(HyperlinkButton)
                if download_btn:
                    download_btn.setEnabled(enabled)

    def _open_model_folder(self):
        """打开模型文件夹"""
        if os.path.exists(MODEL_PATH):
            open_folder(str(MODEL_PATH))


class PyWhisperCppSettingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_signals()

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)

        # 创建单向滚动区域和容器
        self.scrollArea = SingleDirectionScrollArea(orient=Qt.Vertical, parent=self)  # type: ignore
        self.scrollArea.setStyleSheet(
            "QScrollArea{background: transparent; border: none}"
        )

        self.container = QWidget(self)
        self.container.setStyleSheet("QWidget{background: transparent}")
        self.containerLayout = QVBoxLayout(self.container)

        self.setting_group = SettingCardGroup(self.tr("PyWhisper CPP 设置"), self)

        # 模型选择
        self.model_card = ComboBoxSettingCard(
            cfg.pywhisper_model,
            FIF.ROBOT,
            self.tr("模型"),
            self.tr("选择 PyWhisper 模型"),
            [model.value for model in PyWhisperModelEnum],
            self.setting_group,
        )

        # 检查未下载的模型并从下拉框中移除
        for i in range(self.model_card.comboBox.count() - 1, -1, -1):
            model_text = self.model_card.comboBox.itemText(i).lower()
            model_configs = {
                model["label"].lower(): model for model in PYWHISPER_CPP_MODELS
            }
            model_config = model_configs.get(model_text)
            if model_config and (MODEL_PATH / model_config["value"]).exists():
                continue
            self.model_card.comboBox.removeItem(i)

        # CoreML 开关
        self.coreml_card = SwitchSettingCard(
            FIF.SPEED_HIGH,
            self.tr("CoreML 加速"),
            self.tr("启用 CoreML 硬件加速（推荐）"),
            cfg.pywhisper_use_coreml,
            self.setting_group,
        )

        # 线程数设置
        self.threads_card = RangeSettingCard(
            cfg.pywhisper_n_threads,
            FIF.SPEED_OFF,
            self.tr("线程数"),
            self.tr("CPU 线程数（1-16）"),
            self.setting_group,
        )

        # 语言选择
        self.language_card = ComboBoxSettingCard(
            cfg.transcribe_language,
            FIF.LANGUAGE,
            self.tr("源语言"),
            self.tr("音频的源语言"),
            [language.value for language in TranscribeLanguageEnum],
            self.setting_group,
        )

        # 添加模型管理卡片
        self.manage_model_card = HyperlinkCard(
            "",  # 无链接
            self.tr("管理模型"),
            FIF.DOWNLOAD,  # 使用下载图标
            self.tr("模型管理"),
            self.tr("下载或更新 PyWhisper CPP 模型"),
            self.setting_group,  # 添加到设置组
        )

        # 添加 setMaxVisibleItems
        self.language_card.comboBox.setMaxVisibleItems(6)

        # 使用 addSettingCard 添加卡片到组
        self.setting_group.addSettingCard(self.model_card)
        self.setting_group.addSettingCard(self.coreml_card)
        self.setting_group.addSettingCard(self.threads_card)
        self.setting_group.addSettingCard(self.language_card)
        self.setting_group.addSettingCard(self.manage_model_card)

        # VAD设置组
        self.vad_group = SettingCardGroup(self.tr("VAD设置"), self)

        # VAD过滤开关
        self.vad_filter_card = SwitchSettingCard(
            FluentIcon.MICROPHONE,
            self.tr("VAD过滤"),
            self.tr("使用语音活动检测过滤静音段落"),
            cfg.pywhisper_vad_filter,
            self.vad_group,
        )

        # VAD阈值
        self.vad_threshold_card = RangeSettingCard(
            cfg.pywhisper_vad_threshold,
            FluentIcon.SPEED_OFF,
            self.tr("VAD阈值"),
            self.tr("语音检测阈值，越高越严格"),
            self.vad_group,
        )

        # VAD方法
        self.vad_method_card = ComboBoxSettingCard(
            cfg.pywhisper_vad_method,
            FluentIcon.ROBOT,
            self.tr("VAD方法"),
            self.tr("选择VAD检测方法"),
            [method.value for method in VadMethodEnum],
            self.vad_group,
        )

        # VAD最大并发数
        self.vad_max_workers_card = RangeSettingCard(
            cfg.pywhisper_vad_max_workers,
            FluentIcon.SPEED_HIGH,
            self.tr("最大并发数"),
            self.tr("VAD分段并发处理的最大任务数（1-8）"),
            self.vad_group,
        )

        # 添加VAD设置组的卡片
        self.vad_group.addSettingCard(self.vad_filter_card)
        self.vad_group.addSettingCard(self.vad_threshold_card)
        self.vad_group.addSettingCard(self.vad_method_card)
        self.vad_group.addSettingCard(self.vad_max_workers_card)

        # 将设置组添加到容器布局
        self.containerLayout.addWidget(self.setting_group)
        self.containerLayout.addWidget(self.vad_group)
        self.containerLayout.addStretch(1)

        # 设置组件最小宽度
        self.model_card.comboBox.setMinimumWidth(200)
        self.language_card.comboBox.setMinimumWidth(200)
        self.vad_method_card.comboBox.setMinimumWidth(200)

        # 设置滚动区域
        self.scrollArea.setWidget(self.container)
        self.scrollArea.setWidgetResizable(True)

        # 将滚动区域添加到主布局
        self.main_layout.addWidget(self.scrollArea)

    def setup_signals(self):
        self.manage_model_card.linkButton.clicked.connect(self.show_download_dialog)
        self.vad_filter_card.checkedChanged.connect(self._on_vad_filter_changed)

    def _on_vad_filter_changed(self, checked: bool):
        """VAD过滤开关状态改变时的处理"""
        self.vad_threshold_card.setEnabled(checked)
        self.vad_method_card.setEnabled(checked)
        self.vad_max_workers_card.setEnabled(checked)

    def show_download_dialog(self):
        """显示下载对话框"""
        download_dialog = PyWhisperCppDownloadDialog(self.window(), self)
        download_dialog.show()
