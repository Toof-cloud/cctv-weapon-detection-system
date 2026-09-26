"""A bounded model selector with a regular list popup on every platform."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFrame, QListView, QProxyStyle, QSizePolicy, QStyle,
    QStyleFactory, QStyleOptionComboBox, QStylePainter,
)


class _ListPopupStyle(QProxyStyle):
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.StyleHint.SH_ComboBox_Popup:
            return 0
        return super().styleHint(hint, option, widget, returnData)


class ModelSelector(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Fusion alone can still request a menu-style popup with scroll gutters.
        self._list_style = _ListPopupStyle(QStyleFactory.create("Fusion"))
        self._list_style.setParent(self)
        self.setStyle(self._list_style)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        view = QListView(self)
        view.setFrameShape(QFrame.Shape.NoFrame)
        view.setUniformItemSizes(True)
        view.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setView(view)
        self.setMaxVisibleItems(7)
        view.window().setObjectName("modelPopup")
        self.currentIndexChanged.connect(self._update_tooltip)

    def _update_tooltip(self, index):
        self.setToolTip(self.itemData(index, Qt.ItemDataRole.ToolTipRole) or self.currentText())

    def showPopup(self):
        self.view().setFixedWidth(max(1, self.width() - 2))
        super().showPopup()

    def paintEvent(self, event):
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        edit_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox, option,
            QStyle.SubControl.SC_ComboBoxEditField, self,
        )
        option.currentText = self.fontMetrics().elidedText(
            option.currentText, Qt.TextElideMode.ElideMiddle, max(0, edit_rect.width() - 4),
        )
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)
