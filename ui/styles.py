"""
Smart City Command Center — Light Theme Stylesheet
Professional, robust light UI with sophisticated Saffron/Green accents.
"""

LIGHT_THEME = """
/* ===== GLOBAL ===== */
QMainWindow, QWidget {
    background-color: #F0F2F5;
    color: #1F2937;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}

QMainWindow {
    border: none;
}

/* ===== MENU BAR ===== */
QMenuBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E5E7EB;
    padding: 4px;
    color: #374151;
}
QMenuBar::item:selected {
    background-color: #F3F4F6;
    color: #C2410C;
    border-radius: 4px;
}
QMenu {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item:selected {
    background-color: #FFF7ED;
    color: #C2410C;
    border-radius: 4px;
}

/* ===== PANELS ===== */
QFrame#panel, QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 12px;
}

/* ===== LABELS ===== */
QLabel {
    color: #1F2937;
}
QLabel#title {
    font-size: 15px;
    font-weight: 600;
    color: #C2410C;
    padding: 4px 0;
}
QLabel#subtitle {
    font-size: 11px;
    color: #6B7280;
}
QLabel#alert_critical {
    background-color: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 6px;
    padding: 8px 12px;
    color: #991B1B;
}
QLabel#alert_high {
    background-color: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 6px;
    padding: 8px 12px;
    color: #92400E;
}
QLabel#alert_medium {
    background-color: #FEFCE8;
    border: 1px solid #FEF08A;
    border-radius: 6px;
    padding: 8px 12px;
    color: #854D0E;
}
QLabel#alert_low {
    background-color: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 6px;
    padding: 8px 12px;
    color: #166534;
}
QLabel#risk_high {
    color: #DC2626;
    font-size: 24px;
    font-weight: bold;
}
QLabel#risk_medium {
    color: #D97706;
    font-size: 24px;
    font-weight: bold;
}
QLabel#risk_low {
    color: #15803D;
    font-size: 24px;
    font-weight: bold;
}

/* ===== BUTTONS ===== */
QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 6px;
    padding: 8px 16px;
    color: #374151;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #F9FAFB;
    border-color: #9CA3AF;
}
QPushButton:pressed {
    background-color: #F3F4F6;
}
QPushButton#confirm_btn {
    background-color: #F0FDF4;
    border: 1px solid #22C55E;
    color: #15803D;
    font-weight: 600;
}
QPushButton#confirm_btn:hover {
    background-color: #DCFCE7;
}
QPushButton#reject_btn {
    background-color: #FEF2F2;
    border: 1px solid #EF4444;
    color: #B91C1C;
    font-weight: 600;
}
QPushButton#reject_btn:hover {
    background-color: #FEE2E2;
}
QPushButton#action_btn {
    background-color: #FFF7ED;
    border: 1px solid #F97316;
    color: #C2410C;
    font-weight: 600;
}
QPushButton#action_btn:hover {
    background-color: #FFEDD5;
}

/* ===== TABLE ===== */
QTableWidget {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    gridline-color: #F3F4F6;
    selection-background-color: #FFF7ED;
    selection-color: #C2410C;
    color: #374151;
}
QTableWidget::item {
    padding: 6px;
    border-bottom: 1px solid #F3F4F6;
}
QHeaderView::section {
    background-color: #F9FAFB;
    color: #4B5563;
    border: none;
    border-bottom: 1px solid #E5E7EB;
    padding: 8px;
    font-weight: 600;
}

/* ===== SCROLLBAR ===== */
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    background-color: #F9FAFB;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background-color: #D1D5DB;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #9CA3AF;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* ===== TAB WIDGET ===== */
QTabWidget::pane {
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    background-color: #FFFFFF;
}
QTabBar::tab {
    background-color: #F3F4F6;
    border: 1px solid #E5E7EB;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 16px;
    color: #6B7280;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #C2410C;
    border-top: 2px solid #C2410C;
    font-weight: 600;
}

/* ===== STATUS BAR ===== */
QStatusBar {
    background-color: #FFFFFF;
    border-top: 1px solid #E5E7EB;
    color: #4B5563;
    font-size: 11px;
}

/* ===== SPLITTER ===== */
QSplitter::handle {
    background-color: #E5E7EB;
    width: 4px;
}
QSplitter::handle:hover {
    background-color: #D1D5DB;
}

/* ===== COMBO BOX ===== */
QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 6px;
    padding: 6px 12px;
    color: #1F2937;
}
QComboBox:hover {
    border-color: #9CA3AF;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    selection-background-color: #F3F4F6;
}

/* ===== PROGRESS BAR ===== */
QProgressBar {
    background-color: #E5E7EB;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    border-radius: 4px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #F97316, stop:0.5 #FCD34D, stop:1 #22C55E);
}
"""

