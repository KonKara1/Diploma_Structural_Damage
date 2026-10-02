import sys
import os
import json
import datetime
import traceback
import hashlib
import csv
import torch
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget, 
                             QFrame, QFileDialog, QProgressBar, QListWidget, 
                             QListWidgetItem, QAbstractItemView, QScrollArea, 
                             QCheckBox, QComboBox, QFormLayout, QGridLayout, QLineEdit, QDialog, QMessageBox)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from ultralytics import YOLO

#ΡΥΘΜΙΣΗ ΜΟΝΤΕΛΟΥ ΚΑΙ ΒΑΣΗΣ ΔΕΔΟΜΕΝΩΝ
MODEL_PATH = r"C:\Users\Konstantinos\Desktop\Diploma_YOLO\V11\runs\detect\train3\weights\best.pt" #Ο ΚΑΘΕΝΑΣ ΒΑΖΕΙ ΤΟ PATH ΓΙΑ ΤΟ ΔΙΚΟ ΤΟΥ ΜΟΝΤΕΛΟ ΚΑΙ ΒΑΡΥ
HISTORY_FILE = "history.json"
CACHE_FILE = "image_cache.json"

#ΣΥΝΑΡΤΗΣΕΙΣ ΒΑΣΗΣ, ΜΝΗΜΗΣ ΚΑΙ EXIF
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return {}

def save_history(data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f: json.dump(data, f, indent=4, ensure_ascii=False)

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return {}

def save_cache(data):
    with open(CACHE_FILE, "w", encoding="utf-8") as f: json.dump(data, f, indent=4, ensure_ascii=False)

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def safe_float(val):
    try: return float(val)
    except:
        try: return float(val[0]) / float(val[1])
        except: return 0.0

def extract_exif_data(filepath):
    exif_dict = {"altitude": "-", "direction": "-", "latitude": "-", "longitude": "-"}
    try:
        image = Image.open(filepath)
        exif = image._getexif()
        if exif is not None:
            gps_info = {}
            for tag, value in exif.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_info[sub_decoded] = value[t]
            
            if 'GPSAltitude' in gps_info:
                alt = safe_float(gps_info['GPSAltitude'])
                exif_dict["altitude"] = f"{alt:.1f} m"
            
            if 'GPSImgDirection' in gps_info:
                deg = safe_float(gps_info['GPSImgDirection'])
                dirs = ["Βόρεια", "Βορειοανατολικά", "Ανατολικά", "Νοτιοανατολικά", "Νότια", "Νοτιοδυτικά", "Δυτικά", "Βορειοδυτικά"]
                ix = int((deg + 22.5) // 45) % 8
                exif_dict["direction"] = f"{deg:.1f}° ({dirs[ix]})"
                
            if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
                lat_data = gps_info['GPSLatitude']
                lat_ref = gps_info['GPSLatitudeRef']
                lat_deg = safe_float(lat_data[0])
                lat_min = safe_float(lat_data[1])
                lat_sec = safe_float(lat_data[2])
                lat_dd = lat_deg + (lat_min / 60.0) + (lat_sec / 3600.0)
                if lat_ref == 'S': lat_dd = -lat_dd
                exif_dict["latitude"] = f"{lat_dd:.6f}"

            if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
                lon_data = gps_info['GPSLongitude']
                lon_ref = gps_info['GPSLongitudeRef']
                lon_deg = safe_float(lon_data[0])
                lon_min = safe_float(lon_data[1])
                lon_sec = safe_float(lon_data[2])
                lon_dd = lon_deg + (lon_min / 60.0) + (lon_sec / 3600.0)
                if lon_ref == 'W': lon_dd = -lon_dd
                exif_dict["longitude"] = f"{lon_dd:.6f}"

    except Exception:
        pass
    return exif_dict

#GLOBAL ΡΥΘΜΙΣΕΙΣ ΚΑΙ ΜΕΤΑΦΡΑΣΕΙΣ
class Config:
    LANG = "EL"
    SAVE_IMG = True
    USE_GPU = False
    CONF = 0.1

T = {
    "EL": {
        "nav_upload": "Μεταφόρτωση", "nav_dash": "Αναφορά Επιθεώρησης", "nav_settings": "Ρυθμίσεις",
        "up_title": "Ανάλυση Υποδομών", "up_desc": "Επιλέξτε ή σύρετε εικόνες και βίντεο από drone.",
        "up_project": "Αναγνωριστικό Υποδομής (π.χ. Πυλώνας Μ1):",
        "up_drag": "Επιλογή ή Drag & Drop αρχείων", "up_ready": "Έτοιμα προς ανάλυση ({count})",
        "up_btn": "Έναρξη Ανάλυσης", "up_btn_dis": "Αναμονή Αρχείων", "up_clear": "🗑️ Εκκαθάριση Λίστας",
        "up_dup_title": "Διπλότυπο Αρχείο", "up_dup_msg": "Τα παρακάτω αρχεία υπάρχουν ήδη στη λίστα και παραλείφθηκαν:\n\n{name}",
        "dash_title": "Αναφορά Επιθεώρησης", "dash_new": "Νέα Ανάλυση", "dash_results": "Αναλυτικά Αποτελέσματα",
        "dash_export": "Εξαγωγή Δεδομένων",
        "stat_total": "Σύνολο Ευρημάτων", "stat_cracks": "Ρωγμές", 
        "stat_spalling": "Αποφλοιώσεις", "stat_freelime": "Εξανθήματα", "stat_corrosion": "Διαβρώσεις",
        "stat_high": "Υψηλή Σοβαρότητα", "stat_per_img": "Φθορές / Εικόνα", "stat_area": "Ποσοστό Φθοράς (%)",
        "hist_title": "Ιστορικό Εξέλιξης Υποδομής: ",
        "set_title": "Ρυθμίσεις Συστήματος", "set_lang": "Γλώσσα Διεπαφής / Language:",
        "set_save": "Αποθήκευση επεξεργασμένων εικόνων στο δίσκο", "set_gpu": "Χρήση GPU (CUDA) για ταχύτερη ανάλυση",
        "set_conf": "Κατώφλι Εμπιστοσύνης (Confidence):",
        "conf_0": "0.10 (Υψηλή Ευαισθησία)", "conf_1": "0.25 (Ισορροπημένο)", "conf_2": "0.50 (Υψηλή Ακρίβεια)",
        "proc_title": "Επεξεργασία AI...", "proc_init": "Προετοιμασία...",
        "dmg_none": "Η υποδομή φαίνεται υγιής (Δεν εντοπίστηκαν φθορές)", "dmg_found": "Εντοπίστηκαν {count} σημεία ενδιαφέροντος:",
        "sev_high": "Κρίσιμη", "sev_med": "Μέτρια", "sev_low": "Χαμηλή",
        "type": "Τύπος", "sev": "Σοβαρότητα", 
        "crack": "Ρωγμή", "spalling": "Αποφλοίωση", 
        "freelime": "Εξάνθημα", "corrosion": "Διάβρωση", "rebar": "Οπλισμός",
        "rebar_exposure": "Εμφανής Οπλισμός", "rust": "Σκουριά", "dampness": "Υγρασία",
        "efflorescence": "Εξάνθιση", "water_seepage": "Διαρροή Νερού", "delamination": "Αποκόλληση",
        "ctrl_sort": "Ταξινόμηση:", "ctrl_filter": "Φίλτρο:", "ctrl_layout": "Προβολή:",
        "sort_0": "Χρονολογικά (Αρχικά)", 
        "sort_1": "Σοβαρότητα (Φθίνουσα)", "sort_2": "Σοβαρότητα (Αύξουσα)",
        "sort_3": "Πλήθος Φθορών (Φθίνουσα)", "sort_4": "Πλήθος Φθορών (Αύξουσα)",
        "filter_all": "Όλα τα Ευρήματα", 
        "layout_1": "1 Στήλη (Λίστα)", "layout_2": "2 Στήλες (Πλέγμα)"
    },
    "EN": {
        "nav_upload": "Upload Media", "nav_dash": "Inspection Report", "nav_settings": "Settings",
        "up_title": "Structural Analysis", "up_desc": "Upload drone imagery and videos to detect damages.",
        "up_project": "Asset ID (e.g. Pillar M1):",
        "up_drag": "Click here or Drag & Drop files", "up_ready": "Ready for Processing ({count})",
        "up_btn": "Start Analysis", "up_btn_dis": "Waiting for Files", "up_clear": "🗑️ Clear List",
        "up_dup_title": "Duplicate File", "up_dup_msg": "The following files are already in the list and were skipped:\n\n{name}",
        "dash_title": "Inspection Report", "dash_new": "New Analysis", "dash_results": "Detailed Image Reports",
        "dash_export": "Export Data",
        "stat_total": "Total Findings", "stat_cracks": "Cracks",
        "stat_spalling": "Spalling", "stat_freelime": "Efflorescence", "stat_corrosion": "Corrosion",
        "stat_high": "High Severity", "stat_per_img": "Damage / Image", "stat_area": "Damage Area (%)",
        "hist_title": "Asset Evolution History: ",
        "set_title": "System Settings", "set_lang": "Language (Γλώσσα):",
        "set_save": "Save processed images to disk", "set_gpu": "Use GPU (CUDA) acceleration",
        "set_conf": "Confidence Threshold:",
        "conf_0": "0.10 (High Sensitivity)", "conf_1": "0.25 (Balanced)", "conf_2": "0.50 (High Accuracy)",
        "proc_title": "AI Processing...", "proc_init": "Initializing...",
        "dmg_none": "Infrastructure appears healthy (No damage detected)", "dmg_found": "{count} areas of interest detected:",
        "sev_high": "Critical", "sev_med": "Medium", "sev_low": "Low",
        "type": "Type", "sev": "Severity", 
        "crack": "Crack", "spalling": "Spalling", 
        "freelime": "Freelime", "corrosion": "Corrosion", "rebar": "Rebar",
        "rebar_exposure": "Rebar Exposure", "rust": "Rust", "dampness": "Dampness",
        "efflorescence": "Efflorescence", "water_seepage": "Water Seepage", "delamination": "Delamination",
        "ctrl_sort": "Sort by:", "ctrl_filter": "Filter:", "ctrl_layout": "Layout:",
        "sort_0": "Chronological", 
        "sort_1": "Severity (High to Low)", "sort_2": "Severity (Low to High)",
        "sort_3": "Finding Count (High to Low)", "sort_4": "Finding Count (Low to High)",
        "filter_all": "All Findings", 
        "layout_1": "1 Column (List)", "layout_2": "2 Columns (Grid)"
    }
}

def tr(key, **kwargs):
    text = T[Config.LANG].get(key, key)
    return text.format(**kwargs) if kwargs else text

#THEME-STYLING
class AppStyle:
    STYLESHEET = """
        QMainWindow { background-color: #f8fafc; }
        QWidget { font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; color: #1e293b; }
        QFrame#Sidebar { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
        QLabel#Logo { font-size: 22px; font-weight: 900; color: #0f172a; padding: 10px; }
        QPushButton#NavBtn { background-color: transparent; border: none; border-radius: 8px; text-align: left; padding: 12px 20px; color: #64748b; font-weight: 600; font-size: 15px; }
        QPushButton#NavBtn:hover { background-color: #f1f5f9; color: #0f172a; }
        QPushButton#NavBtn:checked { background-color: #f8fafc; color: #0f172a; border: 1px solid #cbd5e1; border-left: 4px solid #3b82f6; }
        QPushButton#PrimaryBtn { background-color: #0f172a; color: #ffffff; border-radius: 8px; padding: 12px 24px; font-weight: bold; font-size: 15px; }
        QPushButton#PrimaryBtn:hover { background-color: #334155; }
        QPushButton#PrimaryBtn:disabled { background-color: #94a3b8; color: #e2e8f0; }
        QPushButton#SecondaryBtn { background-color: #f1f5f9; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 24px; font-weight: bold; font-size: 15px; }
        QPushButton#SecondaryBtn:hover { background-color: #e2e8f0; }
        QFrame#UploadZone { background-color: #ffffff; border: 2px dashed #cbd5e1; border-radius: 12px; }
        QFrame#UploadZone:hover { border-color: #3b82f6; background-color: #f8fafc; cursor: pointer; }
        QFrame#Card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; }
        QLineEdit { border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; font-size: 15px; background-color: #ffffff; color: #0f172a; }
        QLineEdit:focus { border: 1px solid #3b82f6; }
        QListWidget { border: 1px solid #cbd5e1; background-color: #ffffff; border-radius: 8px; padding: 5px; outline: none; }
        QListWidget::item { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; margin-bottom: 5px; color: #0f172a; }
        QScrollArea { border: none; background-color: transparent; }
        QScrollArea > QWidget > QWidget { background-color: transparent; }
        QScrollBar:vertical { border: none; background: transparent; width: 10px; margin: 0px; border-radius: 5px; }
        QScrollBar::handle:vertical { background: #cbd5e1; min-height: 40px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: #94a3b8; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        
        QComboBox { background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 12px; font-size: 14px; }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QListView { background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; outline: none; border-radius: 6px; }
        QComboBox QListView::item { background-color: #ffffff; color: #0f172a; padding: 8px; }
        QComboBox QListView::item:hover, QComboBox QListView::item:selected { background-color: #f1f5f9; color: #0f172a; }
        
        QProgressBar { border: 1px solid #e2e8f0; border-radius: 8px; text-align: center; height: 24px; background-color: #f8fafc; color: #0f172a; }
        QProgressBar::chunk { background-color: #3b82f6; border-radius: 7px; }
        
        QMessageBox { background-color: #ffffff; }
        QMessageBox QLabel { color: #0f172a; font-size: 14px; }
        QMessageBox QPushButton { background-color: #0f172a; color: #ffffff; border-radius: 6px; padding: 8px 16px; font-weight: bold; }
        QMessageBox QPushButton:hover { background-color: #334155; }
    """

#CUSTOM UI WIDGETS(LIGHTBOX)
class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

class LightboxDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Λεπτομερής Προβολή")
        self.resize(1000, 800)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background-color: #0f172a;")
        
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            lbl.setPixmap(pixmap.scaled(1000, 800, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
        layout.addWidget(lbl)

#WORKER THREAD
class YoloWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict, list) 

    def __init__(self, files):
        super().__init__()
        self.raw_files = files

    def run(self):
        try:
            expanded_files = []
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            frames_dir = os.path.abspath("video_frames")
            
            for file in self.raw_files:
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    self.status.emit(f"Εξαγωγή καρέ από βίντεο: {os.path.basename(file)}...")
                    cap = cv2.VideoCapture(file)
                    if not cap.isOpened(): continue
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps == 0 or fps != fps: fps = 30 
                    interval = int(fps) 
                    
                    count = 0
                    frame_id = 0
                    vid_name = os.path.splitext(os.path.basename(file))[0]
                    os.makedirs(frames_dir, exist_ok=True)
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        if count % interval == 0:
                            frame_filename = f"{vid_name}_sec{frame_id:03d}.jpg"
                            frame_path = os.path.join(frames_dir, frame_filename)
                            if not os.path.exists(frame_path): 
                                is_success, im_buf = cv2.imencode(".jpg", frame)
                                if is_success: im_buf.tofile(frame_path)
                            expanded_files.append(frame_path)
                            frame_id += 1
                        count += 1
                    cap.release()
                else:
                    expanded_files.append(file)
            
            self.status.emit("Loading YOLOv11 & Cache...")
            model = YOLO(MODEL_PATH)
            device = 0 if (Config.USE_GPU and torch.cuda.is_available()) else 'cpu'
            image_cache = load_cache()
            total_files = len(expanded_files)
            
            if total_files == 0:
                self.finished.emit({}, [])
                return
                
            stats = {"total": 0, "cracks": 0, "spalling": 0, "freelime": 0, "corrosion": 0, "high": 0, "med": 0, "low": 0}
            results_list = []
            total_img_area = 0.0
            total_dmg_area = 0.0
            
            for i, file in enumerate(expanded_files):
                self.status.emit(f"Processing {i+1}/{total_files}...")
                
                file_hash = get_file_hash(file)
                cache_key = f"{file_hash}_{Config.CONF}_{Config.SAVE_IMG}"
                
                if cache_key in image_cache:
                    c_data = image_cache[cache_key]
                    saved_path = c_data["saved_path"]
                    findings = c_data["findings"]
                    img_area = c_data["img_area"]
                    box_area_sum = c_data["box_area_sum"]
                    exif_data = c_data.get("exif_data", {"altitude": "-", "direction": "-"})
                    img_stats = c_data["stats"]
                    img_ratio = c_data.get("dmg_ratio", (box_area_sum / img_area * 100) if img_area > 0 else 0)
                    
                    stats["total"] += img_stats.get("total", 0)
                    stats["cracks"] += img_stats.get("cracks", 0)
                    stats["spalling"] += img_stats.get("spalling", 0)
                    stats["freelime"] += img_stats.get("freelime", 0)
                    stats["corrosion"] += img_stats.get("corrosion", 0)
                    stats["high"] += img_stats.get("high", 0)
                    stats["med"] += img_stats.get("med", 0)
                    stats["low"] += img_stats.get("low", 0)
                    
                    total_img_area += img_area
                    total_dmg_area += box_area_sum
                    
                else:
                    exif_data = extract_exif_data(file)
                    res = model.predict(file, conf=Config.CONF, device=device, verbose=False)
                    
                    if Config.SAVE_IMG:
                        out_dir = os.path.abspath("analyzed_images")
                        os.makedirs(out_dir, exist_ok=True)
                        
                        base, ext = os.path.splitext(os.path.basename(file))
                        if ext.lower() not in ['.jpg', '.jpeg', '.png']: ext = '.jpg'
                        
                        conf_str = str(Config.CONF).replace('.', '')
                        out_name = f"{base}_analyzed_c{conf_str}{ext}"
                        saved_path = os.path.join(out_dir, out_name)
                        
                        annotated_img = res[0].plot()
                        is_success, im_buf = cv2.imencode(ext, annotated_img)
                        if is_success:
                            im_buf.tofile(saved_path)
                        else:
                            saved_path = file
                    else:
                        saved_path = file
                        
                    img_h, img_w = res[0].orig_shape
                    img_area = float(img_h * img_w)
                    total_img_area += img_area
                    findings = [] 
                    box_area_sum = 0.0
                    img_stats = {"total": 0, "cracks": 0, "spalling": 0, "freelime": 0, "corrosion": 0, "high": 0, "med": 0, "low": 0}
                    
                    for box in res[0].boxes:
                        cls_id = int(box.cls[0].item())
                        class_name = model.names[cls_id].lower()
                        stats["total"] += 1; img_stats["total"] += 1
                        
                        if "crack" in class_name or "ρωγμη" in class_name: 
                            stats["cracks"] += 1; img_stats["cracks"] += 1
                        elif "spall" in class_name or "αποφλοιωση" in class_name: 
                            stats["spalling"] += 1; img_stats["spalling"] += 1
                        elif "freelime" in class_name or "efflorescence" in class_name or "εξανθημα" in class_name or "εξάνθιση" in class_name: 
                            stats["freelime"] += 1; img_stats["freelime"] += 1
                        elif "corrosion" in class_name or "rust" in class_name or "διαβρωση" in class_name: 
                            stats["corrosion"] += 1; img_stats["corrosion"] += 1
                        
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_area = float((x2 - x1) * (y2 - y1))
                        box_area_sum += box_area
                        total_dmg_area += box_area
                        
                        ratio = float(box_area / img_area)
                        if ratio > 0.05:
                            severity = "sev_high"; stats["high"] += 1; img_stats["high"] += 1
                        elif ratio > 0.01:
                            severity = "sev_med"; stats["med"] += 1; img_stats["med"] += 1
                        else:
                            severity = "sev_low"; stats["low"] += 1; img_stats["low"] += 1
                            
                        findings.append({"type": class_name, "severity": severity})

                    img_ratio = (box_area_sum / img_area * 100) if img_area > 0 else 0.0
                    image_cache[cache_key] = {
                        "saved_path": saved_path,
                        "findings": findings,
                        "img_area": img_area,
                        "box_area_sum": box_area_sum,
                        "stats": img_stats,
                        "exif_data": exif_data,
                        "dmg_ratio": img_ratio
                    }

                results_list.append({
                    "original_idx": i,
                    "original_file": os.path.basename(file),
                    "saved_path": saved_path,
                    "findings": findings,
                    "exif_data": exif_data,
                    "finding_count": img_stats.get("total", 0),
                    "dmg_ratio": img_ratio,
                    "img_stats": img_stats
                })
                            
                self.progress.emit(int(((i + 1) / total_files) * 100))
            
            save_cache(image_cache)
            stats["dmg_per_img"] = round(stats["total"] / total_files, 2) if total_files > 0 else 0
            stats["dmg_ratio"] = round(float(total_dmg_area / total_img_area * 100), 2) if total_img_area > 0 else 0.0
                
            self.status.emit("Completed!")
            self.finished.emit(stats, results_list)
            
        except Exception as e:
            traceback.print_exc()
            self.status.emit(f"Error: {str(e)}")
            self.finished.emit({}, [])

#VIEWS ΚΑΙ DASHBOARD
class Sidebar(QFrame):
    def __init__(self, callback):
        super().__init__()
        self.setObjectName("Sidebar")
        self.setFixedWidth(260)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 24, 16, 24)
        self.logo = QLabel("Structural\nDamage AI", objectName="Logo")
        layout.addWidget(self.logo)
        layout.addSpacing(20)
        
        self.btn1 = QPushButton()
        self.btn2 = QPushButton()
        self.btn3 = QPushButton()
        
        for idx, btn in enumerate([self.btn1, self.btn2, self.btn3]):
            btn.setObjectName("NavBtn")
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, i=idx: callback(i))
            layout.addWidget(btn)
            
        layout.addStretch()
        self.btn1.setChecked(True)
        self.update_texts()

    def update_texts(self):
        self.btn1.setText(tr("nav_upload"))
        self.btn2.setText(tr("nav_dash"))
        self.btn3.setText(tr("nav_settings"))

class ResultImageCard(QFrame):
    def __init__(self, data, is_grid=False):
        super().__init__()
        self.setObjectName("Card")
        self.image_path = data['saved_path']
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(20)

        self.img_label = ClickableLabel()
        self.img_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.img_label.setStyleSheet("background-color: #e2e8f0; border-radius: 8px;")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.clicked.connect(self.open_lightbox)
        
        img_width = 300 if is_grid else 400
        img_height = 225 if is_grid else 300
        self.img_label.setFixedSize(img_width, img_height)
        
        if os.path.exists(self.image_path):
            pixmap = QPixmap(self.image_path)
            self.img_label.setPixmap(pixmap.scaled(img_width, img_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        info_layout = QVBoxLayout()
        info_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        info_layout.setSpacing(8)
        
        title = QLabel(data['original_file'])
        title.setStyleSheet("font-size: 18px; font-weight: 800; color: #0f172a;")
        info_layout.addWidget(title)
        
        exif = data.get('exif_data', {})
        exif_lbl = QLabel(f"📍 Ύψος: <b>{exif.get('altitude', '-')}</b> | 🧭 Κατ: <b>{exif.get('direction', '-')}</b> | 📊 Φθορά: <b>{data.get('dmg_ratio', 0):.1f}%</b>")
        exif_lbl.setStyleSheet("color: #475569; font-size: 13px; margin-bottom: 5px;")
        info_layout.addWidget(exif_lbl)
        
        if not data['findings']:
            no_dmg = QLabel(tr("dmg_none"))
            no_dmg.setStyleSheet("color: #059669; font-weight: bold; font-size: 14px; margin-top: 5px;")
            no_dmg.setWordWrap(True)
            info_layout.addWidget(no_dmg)
        else:
            dmg_title = QLabel(tr("dmg_found", count=len(data['findings'])))
            dmg_title.setStyleSheet("color: #dc2626; font-weight: bold; font-size: 14px; margin-top: 5px;")
            info_layout.addWidget(dmg_title)
            
            display_limit = 4 if is_grid else 6
            for f in data['findings'][:display_limit]: 
                f_lbl = QLabel(f"• {tr('type')}: <b>{tr(f['type']) if f['type'] in T[Config.LANG] else f['type'].capitalize()}</b> | {tr('sev')}: <b>{tr(f['severity'])}</b>")
                f_lbl.setStyleSheet("color: #334155; font-size: 14px; padding-left: 10px;")
                f_lbl.setWordWrap(True)
                info_layout.addWidget(f_lbl)
            
            if len(data['findings']) > display_limit:
                more = QLabel(f"...+ {len(data['findings']) - display_limit} ευρήματα.")
                more.setStyleSheet("color: #94a3b8; font-style: italic; font-size: 13px;")
                info_layout.addWidget(more)

        layout.addWidget(self.img_label)
        layout.addLayout(info_layout)
        layout.addStretch()

    def open_lightbox(self):
        dialog = LightboxDialog(self.image_path, self)
        dialog.exec()

class UploadView(QWidget):
    def __init__(self, on_start_processing):
        super().__init__()
        self.on_start_processing = on_start_processing
        self.files = [] 
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)
        
        self.title = QLabel(styleSheet="font-size: 32px; font-weight: 900; color: #0f172a;")
        self.desc = QLabel(styleSheet="color: #64748b; font-size: 17px;")
        
        self.lbl_project = QLabel(styleSheet="font-weight: 700; font-size: 15px; margin-top: 10px;")
        self.input_project = QLineEdit()
        self.input_project.setPlaceholderText("π.χ. Pillar_1")
        
        main_layout.addWidget(self.title)
        main_layout.addWidget(self.desc)
        main_layout.addWidget(self.lbl_project)
        main_layout.addWidget(self.input_project)
        
        self.drop_zone = QFrame(objectName="UploadZone")
        self.drop_zone.setMinimumHeight(220)
        self.drop_zone.setAcceptDrops(True)
        self.drop_zone.mousePressEvent = lambda e: self.browse_files() if e.button() == Qt.MouseButton.LeftButton else None
        
        dz_layout = QVBoxLayout(self.drop_zone)
        self.dz_icon = QLabel("+", styleSheet="font-size: 50px; font-weight: 300; color: #94a3b8;")
        self.dz_text = QLabel(styleSheet="font-size: 20px; font-weight: 600; color: #334155;")
        dz_layout.addWidget(self.dz_icon, alignment=Qt.AlignmentFlag.AlignCenter)
        dz_layout.addWidget(self.dz_text, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.drop_zone)
        
        self.list_label = QLabel(styleSheet="font-weight: 700; font-size: 16px; margin-top: 15px;")
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        
        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton(objectName="SecondaryBtn")
        self.clear_btn.setFixedSize(200, 50)
        self.clear_btn.clicked.connect(self.clear_all_files)
        
        self.start_btn = QPushButton(objectName="PrimaryBtn")
        self.start_btn.setFixedSize(220, 50)
        self.start_btn.clicked.connect(lambda: self.on_start_processing(self.files, self.input_project.text().strip())) 
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.start_btn)
        
        main_layout.addWidget(self.list_label)
        main_layout.addWidget(self.file_list)
        main_layout.addLayout(btn_layout)
        self.update_texts()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()
    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls() if os.path.exists(url.toLocalFile())]
        if files: self.add_files(files)
    def browse_files(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, "Select Media", "", "Media (*.png *.jpg *.jpeg *.mp4 *.avi *.mov *.mkv)")
        if fnames: self.add_files(fnames)

    def add_files(self, file_paths):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        duplicates = []
        existing_names = [os.path.basename(f) for f in self.files]
        
        for path in file_paths:
            fname = os.path.basename(path)
            if fname in existing_names:
                duplicates.append(fname)
                continue
                
            self.files.append(path)
            existing_names.append(fname)
            
            item = QListWidgetItem(self.file_list)
            item.setSizeHint(QSize(100, 50))
            widget = QWidget()
            widget.setStyleSheet("background-color: transparent; color: #0f172a;")
            l = QHBoxLayout(widget)
            l.setContentsMargins(10, 0, 10, 0)
            
            icon = "🎬" if os.path.splitext(path)[1].lower() in video_extensions else "📄"
            lbl = QLabel(f"<b>{icon} {fname}</b>")
            lbl.setStyleSheet("font-size:15px; color: #0f172a;")
            
            del_btn = QPushButton("❌")
            del_btn.setFixedSize(30, 30)
            del_btn.setStyleSheet("background: transparent; border: none; font-size: 14px;")
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.clicked.connect(lambda checked, p=path, i=item: self.remove_file(p, i))
            
            l.addWidget(lbl)
            l.addStretch()
            l.addWidget(del_btn)
            self.file_list.setItemWidget(item, widget)
            
        if duplicates:
            msg = "\n".join(duplicates[:10])
            if len(duplicates) > 10: msg += f"\n...και άλλα {len(duplicates)-10} αρχεία."
            QMessageBox.warning(self, tr("up_dup_title"), tr("up_dup_msg", name=msg))
            
        self.update_texts()

    def remove_file(self, path, item):
        if path in self.files: self.files.remove(path)
        row = self.file_list.row(item)
        self.file_list.takeItem(row)
        self.update_texts()

    def clear_all_files(self):
        self.files.clear()
        self.file_list.clear()
        self.update_texts()

    def update_texts(self):
        self.title.setText(tr("up_title"))
        self.desc.setText(tr("up_desc"))
        self.lbl_project.setText(tr("up_project"))
        self.dz_text.setText(tr("up_drag"))
        self.list_label.setText(tr("up_ready", count=len(self.files)))
        self.start_btn.setText(tr("up_btn") if self.files else tr("up_btn_dis"))
        self.start_btn.setEnabled(bool(self.files))
        self.clear_btn.setText(tr("up_clear"))
        self.clear_btn.setEnabled(bool(self.files))

class ProcessingView(QWidget):
    def __init__(self, on_complete):
        super().__init__()
        self.on_complete = on_complete
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.card = QFrame(objectName="Card")
        self.card.setFixedSize(500, 280)
        card_layout = QVBoxLayout(self.card)
        card_layout.setSpacing(25)
        
        self.title = QLabel(styleSheet="font-size: 24px; font-weight: 900; color: #0f172a;")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.desc = QLabel(styleSheet="color: #64748b; font-size: 15px;")
        self.desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        card_layout.addStretch()
        card_layout.addWidget(self.title)
        card_layout.addWidget(self.progress_bar)
        card_layout.addWidget(self.desc)
        card_layout.addStretch()
        layout.addWidget(self.card)

    def start_processing(self, files, project_name):
        self.project_name = project_name
        self.title.setText(tr("proc_title"))
        self.desc.setText(tr("proc_init"))
        self.progress_bar.setValue(0)
        self.worker = YoloWorker(files)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.desc.setText)
        self.worker.finished.connect(lambda stats, res: self.on_complete(stats, res, self.project_name))
        self.worker.start()

class Dashboard(QWidget):
    def __init__(self, on_new_analysis):
        super().__init__()
        self.raw_results = []
        self.filtered_results = []
        self.current_project = ""
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(25)
        
        top_bar = QHBoxLayout()
        self.title = QLabel(styleSheet="font-size: 32px; font-weight: 900; color: #0f172a;")
        self.reset_btn = QPushButton(objectName="PrimaryBtn")
        self.reset_btn.clicked.connect(on_new_analysis)
        top_bar.addWidget(self.title)
        top_bar.addStretch()
        top_bar.addWidget(self.reset_btn)
        
        self.hist_label = QLabel()
        self.hist_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #3b82f6; background: #eff6ff; padding: 10px; border-radius: 8px;")
        self.hist_label.hide()
        
        # 8 Cards in a 2x4 Grid with strictly professional monochromatic/corporate colors
        self.stats_grid = QGridLayout()
        self.stats_grid.setSpacing(15)
        
        # Σειρά 1: Γενικά
        self.lbl_total = self.create_stat_card("stat_total", "0", "#1e293b", 0, 0)
        self.lbl_area = self.create_stat_card("stat_area", "0%", "#2563eb", 0, 1)
        self.lbl_high = self.create_stat_card("stat_high", "0", "#dc2626", 0, 2)
        self.lbl_per_img = self.create_stat_card("stat_per_img", "0.0", "#2563eb", 0, 3)
        
        # Σειρά 2: Κατηγορίες Φθορών (Σκούρο Μπλε-Γκρι Slate για όλα, αποφεύγοντας το ουράνιο τόξο)
        self.lbl_cracks = self.create_stat_card("stat_cracks", "0", "#334155", 1, 0)
        self.lbl_spalling = self.create_stat_card("stat_spalling", "0", "#334155", 1, 1)
        self.lbl_freelime = self.create_stat_card("stat_freelime", "0", "#334155", 1, 2)
        self.lbl_corrosion = self.create_stat_card("stat_corrosion", "0", "#334155", 1, 3)
        
        control_bar = QHBoxLayout()
        self.results_title = QLabel(styleSheet="font-size: 22px; font-weight: bold; color: #0f172a;")
        control_bar.addWidget(self.results_title)
        control_bar.addStretch()
        
        self.export_btn = QPushButton(objectName="SecondaryBtn")
        self.export_btn.clicked.connect(self.export_csv)
        
        self.lbl_sort = QLabel(styleSheet="font-weight: bold; color: #475569;")
        self.combo_sort = QComboBox()
        self.combo_sort.currentIndexChanged.connect(self.refresh_view)
        
        self.lbl_filter = QLabel(styleSheet="font-weight: bold; color: #475569;")
        self.combo_filter = QComboBox()
        self.combo_filter.currentIndexChanged.connect(self.refresh_view)
        
        self.lbl_layout = QLabel(styleSheet="font-weight: bold; color: #475569;")
        self.combo_layout = QComboBox()
        self.combo_layout.currentIndexChanged.connect(self.refresh_view)
        
        control_bar.addWidget(self.export_btn)
        control_bar.addSpacing(15)
        control_bar.addWidget(self.lbl_filter)
        control_bar.addWidget(self.combo_filter)
        control_bar.addSpacing(15)
        control_bar.addWidget(self.lbl_sort)
        control_bar.addWidget(self.combo_sort)
        control_bar.addSpacing(15)
        control_bar.addWidget(self.lbl_layout)
        control_bar.addWidget(self.combo_layout)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.results_layout = QGridLayout(self.scroll_content)
        self.results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results_layout.setSpacing(25)
        self.results_layout.setContentsMargins(0, 0, 15, 0)
        self.scroll_area.setWidget(self.scroll_content)
        
        layout.addLayout(top_bar)
        layout.addWidget(self.hist_label)
        layout.addLayout(self.stats_grid)
        layout.addSpacing(15)
        layout.addLayout(control_bar)
        layout.addWidget(self.scroll_area, 1)
        self.update_texts()

    def create_stat_card(self, key, val, color, row, col):
        card = QFrame(objectName="Card")
        card.setFixedSize(220, 100)
        l = QVBoxLayout(card)
        lbl_t = QLabel(tr(key))
        lbl_t.setProperty("trans_key", key) 
        lbl_t.setStyleSheet("color: #64748b; font-weight: 700; font-size: 14px;")
        lbl_v = QLabel(val)
        lbl_v.setStyleSheet(f"font-size: 34px; font-weight: 700; color: {color};")
        l.addWidget(lbl_t)
        l.addWidget(lbl_v)
        self.stats_grid.addWidget(card, row, col)
        return lbl_v

    def update_dashboard(self, stats, results_list, project_name):
        self.raw_results = results_list
        self.current_project = project_name
        
        unique_types = set()
        for res in results_list:
            for f in res['findings']: unique_types.add(f['type'])
        
        self.combo_filter.blockSignals(True)
        self.combo_filter.clear()
        self.combo_filter.addItem(tr("filter_all"), "ALL")
        for t in sorted(unique_types):
            self.combo_filter.addItem(tr(t) if t in T[Config.LANG] else t.capitalize(), t)
        self.combo_filter.blockSignals(False)
        
        if project_name:
            self.hist_label.hide()
            history = load_history()
            date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            current_run = {"date": date_str, "dmg_per_img": stats.get("dmg_per_img", 0), "dmg_ratio": stats.get("dmg_ratio", 0)}
            
            if project_name in history and len(history[project_name]) > 0:
                last_run = history[project_name][-1]
                trend = "Επιδείνωση" if current_run["dmg_ratio"] > last_run["dmg_ratio"] else "Σταθερότητα/Βελτίωση"
                msg = f"Ιστορικό: <b>{project_name}</b> | Προηγούμενη Ανάλυση ({last_run['date']}): {last_run['dmg_ratio']}% Φθορά | Τρέχουσα: {current_run['dmg_ratio']}% ({trend})"
                self.hist_label.setText(msg)
                self.hist_label.show()
            
            if project_name not in history: history[project_name] = []
            history[project_name].append(current_run)
            save_history(history)
            
        self.refresh_view()

    def refresh_view(self):
        filter_val = self.combo_filter.currentData()
        self.filtered_results = []
        dyn_stats = {"total": 0, "cracks": 0, "spalling": 0, "freelime": 0, "corrosion": 0, "high": 0, "med": 0, "low": 0}
        total_img_ratio_sum = 0.0
        
        for res in self.raw_results:
            if filter_val == "ALL" or any(f['type'] == filter_val for f in res['findings']):
                self.filtered_results.append(res)
                img_stats = res.get('img_stats', {})
                dyn_stats["total"] += img_stats.get("total", 0)
                dyn_stats["cracks"] += img_stats.get("cracks", 0)
                dyn_stats["spalling"] += img_stats.get("spalling", 0)
                dyn_stats["freelime"] += img_stats.get("freelime", 0)
                dyn_stats["corrosion"] += img_stats.get("corrosion", 0)
                dyn_stats["high"] += img_stats.get("high", 0)
                dyn_stats["med"] += img_stats.get("med", 0)
                dyn_stats["low"] += img_stats.get("low", 0)
                total_img_ratio_sum += res.get("dmg_ratio", 0)

        visible_count = len(self.filtered_results)
        self.lbl_total.setText(str(dyn_stats["total"]))
        self.lbl_cracks.setText(str(dyn_stats["cracks"]))
        self.lbl_spalling.setText(str(dyn_stats["spalling"]))
        self.lbl_freelime.setText(str(dyn_stats["freelime"]))
        self.lbl_corrosion.setText(str(dyn_stats["corrosion"]))
        self.lbl_high.setText(str(dyn_stats["high"]))
        
        self.lbl_per_img.setText(str(round(dyn_stats["total"] / visible_count, 2) if visible_count > 0 else 0))
        self.lbl_area.setText(f"{round(total_img_ratio_sum / visible_count, 2) if visible_count > 0 else 0.0}%")

        sort_val = self.combo_sort.currentIndex()
        if sort_val == 1: self.filtered_results.sort(key=lambda x: x.get('dmg_ratio', 0), reverse=True)
        elif sort_val == 2: self.filtered_results.sort(key=lambda x: x.get('dmg_ratio', 0), reverse=False)
        elif sort_val == 3: self.filtered_results.sort(key=lambda x: x.get('finding_count', 0), reverse=True)
        elif sort_val == 4: self.filtered_results.sort(key=lambda x: x.get('finding_count', 0), reverse=False)
        else: self.filtered_results.sort(key=lambda x: x.get('original_idx', 0))

        for i in reversed(range(self.results_layout.count())): 
            widget = self.results_layout.itemAt(i).widget()
            if widget: widget.setParent(None)
            
        col_count = 2 if self.combo_layout.currentIndex() == 1 else 1
        row, col = 0, 0
        for data in self.filtered_results:
            card = ResultImageCard(data, is_grid=(col_count == 2))
            self.results_layout.addWidget(card, row, col)
            col += 1
            if col >= col_count:
                col = 0; row += 1

    def export_csv(self):
        if not self.filtered_results: return
        path, _ = QFileDialog.getSaveFileName(self, "Αποθήκευση Δεδομένων", f"Analysis_Export_{datetime.datetime.now().strftime('%Y%m%d')}.csv", "CSV Files (*.csv)")
        if path:
            with open(path, 'w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file)
                header = [
                    "Όνομα Αρχείου", "Συνολικές Φθορές", "Ποσοστό Φθοράς (%)", 
                    "Υψόμετρο", "Κατεύθυνση (Μοίρες)", 
                    "Ρωγμές", "Αποφλοιώσεις", "Εξανθήματα", "Διαβρώσεις",
                    "Κρίσιμη Σοβαρότητα", "Μέτρια Σοβαρότητα", "Χαμηλή Σοβαρότητα",
                    "Αναλυτικά Ευρήματα"
                ]
                writer.writerow(header)
                
                for res in self.filtered_results:
                    stats = res.get('img_stats', {})
                    
                    details = []
                    for f in res['findings']:
                        translated_type = tr(f['type']) if f['type'] in T[Config.LANG] else f['type'].capitalize()
                        translated_sev = tr(f['severity'])
                        details.append(f"{translated_type} ({translated_sev})")
                    details_str = " | ".join(details) if details else "Κανένα Εύρημα"

                    writer.writerow([
                        res['original_file'],
                        stats.get('total', 0),
                        round(res.get('dmg_ratio', 0), 2),
                        res['exif_data'].get('altitude', '-'),
                        res['exif_data'].get('direction', '-'),
                        stats.get('cracks', 0),
                        stats.get('spalling', 0),
                        stats.get('freelime', 0),
                        stats.get('corrosion', 0),
                        stats.get('high', 0),
                        stats.get('med', 0),
                        stats.get('low', 0),
                        details_str
                    ])

    def update_texts(self):
        self.title.setText(tr("dash_title"))
        self.reset_btn.setText(tr("dash_new"))
        self.results_title.setText(tr("dash_results"))
        self.export_btn.setText(tr("dash_export"))
        self.lbl_sort.setText(tr("ctrl_sort"))
        self.lbl_filter.setText(tr("ctrl_filter"))
        self.lbl_layout.setText(tr("ctrl_layout"))
        
        for cb, items in [
            (self.combo_sort, [tr("sort_0"), tr("sort_1"), tr("sort_2"), tr("sort_3"), tr("sort_4")]),
            (self.combo_layout, [tr("layout_1"), tr("layout_2")])
        ]:
            idx = cb.currentIndex()
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(items)
            cb.setCurrentIndex(idx if idx >= 0 else 0)
            cb.blockSignals(False)

        for i in range(self.stats_grid.count()):
            card = self.stats_grid.itemAt(i).widget()
            if card:
                title_lbl = card.findChildren(QLabel)[0]
                key = title_lbl.property("trans_key")
                if key: title_lbl.setText(tr(key))
                
        if self.combo_filter.count() > 0:
            self.combo_filter.blockSignals(True)
            self.combo_filter.setItemText(0, tr("filter_all"))
            for i in range(1, self.combo_filter.count()):
                t = self.combo_filter.itemData(i)
                self.combo_filter.setItemText(i, tr(t) if t in T[Config.LANG] else t.capitalize())
            self.combo_filter.blockSignals(False)

class SettingsView(QWidget):
    language_changed = pyqtSignal()
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        self.title = QLabel(styleSheet="font-size: 32px; font-weight: 900; color: #0f172a;")
        card = QFrame(objectName="Card")
        self.form_layout = QFormLayout(card)
        self.form_layout.setContentsMargins(40, 40, 40, 40)
        self.form_layout.setSpacing(25)
        
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(["Ελληνικά", "English"])
        self.combo_lang.setFixedSize(250, 40)
        self.combo_lang.currentIndexChanged.connect(self.change_lang)
        self.chk_save = QCheckBox()
        self.chk_save.setChecked(Config.SAVE_IMG)
        self.chk_save.toggled.connect(self.toggle_save)
        self.chk_gpu = QCheckBox()
        self.chk_gpu.setChecked(Config.USE_GPU)
        self.chk_gpu.toggled.connect(self.toggle_gpu)
        self.combo_conf = QComboBox()
        self.combo_conf.addItems([tr("conf_0"), tr("conf_1"), tr("conf_2")])
        self.combo_conf.setFixedSize(250, 40)
        self.combo_conf.currentIndexChanged.connect(self.change_conf)
        self.lbl_lang = QLabel(styleSheet="font-weight: 600;")
        self.lbl_conf = QLabel(styleSheet="font-weight: 600;")
        
        self.form_layout.addRow(self.lbl_lang, self.combo_lang)
        self.form_layout.addRow(self.chk_save)
        self.form_layout.addRow(self.chk_gpu)
        self.form_layout.addRow(self.lbl_conf, self.combo_conf)
        
        layout.addWidget(self.title)
        layout.addWidget(card)
        layout.addStretch()
        self.update_texts()

    def change_lang(self, idx):
        Config.LANG = "EL" if idx == 0 else "EN"
        self.language_changed.emit()

    def toggle_save(self, state): Config.SAVE_IMG = state
    def toggle_gpu(self, state): Config.USE_GPU = state
    def change_conf(self, idx): Config.CONF = 0.10 if idx == 0 else (0.25 if idx == 1 else 0.50)

    def update_texts(self):
        self.title.setText(tr("set_title"))
        self.lbl_lang.setText(tr("set_lang"))
        self.chk_save.setText(tr("set_save"))
        self.chk_gpu.setText(tr("set_gpu"))
        self.lbl_conf.setText(tr("set_conf"))
        
        idx = self.combo_conf.currentIndex()
        self.combo_conf.blockSignals(True)
        self.combo_conf.clear()
        self.combo_conf.addItems([tr("conf_0"), tr("conf_1"), tr("conf_2")])
        self.combo_conf.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_conf.blockSignals(False)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Structural Damage AI")
        self.resize(1400, 900)
        self.setStyleSheet(AppStyle.STYLESHEET)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.stack = QStackedWidget()
        self.sidebar = Sidebar(lambda idx: self.stack.setCurrentIndex(idx))
        self.upload_page = UploadView(self.start_processing)
        self.dashboard_page = Dashboard(self.reset_to_upload)
        self.settings_page = SettingsView()
        self.processing_page = ProcessingView(self.show_dashboard)
        self.settings_page.language_changed.connect(self.refresh_all_texts)
        
        self.stack.addWidget(self.upload_page)
        self.stack.addWidget(self.dashboard_page)
        self.stack.addWidget(self.settings_page)
        self.stack.addWidget(self.processing_page)
        layout.addWidget(self.sidebar)
        layout.addWidget(self.stack)

    def refresh_all_texts(self):
        self.sidebar.update_texts()
        self.upload_page.update_texts()
        self.dashboard_page.update_texts()
        self.settings_page.update_texts()

    def start_processing(self, files, project_name):
        self.stack.setCurrentIndex(3)
        self.processing_page.start_processing(files, project_name)

    def show_dashboard(self, stats, results_list, project_name):
        self.dashboard_page.update_dashboard(stats, results_list, project_name)
        self.stack.setCurrentIndex(1)
        self.sidebar.btn2.setChecked(True)

    def reset_to_upload(self):
        self.stack.setCurrentIndex(0)
        self.sidebar.btn1.setChecked(True)
        self.upload_page.files.clear()
        self.upload_page.file_list.clear()
        self.upload_page.input_project.clear()
        self.upload_page.update_texts()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())