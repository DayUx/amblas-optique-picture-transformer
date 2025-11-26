from __future__ import annotations

import os
import sys
from typing import List, Tuple
import cv2
import numpy as np
import random

# Import AVIF plugin pour Pillow
try:
    import pillow_avif
except ImportError:
    pass

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QProgressBar,
    QTextEdit,
    QCheckBox,
    QSizePolicy,
    QSpacerItem,
)

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".avif"}

def load_image(path: str) -> tuple[np.ndarray, bool]:
    """
    Load an image from path. Uses PIL for AVIF files, cv2 for others.
    Returns tuple: (image in BGRA format, has_transparency)
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".avif":
        # Use PIL to load AVIF, then convert to OpenCV format
        pil_img = Image.open(path)
        has_alpha = pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info)

        # Convert to RGBA
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')

        # Convert PIL Image to numpy array (RGBA)
        img_rgba = np.array(pil_img)
        # Convert RGBA to BGRA for OpenCV
        img_bgra = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA)
        return img_bgra, has_alpha
    else:
        # Use OpenCV for other formats - load with alpha channel
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        # Check if image has alpha channel
        has_alpha = (len(img.shape) == 3 and img.shape[2] == 4)

        # Convert to BGRA if needed
        if not has_alpha:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            # Set alpha to fully opaque
            img[:, :, 3] = 255

        return img, has_alpha



def list_images_in_folder(folder: str) -> List[str]:
    jpg_files = []
    for dirpath, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png") or filename.lower().endswith(".bmp") or filename.lower().endswith(".tif") or filename.lower().endswith(".tiff") or filename.lower().endswith(".webp") or filename.lower().endswith(".avif"):
                full_path = os.path.join(dirpath, filename)
                jpg_files.append(full_path)
    return jpg_files


def parse_rules_from_table(table: QTableWidget) -> List[Tuple[str, float, bool, bool]]:
    rules: List[Tuple[str, float, bool, bool]] = []
    for row in range(table.rowCount()):
        item_key = table.item(row, 0)
        item_percent = table.item(row, 1)
        item_center = table.item(row, 2)
        item_no_shadow = table.item(row, 3)
        key = (item_key.text().strip() if item_key else "")
        percent_str = (item_percent.text().strip() if item_percent else "")
        if not key:
            continue
        if not percent_str:
            continue
        # Supporte "10" ou "10.5" ou "10,5"
        percent_str = percent_str.replace(",", ".")
        try:
            val = float(percent_str)
        except ValueError:
            continue
        if val < 0:
            val = 0.0
        # Limite raisonnable: 0-100 (par côté)
        if val > 100:
            val = 100.0
        center_flag = True
        if item_center is not None:
            try:
                center_flag = (item_center.checkState() == Qt.Checked)
            except Exception:
                center_flag = True
        
        no_shadow_flag = False
        if item_no_shadow is not None:
            try:
                no_shadow_flag = (item_no_shadow.checkState() == Qt.Checked)
            except Exception:
                no_shadow_flag = False

        rules.append((key, val / 100.0, center_flag, no_shadow_flag))
    return rules


class ImageProcessorWorker(QObject):
    progress = pyqtSignal(int, int)  # current, total
    message = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        rules: List[Tuple[str, float, bool, bool]],
        ignore_case: bool,
        bg_images: List[str],
    ):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.rules = rules
        self.ignore_case = ignore_case
        self.bg_images = bg_images

    def _match_key(self, base_name: str) -> tuple[bool, float, str, bool, bool]:
        """
        Retourne (matched, padding_ratio, matched_key, center_flag, no_shadow_flag)
        Le nom de fichier correspond s'il CONTIENT le mot-clé (et non plus s'il commence par).
        """
        name = base_name
        if self.ignore_case:
            name = name.lower()

        for (key, ratio, center_flag, no_shadow_flag) in self.rules:
            k = key.lower() if self.ignore_case else key
            if k and k in name:
                return True, ratio, key, center_flag, no_shadow_flag
        return False, 0.0, "", True, False

    def _transform(self, path, padding_ratio, center: bool = True, no_shadow: bool = False) -> bool:
        """
        Transform image and return True if image had transparency
        """
        # load image with transparency info
        img, has_transparency = load_image(path)

        if has_transparency:
            # Image already has transparency - use existing alpha channel
            result = img.copy()
            mask = result[:, :, 3]
            # Disable shadow for transparent images
            no_shadow = True
        else:
            # No transparency - extract from white background
            # convert to gray
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

            # threshold input image as mask
            mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
            mask = 255 - mask

            # clean up mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # anti-alias mask
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2)
            mask = (2 * mask.astype(np.float32) - 255).clip(0, 255).astype(np.uint8)

            # put alpha into image
            result = img.copy()
            result[:, :, 3] = mask

        fg_h, fg_w, _ = result.shape

        bg_img = cv2.imread(random.choice(self.bg_images))
        bg_h, bg_w, _ = bg_img.shape

        # Resize full image to fit width
        aspect_ratio = result.shape[1] / result.shape[0]
        resized_w = bg_w
        resized_h = int(resized_w / aspect_ratio)
        result_resized = cv2.resize(result, (resized_w, resized_h))

        # Bounding box from alpha
        non_zero = cv2.findNonZero(result_resized[:, :, 3])
        x, y, w, h = cv2.boundingRect(non_zero)

        # Crop object from result_resized
        object_crop = result_resized[y:y + h, x:x + w]
        pad = int(bg_w * padding_ratio)
        usable_w = bg_w - 2 * pad
        obj_ar = w / h
        obj_new_w = usable_w
        obj_new_h = int(obj_new_w / obj_ar)
        fg_resized = cv2.resize(object_crop, (obj_new_w, obj_new_h))

        # Place position
        center_x = pad + (usable_w - obj_new_w) // 2
        if center:
            center_y = int((bg_h - obj_new_h) / 2 + bg_h / 17)
        else:
            bottom_margin = int(bg_h / 20)
            center_y = max(0, bg_h - obj_new_h)

        # Start composite
        bg_img = bg_img.astype(np.float32)
        roi = bg_img[center_y:center_y + obj_new_h, center_x:center_x + obj_new_w].copy()
        fg_float = fg_resized.astype(np.float32)
        alpha = fg_float[:, :, 3] / 255.0
        alpha_3 = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        blended_fg = roi * (1 - alpha_3) + fg_float[:, :, :3] * alpha_3

        # Shadow generation from full image to avoid cutoff
        if not no_shadow:
            shadow_mask = result_resized[:, :, 3]
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(resized_h / 5), int(resized_h / 5)))
            shadow_mask = cv2.erode(shadow_mask, erode_kernel)
            shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=resized_h / 10, sigmaY=resized_h / 10)

            # Calculate scaled Y offset
            scale_factor = obj_new_w / w
            scaled_bbox_h = int(h * scale_factor)
            y_offset = int((resized_h - h) / 2 * scale_factor)
            w_offset = int((resized_w - w) / 2 * scale_factor)

            shadow_new_w = bg_w - 2 * (pad - w_offset)

            # Resize shadow width
            shadow_height = 50
            shadow_mask_resized = cv2.resize(shadow_mask, (shadow_new_w, shadow_height))

            # Calculate Y offset between result_resized and fg_resized
            fg_bbox_in_resized = [x, y, w, h]
            scaled_bbox_w = obj_new_w
            scale_factor = scaled_bbox_w / w
            scaled_bbox_h = int(h * scale_factor)
            y_offset = int((resized_h - h) / 2 * scale_factor)
            shadow_top = int(center_y + scaled_bbox_h - ((shadow_height / 5) + shadow_height * (y_offset / fg_h)))
            shadow_left = pad + (usable_w - shadow_new_w) // 2

        if center and not no_shadow:
            # Place shadow
            shadow_full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
            if 0 <= shadow_top < bg_h - shadow_height:
                shadow_full_mask[shadow_top:shadow_top + shadow_height,
                shadow_left:shadow_left + shadow_new_w] = shadow_mask_resized

            # Blend shadow on background
            shadow_alpha = shadow_full_mask.astype(np.float32) / 255.0
            shadow_alpha_3ch = cv2.cvtColor(shadow_alpha, cv2.COLOR_GRAY2BGR)
            shadow_color = (0, 0, 0)
            shadow_opacity = 0.4
            shadow_layer = np.full_like(bg_img, shadow_color, dtype=np.float32)
            bg_img = bg_img * (1 - shadow_alpha_3ch * shadow_opacity) + shadow_layer * shadow_alpha_3ch * shadow_opacity

        # Blend foreground again
        bg_roi = bg_img[center_y:center_y + obj_new_h, center_x:center_x + obj_new_w]
        fg_over_bg = blended_fg
        bg_img[center_y:center_y + obj_new_h, center_x:center_x + obj_new_w] = (
                bg_img[center_y:center_y + obj_new_h, center_x:center_x + obj_new_w] * (1 - alpha_3) +
                fg_float[:, :, :3] * alpha_3
        )
        if center and not no_shadow:
            # Darken under shadow
            shadow_fg = shadow_alpha[center_y:center_y + obj_new_h, center_x:center_x + obj_new_w] * 0.25
            darken_mask = (shadow_fg * alpha).astype(np.float32)
            darken_mask_3ch = cv2.cvtColor(darken_mask, cv2.COLOR_GRAY2BGR)
            darkness_strength = 0.8
            fg_rgb = fg_float[:, :, :3]
            fg_rgb_darkened = fg_rgb * (1 - darken_mask_3ch * darkness_strength)
            fg_float[:, :, :3] = fg_rgb_darkened
            fg_over_bg = bg_roi * (1 - alpha_3) + fg_float[:, :, :3] * alpha_3
            bg_img[center_y:center_y + obj_new_h, center_x:center_x + obj_new_w] = fg_over_bg

        final_img = bg_img.astype(np.uint8)

        # Get relative path from ./images
        rel_path = os.path.relpath(path, self.input_dir)

        # Change extension to .webp
        rel_path_no_ext = os.path.splitext(rel_path)[0]
        output_file_path = os.path.join(self.output_dir, rel_path_no_ext + ".webp")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(output_file_path)

        cv2.imwrite(output_file_path, final_img, [cv2.IMWRITE_WEBP_QUALITY, 95])

        return has_transparency

    def run(self):
        try:
            if not os.path.isdir(self.input_dir):
                self.message.emit("Dossier d’entrée introuvable.")
                self.finished.emit()
                return

            os.makedirs(self.output_dir, exist_ok=True)

            files = list_images_in_folder(self.input_dir)
            total = len(files)
            if total == 0:
                self.message.emit("Aucune image trouvée dans le dossier d’entrée.")
                self.finished.emit()
                return

            processed = 0
            skipped = 0

            for idx, in_path in enumerate(files, start=1):
                self.progress.emit(idx, total)
                base = os.path.basename(in_path)
                stem, ext = os.path.splitext(base)
                matched, ratio, matched_key, center_flag, no_shadow_flag = self._match_key(stem)
                if not matched:
                    skipped += 1
                    self.message.emit(f"Ignorée (pas de mot-clé) : {base}")
                    continue

                try:
                    # with Image.open(in_path) as img:
                    #     w, h = img.size
                    #     pad_x = int(round(w * ratio))
                    #     pad_y = int(round(h * ratio))
                    #     new_w = w + 2 * pad_x
                    #     new_h = h + 2 * pad_y
                    #
                    #     img, out_mode, bg = self._compute_output_mode_and_bg(img, ext)
                    #     canvas = Image.new(out_mode, (new_w, new_h), bg)
                    #     canvas.paste(img, (pad_x, pad_y))
                    #
                    #     out_path = os.path.join(self.output_dir, base)
                    #     save_kwargs = {}
                    #     if ext.lower() in {".jpg", ".jpeg"}:
                    #         save_kwargs.update({"quality": 95, "subsampling": 1})
                    #     canvas.save(out_path, **save_kwargs)

                    had_transparency = self._transform(in_path, ratio, center_flag, no_shadow_flag)

                    processed += 1
                    transparency_info = " [transparence détectée]" if had_transparency else ""
                    self.message.emit(
                        f"OK [{matched_key} {int(ratio*100)}% - {'center' if center_flag else 'bottom'}]{transparency_info} → {base}"
                    )
                except Exception as e:
                    self.message.emit(f"Erreur sur {base}: {e}")

            self.message.emit(
                f"Terminé. Traitées: {processed}, ignorées: {skipped}, total: {total}."
            )
        finally:
            self.finished.emit()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Amblas Optique Image Converter 3000")
        self.resize(900, 600)

        self.thread: QThread | None = None
        self.worker: ImageProcessorWorker | None = None

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Dossier d’entrée
        row_in = QHBoxLayout()
        row_in.addWidget(QLabel("Dossier d’entrée:"))
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Sélectionnez le dossier contenant les images…")
        btn_in = QPushButton("Parcourir…")
        btn_in.clicked.connect(self._choose_input_dir)
        row_in.addWidget(self.input_edit, stretch=1)
        row_in.addWidget(btn_in)
        root.addLayout(row_in)

        # Dossier de sortie
        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Dossier de sortie:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Sélectionnez le dossier où enregistrer les résultats…")
        btn_out = QPushButton("Parcourir…")
        btn_out.clicked.connect(self._choose_output_dir)
        row_out.addWidget(self.output_edit, stretch=1)
        row_out.addWidget(btn_out)
        root.addLayout(row_out)

        # Fonds multiples (sélection multi-fichiers)
        row_bg = QHBoxLayout()
        row_bg.addWidget(QLabel("Fonds (multiples):"))
        self.bg_edit = QLineEdit()
        self.bg_edit.setReadOnly(True)
        self.bg_edit.setPlaceholderText("Aucun fond sélectionné (optionnel)")
        btn_bg = QPushButton("Choisir…")
        btn_bg.clicked.connect(self._choose_backgrounds)
        row_bg.addWidget(self.bg_edit, stretch=1)
        row_bg.addWidget(btn_bg)
        root.addLayout(row_bg)

        # Options simples
        opts_row = QHBoxLayout()
        self.chk_ignore_case = QCheckBox("Ignorer la casse (mots-clés)")
        self.chk_ignore_case.setChecked(True)
        opts_row.addWidget(self.chk_ignore_case)
        opts_row.addItem(QSpacerItem(20, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))
        root.addLayout(opts_row)

        # Table des règles: Mot-clé / Padding (%) / Centrer
        root.addWidget(QLabel("Règles de padding par mot-clé:"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Mot-clé", "Padding (%)", "Centrer", "Sans Ombre"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        root.addWidget(self.table)

        btns_row = QHBoxLayout()
        btn_add = QPushButton("Ajouter une ligne")
        btn_add.clicked.connect(self._add_rule_row)
        btn_del = QPushButton("Supprimer la sélection")
        btn_del.clicked.connect(self._delete_selected_rows)
        btns_row.addWidget(btn_add)
        btns_row.addWidget(btn_del)
        btns_row.addItem(QSpacerItem(20, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))
        root.addLayout(btns_row)

        # Actions
        actions_row = QHBoxLayout()
        self.btn_start = QPushButton("Démarrer le traitement")
        self.btn_start.clicked.connect(self._start_processing)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        actions_row.addWidget(self.btn_start)
        actions_row.addWidget(self.progress_bar)
        root.addLayout(actions_row)

        # Journal
        root.addWidget(QLabel("Journal:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, stretch=1)

        # Ligne initiale d’exemple
        self._add_rule_row("FRONT", "10")
        self._add_rule_row("BACK", "10")
        self._add_rule_row("CAT", "15")
        self._add_rule_row("SIDE", "15")
        self._add_rule_row("ZOOM", "0",False)
        self._add_rule_row("GUY", "15",False)
        self._add_rule_row("GUYS", "15",False)
        self._add_rule_row("PEOPLE", "15",False)

    def _add_rule_row(self, key: str = "", percent: str = "", center: bool = True, no_shadow: bool = False):
        r = self.table.rowCount()
        self.table.insertRow(r)
        item_pfx = QTableWidgetItem(key)
        item_pct = QTableWidgetItem(percent)
        # Édition directe
        item_pfx.setFlags(item_pfx.flags() | Qt.ItemIsEditable)
        item_pct.setFlags(item_pct.flags() | Qt.ItemIsEditable)
        self.table.setItem(r, 0, item_pfx)
        self.table.setItem(r, 1, item_pct)
        # Colonne Centrer (checkbox)
        item_center = QTableWidgetItem()
        item_center.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        item_center.setCheckState(Qt.Checked if center else Qt.Unchecked)
        self.table.setItem(r, 2, item_center)
        # Colonne Sans Ombre (checkbox)
        item_no_shadow = QTableWidgetItem()
        item_no_shadow.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        item_no_shadow.setCheckState(Qt.Checked if no_shadow else Qt.Unchecked)
        self.table.setItem(r, 3, item_no_shadow)

    def _delete_selected_rows(self):
        selected = sorted(set(idx.row() for idx in self.table.selectedIndexes()), reverse=True)
        for r in selected:
            self.table.removeRow(r)

    def _choose_input_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choisir le dossier d’entrée", "")
        if d:
            self.input_edit.setText(d)

    def _choose_backgrounds(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Choisir un ou plusieurs fonds",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.avif)",
        )
        if files:
            self.background_paths = files
            # Affichage: nombre de fichiers et premier nom
            if len(files) == 1:
                self.bg_edit.setText(os.path.basename(files[0]))
            else:
                self.bg_edit.setText(f"{len(files)} fichiers sélectionnés")
        else:
            self.background_paths = []
            self.bg_edit.clear()

    def _choose_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie", "")
        if d:
            self.output_edit.setText(d)

    def _start_processing(self):
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        bg_files = self.background_paths
        rules = parse_rules_from_table(self.table)
        ignore_case = self.chk_ignore_case.isChecked()
        print(bg_files)

        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Validation", "Veuillez sélectionner un dossier d’entrée valide.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Validation", "Veuillez sélectionner un dossier de sortie.")
            return
        if not rules:
            QMessageBox.warning(self, "Validation", "Veuillez ajouter au moins une règle (mot-clé + padding).")
            return
        if not bg_files:
            QMessageBox.warning(self, "Validation", "Veuillez ajouter au moins une image de fond.")
            return

        self._set_ui_busy(True)
        self.log.clear()
        self.progress_bar.setValue(0)

        self.thread = QThread(self)
        self.worker = ImageProcessorWorker(input_dir, output_dir, rules, ignore_case,bg_files)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.message.connect(self._log)
        self.worker.finished.connect(self._on_finished)

        self.thread.start()

    def _on_progress(self, current: int, total: int):
        if total <= 0:
            self.progress_bar.setValue(0)
            return
        pct = int((current / total) * 100)
        self.progress_bar.setValue(pct)

    def _log(self, text: str):
        self.log.append(text)

    def _on_finished(self):
        self._set_ui_busy(False)
        # Nettoyage du thread/worker
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self.worker = None
        self._log("Traitement terminé.")

    def _set_ui_busy(self, busy: bool):
        self.btn_start.setEnabled(not busy)
        self.input_edit.setEnabled(busy if hasattr(bool, "__call__") else not busy)  # avoid linter quirks
        self.output_edit.setEnabled(not busy)
        self.table.setEnabled(not busy)
        self.chk_ignore_case.setEnabled(not busy)

        # Curseur d’attente
        QApplication.setOverrideCursor(Qt.WaitCursor if busy else Qt.ArrowCursor)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    try:
        sys.exit(app.exec_())
    finally:
        # Restaure le curseur au cas où
        QApplication.setOverrideCursor(Qt.ArrowCursor)


if __name__ == "__main__":
    main()