import streamlit as st
import ultralytics
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from PIL import Image
import sqlite3
import pandas as pd
from datetime import datetime
import io
import tempfile
import argparse

TARGET_CLASSES = [45,53,54,55]  # Пицца
DB_FILE = 'pizza.db'
margin = 0.05
ZONE_POLYGON = np.array([
    [0, 0],
    [0.5 - margin, 0],
    [0.5 - margin, 1],
    [0, 1]
])
ZONE_POLYGON_ = np.array([
    [0.5 + margin, 0],
    [1, 0],
    [1, 1],
    [0.5 + margin, 1]
])
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS shifts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  max_items_detected INTEGER,
                  duration_seconds INTEGER)''')
    conn.commit()
    conn.close()


def save_shift(max_count, duration):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO shifts (timestamp, max_items_detected, duration_seconds) VALUES (?, ?, ?)",
        (timestamp, max_count, int(duration)))
    conn.commit()
    conn.close()
    return timestamp


def get_stats():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM shifts ORDER BY id DESC", conn)
    conn.close()
    return df

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

def process_frame(frame: np.ndarray, _) -> np.ndarray:
   results = model(frame, imgsz=1280)[0]
   detections = sv.Detections.from_ultralytics(results)
   detections = detections[detections.class_id == 0]
   zone.trigger(detections=detections)

   box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

   labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _, _ in detections]
   frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
   frame = zone_annotator.annotate(scene=frame)


   return frame

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    st.set_page_config(page_title="Подсчёт пиццы", layout="wide")
    init_db()

    if 'video_pos' not in st.session_state: st.session_state.video_pos = 0
    if 'current_max' not in st.session_state: st.session_state.current_max = 0

    st.title("Подсчёт пиццы")

    menu = st.sidebar.selectbox("Меню", ["Видео-анализ", "Фото-анализ", "Статистика"])
    model = load_model()
    conf_thresh = st.sidebar.slider("Пороговое значение", 0.05, 1.0, 0.15)
    frame_skip = st.sidebar.slider("Пропуск кадров", 1, 10, 2)

    if menu == "Видео-анализ":
        video_file = st.file_uploader("Загрузить видео", type=['mp4', 'mov', 'avi'])

        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())

            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
            with ctrl_col1:
                start_trigger = st.button("▶️ ЗАПУСТИТЬ / ПРОДОЛЖИТЬ", width='stretch')
            with ctrl_col2:
                stop_trigger = st.button("⏹️ ЗАВЕРШИТЬ И СОХРАНИТЬ", width='stretch')
            with ctrl_col3:
                if st.button("⏪ СБРОСИТЬ ВИДЕО", width='stretch'):
                    st.session_state.video_pos = 0
                    st.session_state.current_max = 0
                    st.rerun()

            m_col1, m_col2, m_col3 = st.columns(3)
            curr_metric = m_col1.empty()
            max_metric = m_col2.empty()
            pos_metric = m_col3.empty()

            video_placeholder = st.empty()

            if start_trigger:
                cap = cv2.VideoCapture(tfile.name)
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.video_pos)
                st.session_state.current_max = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.session_state.video_pos = 0
                        st.info("Конец видео.")
                        break

                    curr_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    st.session_state.video_pos = curr_pos
                    H, W = frame.shape[:2]
                    line_zone = sv.LineZone(
                        start=sv.Point(W // 2, 0),
                        end=sv.Point(W // 2, 1)
                    )
                    if curr_pos % frame_skip != 0:
                        continue
                    results = model(frame, classes=TARGET_CLASSES, conf=conf_thresh)
                    count = len(results[0].boxes)
                    st.session_state.last_count = count

                    if count > st.session_state.current_max:
                        st.session_state.current_max = count

                    curr_metric.metric("Сейчас в кадре", count)
                    max_metric.metric("Максимум в кадре", st.session_state.current_max)
                    pos_metric.write(f"Кадр: {curr_pos}")

                    frame_res = results[0].plot()
                    frame_res = cv2.cvtColor(frame_res, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_res, use_container_width=True)

                cap.release()
            if stop_trigger:
                if st.session_state.current_max > 0:
                    ts = save_shift(st.session_state.current_max, 0)
                    st.success(
                        f"Заказ сохранен в {ts}! Итого: {st.session_state.current_max}")
                else:
                    st.warning("Нечего сохранять (0 предметов).")

    elif menu == "Фото-анализ":
        uploaded_file = st.file_uploader("Загрузите фото", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            img = Image.open(uploaded_file)
            if st.button("Проанализировать"):
                results = model(np.array(img), classes=TARGET_CLASSES, conf=conf_thresh)
                count = len(results[0].boxes)
                st.image(results[0].plot(), width='stretch')
                save_order(count, count, 0)
                st.success(f"Найдено и сохранено: {count}")

    elif menu == "Статистика":
        df = get_stats()
        if not df.empty:
            st.metric("Всего пиццы сделано", df['max_items_detected'].sum())
            st.dataframe(df, width='stretch')

            # Экспорт в Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button("📥 Скачать отчет (Excel)", buffer.getvalue(), "coffee_report.xlsx")

            if st.button("🗑 Очистить БД"):
                conn = sqlite3.connect(DB_FILE)
                conn.cursor().execute("DELETE FROM shifts")
                conn.commit()
                conn.close()
                st.rerun()
if __name__ == "__main__":
    main()
