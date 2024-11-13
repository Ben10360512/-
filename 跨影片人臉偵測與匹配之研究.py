import streamlit as st
import os
import cv2
import face_recognition
import numpy as np
import networkx as nx
from random import shuffle
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd
from io import BytesIO

st.set_page_config(
    page_title="跨影片人臉偵測與匹配之研究",
    layout="wide",
)

st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        color: #4CAF50;
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .main-content {
        max-width: 800px;
        margin: auto;
    }
    .upload-section {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="centered-title">跨影片人臉偵測與匹配之研究</div>', unsafe_allow_html=True)

with st.sidebar:
    if os.path.exists("00.png"):
        st.image('00.png', width=150)

    else:
        st.warning("找不到 images.jpg 圖片，請確認檔案是否存在。")
        
    st.markdown("""
        <div style="text-align: left;">
            <h3>第九組</h3>
            <h3>指導老師：李御璽</h3>
            <h4>組長：蔡宇翔</h4>
            <h4>組員：湯凱翔、陳宥銘、曾宥維</h4>
        </div>
    """, unsafe_allow_html=True)
    
    

    # 新增演算法選擇按鈕
    algorithm_choice = st.selectbox(
        "選擇人臉辨識演算法",
        ["Chinese Whispers", "DBSCAN"]
    )

st.markdown('<div class="main-content">', unsafe_allow_html=True)

def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def _chinese_whispers(encoding_list, threshold=0.37, iterations=20):
    nodes = []
    edges = []

    image_paths, encodings = zip(*encoding_list)

    if len(encodings) <= 1:
        st.warning("Not enough encodings to cluster!")
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        node_id = idx + 1
        node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
        nodes.append(node)

        if (idx + 1) >= len(encodings):
            break

        compare_encodings = encodings[idx + 1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance <= threshold:
                edge_id = idx + i + 2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges += encoding_edges

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    for _ in range(0, iterations):
        cluster_nodes = list(G.nodes())
        shuffle(cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}
            for ne in neighbors:
                if isinstance(ne, int):
                    if G.nodes[ne]['cluster'] in clusters:
                        clusters[G.nodes[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.nodes[ne]['cluster']] = G[node][ne]['weight']
            edge_weight_sum = 0
            max_cluster = 0
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster
            G.nodes[node]['cluster'] = max_cluster

    clusters = {}
    for (_, data) in G.nodes.items():
        cluster = data['cluster']
        path = data['path']
        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)

    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
    return sorted_clusters

def cluster_facial_encodings(facial_encodings, threshold):
    if len(facial_encodings) <= 1:
        st.warning("Number of facial encodings must be greater than one, can't cluster")
        return []
    sorted_clusters = _chinese_whispers(facial_encodings.items(), threshold)
    return sorted_clusters

def extract_faces_from_video(video_path, output_dir, frame_interval=30, consecutive_frame_threshold=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    face_images = []
    last_seen_faces = {}

    progress_bar = st.progress(0)
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for j, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                face_path = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_frame_{i}_face_{j}.jpg")
                cv2.imwrite(face_path, face_image)

                # 將 face_encoding 轉換為可哈希的類型（元組）
                face_encoding_tuple = tuple(face_encoding)

                # 檢查人臉是否已經出現過，並且時間間隔是否超過閾值
                distances = face_distance(list(last_seen_faces.keys()), face_encoding)
                if len(distances) == 0 or min(distances) > 0.6:
                    # 沒有相似人臉，記錄新的出現時間
                    face_images.append((face_path, i))
                    last_seen_faces[face_encoding_tuple] = i
                else:
                    # 找到相似人臉，檢查是否是連續幀出現
                    closest_face_index = np.argmin(distances)
                    last_seen_frame = list(last_seen_faces.values())[closest_face_index]
                    if i - last_seen_frame > consecutive_frame_threshold:
                        # 非連續出現，記錄新的時間
                        face_images.append((face_path, i))
                        last_seen_faces[face_encoding_tuple] = i

        progress_bar.progress((i + 1) / frame_count)
    cap.release()
    return face_images

def compute_facial_encodings(face_image_paths):
    facial_encodings = {}
    for face_image_path in face_image_paths:
        image = face_recognition.load_image_file(face_image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            facial_encodings[face_image_path] = encodings[0]
    return facial_encodings

def create_excel_report(cluster_info, video_info, temp_dir):
    data = []
    for cluster_name, faces in cluster_info.items():
        for face_path in faces:
            video_name, minutes, seconds = video_info.get(face_path, (None, None, None))
            if video_name:
                # 生成直接打開影片的本地檔案鏈接
                video_file_path = os.path.abspath(os.path.join(temp_dir, f"{video_name}.mp4"))
                play_link = f'=HYPERLINK("{video_file_path}", "觀看影片")'
                data.append({
                    '類別': cluster_name,
                    '照片名': os.path.basename(face_path),
                    '所屬影片': video_name,
                    '出現時間': f"{minutes}分{seconds}秒",
                    '播放鏈結': play_link
                })

    df = pd.DataFrame(data)

    # 使用 BytesIO 在記憶體中建立 Excel 檔案
    buffer = BytesIO()

    # 使用 xlsxwriter 創建 Excel 文件
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

        # 獲取 xlsxwriter 的 workbook 和 worksheet 物件
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        # 添加超連結，從第二行開始（index + 2），最後一個欄位是 "播放鏈結"
        for index, link in enumerate(df['播放鏈結']):
            worksheet.write_formula(index + 1, 4, link)

    buffer.seek(0)
    return buffer

def find_best_dbscan_params(encodings, eps_start=0.3, eps_end=0.7, eps_step=0.05, min_samples_range=range(3, 8)):
    best_score = -1
    best_eps = 0.5
    best_min_samples = 5

    eps_values = np.arange(eps_start, eps_end, eps_step)
    for eps_val in eps_values:
        for min_samp in min_samples_range:
            clustering = DBSCAN(eps=eps_val, min_samples=min_samp, metric='euclidean').fit(encodings)
            labels = clustering.labels_
            # 需要至少 2 個群集來計算輪廓係數
            if len(set(labels)) > 1:
                try:
                    score = silhouette_score(encodings, labels)
                    if score > best_score:
                        best_score = score
                        best_eps = eps_val
                        best_min_samples = min_samp
                except:
                    continue

    return best_eps, best_min_samples
import shutil  # 引入 shutil 模組來複製檔案

def save_cluster_results_to_output(sorted_clusters, output_dir, video_info):
    # 創建保存分類結果的總目錄
    clusters_output_dir = os.path.join(output_dir, "clusters")
    if not os.path.exists(clusters_output_dir):
        os.makedirs(clusters_output_dir)

    # 迭代分類結果，為每個分類創建一個目錄，並將圖片複製到目錄中
    for idx, cluster in enumerate(sorted_clusters):
        cluster_dir = os.path.join(clusters_output_dir, f"cluster_{idx}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

        for face_path in cluster:
            face_filename = os.path.basename(face_path)
            destination_path = os.path.join(cluster_dir, face_filename)
            shutil.copy(face_path, destination_path)

        st.write(f"分類 {idx} 的結果已保存到目錄: {cluster_dir}")
def main():
    threshold = st.number_input(
        "輸入希望同類別彼此的相似度 (預設值: 0.37)提示:數值越小相似度越接近",
        min_value=0.0, max_value=2.0, value=0.37, step=0.01
    )
    
    frame_interval = st.number_input(
        "輸入每幾偵擷取一張影片圖片 (預設值: 30偵(約1秒))", min_value=1, value=30, step=1
    )
    frame_interval = frame_interval 
    
    video_file_1 = st.file_uploader("選擇影片1", type=["mp4", "avi", "mov"], key="video_1")
    video_file_2 = st.file_uploader("選擇影片2", type=["mp4", "avi", "mov"], key="video_2")
    output_dir = st.text_input("輸出目錄", "output")

    col1, col2, col3 = st.columns([1, 0.1, 1])

    with col1:
        if st.button("觸發人臉分類", key="classify_button"):
            if video_file_1 is not None and video_file_2 is not None:
                temp_dir = os.path.join(output_dir, "temp")
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                video_path_1 = os.path.join(temp_dir, video_file_1.name)
                video_path_2 = os.path.join(temp_dir, video_file_2.name)
                with open(video_path_1, "wb") as f:
                    f.write(video_file_1.getbuffer())
                with open(video_path_2, "wb") as f:
                    f.write(video_file_2.getbuffer())

                all_face_images = []
                video_info = {}

                for video_file in [video_path_1, video_path_2]:
                    video_output_dir = os.path.join(output_dir, os.path.basename(video_file).split('.')[0])
                    st.write(f"處理影片 {video_file}")
                    face_images = extract_faces_from_video(video_file, video_output_dir, frame_interval)
                    all_face_images.extend(face_images)

                    for face_path, frame_number in face_images:
                        video_name = os.path.basename(video_file).split('.')[0]
                        timestamp = frame_number // 30  # 假設 30 FPS
                        minutes = timestamp // 60
                        seconds = timestamp % 60
                        video_info[face_path] = (video_name, minutes, seconds)

                facial_encodings = compute_facial_encodings([face_path for face_path, _ in all_face_images])

                if algorithm_choice == "Chinese Whispers":
                    sorted_clusters = cluster_facial_encodings(facial_encodings, threshold)
                elif algorithm_choice == "DBSCAN":
                    encodings = list(facial_encodings.values())
                    face_paths = list(facial_encodings.keys())
                    st.info("正在尋找最佳 DBSCAN 參數...")
                    best_eps, best_min_samples = find_best_dbscan_params(encodings)
                    st.success(f"最佳 DBSCAN 參數: eps={best_eps}, min_samples={best_min_samples}")
                    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
                    labels = dbscan.fit_predict(encodings)
                    clusters = {}
                    for label, face_path in zip(labels, face_paths):
                        if label == -1:
                            continue
                        if label not in clusters:
                            clusters[label] = []
                        clusters[label].append(face_path)
                    sorted_clusters = list(clusters.values())

                if sorted_clusters:
                    st.session_state['sorted_clusters'] = sorted_clusters
                    st.session_state['video_info'] = video_info
                    st.session_state['temp_dir'] = temp_dir

                    # 保存分類結果到 output 資料夾
                    save_cluster_results_to_output(sorted_clusters, output_dir, video_info)
                    st.success("處理完成並保存分類結果！您現在可以選擇一個類別來查看。")

    with col3:
        if 'sorted_clusters' in st.session_state:
            if st.button("播放出現影片"):
                show_dialog()
@st.dialog("選擇影片時間")
def show_dialog():
    if 'sorted_clusters' in st.session_state:
        st.subheader("分類結果")
        sorted_clusters = st.session_state['sorted_clusters']
        video_info = st.session_state['video_info']
        temp_dir = st.session_state['temp_dir']

        category_options = []
        category_map = {}

        for idx, cluster in enumerate(sorted_clusters):
            video_names = set()
            for face_path in cluster:
                video_name, _, _ = video_info.get(face_path, (None, None, None))
                if video_name:
                    video_names.add(video_name)

            if len(video_names) > 1:
                category_name = f"類別 {idx} (同時出現在兩部影片)"
            else:
                category_name = f"類別 {idx}"
            
            if "同時出現在兩部影片" in category_name:
                category_options.append(category_name)
            category_map[category_name] = cluster

        if category_options:
            selected_category = st.selectbox("選擇類別", category_options)

            if selected_category:
                cluster = category_map[selected_category]
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(cluster[0], use_container_width=True)
                with col2:
                    time_threshold = st.number_input("設定秒數篩選影片時間（秒）", min_value=0, value=1)
                    video_options = []
                    video_map = {}
                    last_kept_time_per_video = {}

                    for face_path in cluster:
                        video_name, minutes, seconds = video_info.get(face_path, (None, None, None))
                        if video_name:
                            total_seconds = minutes * 60 + seconds
                            last_kept_time = last_kept_time_per_video.get(video_name, None)
                            if last_kept_time is None or abs(total_seconds - last_kept_time) > time_threshold:
                                link_text = f"所屬 {video_name}_{minutes}分{seconds}秒"
                                video_options.append(link_text)
                                video_map[link_text] = (video_name, minutes, seconds)
                                last_kept_time_per_video[video_name] = total_seconds

                    selected_video_option = st.selectbox("選擇影片", video_options)

                    if selected_video_option:
                        video_name, minutes, seconds = video_map[selected_video_option]
                        st.session_state['selected_video'] = os.path.join(temp_dir, f"{video_name}.mp4")
                        st.session_state['start_time'] = minutes * 60 + seconds

        if 'selected_video' in st.session_state and st.session_state['selected_video']:
            video_path = st.session_state['selected_video']
            start_time = st.session_state.get('start_time', 0)
            if os.path.exists(video_path):
                with open(video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes, start_time=int(start_time))
            else:
                st.error(f"影片檔案不存在: {video_path}")

        if 'sorted_clusters' in st.session_state:
            cluster_info = {}
            for idx, cluster in enumerate(sorted_clusters):
                cluster_name = f"類別 {idx}"
                cluster_info[cluster_name] = cluster

            excel_buffer = create_excel_report(cluster_info, video_info, temp_dir)
            st.download_button(
                label="下載 Excel 報告",
                data=excel_buffer,
                file_name="face_clustering_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("請先進行人臉分類。")




if __name__ == "__main__":
    main()

st.markdown('</div>', unsafe_allow_html=True)
