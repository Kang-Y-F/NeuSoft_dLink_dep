import numpy as np
import faiss
from app.db.mysql import load_all_features

class FeatureRetrieval:
    def __init__(self, dim=960):
        self.dim = dim
        self.index = None
        self.id_map = []  # index -> patient_id
        self.build_index()

    def build_index(self):
        """从 MySQL 加载特征，构建 FAISS 索引"""
        data = load_all_features()
        if not data:
            self.index = faiss.IndexFlatL2(self.dim)
            self.id_map = []
            return

        pids = list(data.keys())
        vecs = np.array(list(data.values()), dtype=np.float32).reshape(-1, self.dim)

        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(vecs)
        self.id_map = pids

    def search(self, query_vec: np.ndarray, top_k=5):
        """
        检索最相似患者
        query_vec: (960,)
        return: list[(pid, distance)]
        """
        if self.index.ntotal == 0:
            return []
        query = query_vec.astype(np.float32).reshape(1, self.dim)
        distances, indices = self.index.search(query, top_k)
        result = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.id_map):
                result.append((self.id_map[idx], float(distances[0][i])))
        return result

# 全局单例
retrieval = FeatureRetrieval()