# modules/data_loader.py
import pandas as pd

def load_menu_data(path: str):
    """从CSV文件加载菜单数据"""
    try:
        df = pd.read_csv(path)
        # 把标签字符串分割成列表
        df["标签"] = df["标签"].apply(lambda x: x.split(";") if isinstance(x, str) else [])
        return df
    except Exception as e:
        raise RuntimeError(f"加载菜单数据失败: {e}")
